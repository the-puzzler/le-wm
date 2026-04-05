from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from sklearn import preprocessing
from torchvision.utils import make_grid, save_image

import config as cfg
import eval as eval_runner


DEFAULT_DECODER_CHECKPOINT = Path(
    "/workspace/le-wm/runs/20260403-171752-decoder/lewm_decoder_epoch_3_object.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode saved eval planning latents into imagined trajectory visualizations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(cfg.RUNS_DIR),
        help="Task folder, eval run folder, or root directory to scan for task_* folders.",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=DEFAULT_DECODER_CHECKPOINT,
        help="Visual decoder object checkpoint used to decode planning latents.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for decoding latents.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="FPS for saved imagined trajectory videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously rendered imagined outputs.",
    )
    parser.add_argument(
        "--trim-successes",
        action="store_true",
        help="For successful tasks, trim outputs at the first true success step by replaying the eval policy.",
    )
    return parser.parse_args()


def denormalize_pixels(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def load_decoder(checkpoint_path: Path, device: torch.device):
    decoder = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder = decoder.to(device)
    decoder.eval()
    decoder.requires_grad_(False)
    return decoder


def build_success_runtime():
    import stable_worldmodel as swm

    config = eval_runner.build_eval_config()
    world_kwargs = {
        key: value for key, value in config["world"].items() if value is not None
    }
    world_kwargs["num_envs"] = 1
    world_kwargs["max_episode_steps"] = 2 * config["eval_budget"]
    world = swm.World(**world_kwargs, image_shape=(224, 224))

    dataset = eval_runner.get_dataset(config)
    process = {}
    for col in config["dataset"]["keys_to_cache"]:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor

    train_config = eval_runner.build_train_config()
    transform = {
        "pixels": eval_runner.img_transform(config),
        "goal": eval_runner.img_transform(config),
    }

    policy_name = config["policy"]
    if policy_name != "random":
        model, _ = eval_runner.load_cost_model(policy_name)
        if config["use_action_translator"]:
            translator, _ = eval_runner.load_action_translator(
                device=torch.device("cuda"),
                train_config=train_config,
                config=config,
            )
            action_chunk_mean, action_chunk_std = eval_runner.get_action_chunk_stats(train_config)
            latent_model = eval_runner.LatentActionCostModel(
                model=model,
                translator=translator,
                history_size=train_config["wm"]["history_size"],
                num_codes=train_config["codebook"]["num_codes"],
                real_action_block=train_config["dataset"]["frameskip"],
                real_action_dim=train_config["wm"]["action_dim"],
                action_chunk_mean=action_chunk_mean,
                action_chunk_std=action_chunk_std,
                translator_action_scale=config["translator_action_scale"],
            )
            latent_plan_config = swm.PlanConfig(
                horizon=config["plan"]["horizon"],
                receding_horizon=config["plan"]["receding_horizon"],
                history_len=1,
                action_block=1,
                warm_start=True,
            )
            latent_solver_config = dict(config)
            latent_solver_config["plan"] = {
                "horizon": latent_plan_config.horizon,
                "receding_horizon": latent_plan_config.receding_horizon,
                "action_block": latent_plan_config.action_block,
            }
            solver = eval_runner.build_solver(model=latent_model, config=latent_solver_config)
            policy = eval_runner.LatentActionWorldModelPolicy(
                solver=solver,
                config=latent_plan_config,
                model=latent_model,
                process={k: v for k, v in process.items() if k != "action"},
                transform=transform,
            )
        else:
            plan_config = swm.PlanConfig(**config["plan"])
            solver = eval_runner.build_solver(model=model, config=config)
            policy = swm.policy.WorldModelPolicy(
                solver=solver,
                config=plan_config,
                process=process,
                transform=transform,
            )
    else:
        policy = swm.policy.RandomPolicy()

    world.set_policy(policy)
    return {
        "config": config,
        "dataset": dataset,
        "world": world,
    }


def find_task_dirs(root: Path) -> list[Path]:
    if root.is_file():
        root = root.parent

    if (root / "planning_artifacts.pt").exists() or (root / "planning_latents.pt").exists():
        return [root]

    task_dirs = []
    for candidate in sorted(root.rglob("task_*")):
        if candidate.is_dir() and (
            (candidate / "planning_artifacts.pt").exists()
            or (candidate / "planning_latents.pt").exists()
        ):
            task_dirs.append(candidate)
    return task_dirs


def decode_latents(decoder, latents: torch.Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    frames = []
    with torch.no_grad():
        for start in range(0, latents.size(0), batch_size):
            chunk = latents[start : start + batch_size].to(device)
            recon = decoder(chunk)
            recon = denormalize_pixels(recon.float()).mul(255).round().byte()
            recon = recon.permute(0, 2, 3, 1).cpu().numpy()
            frames.append(recon)
    return np.concatenate(frames, axis=0)


def load_replans(task_dir: Path) -> list[dict]:
    artifact_path = (
        task_dir / "planning_artifacts.pt"
        if (task_dir / "planning_artifacts.pt").exists()
        else task_dir / "planning_latents.pt"
    )
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    replans = payload.get("replans", [])
    if not replans:
        raise ValueError(f"No replans found in {artifact_path}")
    return replans


def executed_horizon(record: dict) -> int:
    value = record.get("executed_latent_action_plan")
    if torch.is_tensor(value):
        return int(value.shape[0])
    return 0


def get_action_block(replans: list[dict], task_dir: Path) -> int:
    artifact_path = task_dir / "planning_artifacts.pt"
    if artifact_path.exists():
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        value = int(payload.get("real_action_block", 1))
        return max(1, value)
    return 1


def build_executed_latent_sequence(replans: list[dict]) -> torch.Tensor:
    segments = []
    for record in replans:
        state_latents = record["state_plan_latents"]
        executed_len = executed_horizon(record)
        if executed_len <= 0:
            continue
        segments.append(state_latents[:executed_len])
    if not segments:
        raise ValueError("No executed latent segments found to render.")
    return torch.cat(segments, dim=0)


def build_replan_grid_frames(replans: list[dict], decoded_replans: list[np.ndarray]) -> torch.Tensor:
    rows = []
    for record, decoded in zip(replans, decoded_replans, strict=True):
        horizon = decoded.shape[0]
        label_pad = torch.zeros(1, 3, decoded.shape[1], decoded.shape[2], dtype=torch.float32)
        frames = torch.from_numpy(decoded).permute(0, 3, 1, 2).float() / 255.0
        rows.append(torch.cat([label_pad, frames], dim=0))

    if not rows:
        raise ValueError("No decoded replans available for visualization grid.")

    max_cols = max(row.size(0) for row in rows)
    padded_rows = []
    for row in rows:
        if row.size(0) < max_cols:
            pad = torch.zeros(max_cols - row.size(0), *row.shape[1:], dtype=row.dtype)
            row = torch.cat([row, pad], dim=0)
        padded_rows.append(row)

    grid = make_grid(torch.cat(padded_rows, dim=0), nrow=max_cols, padding=2)
    return grid


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def load_real_rollout_frames(task_dir: Path) -> np.ndarray | None:
    rollout_path = task_dir / "rollout.mp4"
    if not rollout_path.exists():
        return None
    frames = imageio.mimread(rollout_path)
    if not frames:
        return None
    video = np.stack(frames)
    return video


def resize_nearest(frames: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames

    y_idx = np.clip(
        np.round(np.linspace(0, frames.shape[1] - 1, target_h)).astype(np.int64),
        0,
        frames.shape[1] - 1,
    )
    x_idx = np.clip(
        np.round(np.linspace(0, frames.shape[2] - 1, target_w)).astype(np.int64),
        0,
        frames.shape[2] - 1,
    )
    return frames[:, y_idx][:, :, x_idx]


def build_comparison_video(imagined_frames: np.ndarray, real_frames: np.ndarray) -> np.ndarray:
    imagined_frames = resize_nearest(imagined_frames, real_frames.shape[1:3])
    frame_count = min(len(imagined_frames), len(real_frames))
    return np.concatenate(
        [imagined_frames[:frame_count], real_frames[:frame_count]],
        axis=2,
    )


def expand_imagined_frames(imagined_frames: np.ndarray, action_block: int, target_len: int | None) -> np.ndarray:
    expanded = np.repeat(imagined_frames, repeats=max(1, action_block), axis=0)
    if target_len is None:
        return expanded
    if len(expanded) >= target_len:
        return expanded[:target_len]
    if len(expanded) == 0:
        raise ValueError("Cannot expand an empty imagined trajectory.")
    pad = np.repeat(expanded[-1:], repeats=target_len - len(expanded), axis=0)
    return np.concatenate([expanded, pad], axis=0)


def task_succeeded(task_dir: Path) -> bool:
    name = task_dir.name
    if name.endswith("_true"):
        return True
    info_path = task_dir / "task_info.json"
    if info_path.exists():
        payload = torch.load(info_path, map_location="cpu", weights_only=False) if info_path.suffix == ".pt" else None
        if payload is not None:
            return bool(payload.get("success"))
        import json

        with info_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("success"))
    return False


def load_task_info_json(task_dir: Path) -> dict:
    import json

    with (task_dir / "task_info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_success_frame_limit(task_dir: Path, runtime: dict) -> int | None:
    if not task_succeeded(task_dir):
        return None

    import stable_worldmodel as swm  # noqa: F401

    config = runtime["config"]
    dataset = runtime["dataset"]
    world = runtime["world"]
    policy = world.policy
    if hasattr(policy, "_action_buffer") and policy._action_buffer is not None:
        policy._action_buffer.clear()
    if hasattr(policy, "action_buffer") and policy.action_buffer is not None:
        policy.action_buffer.clear()
    if hasattr(policy, "_next_init"):
        policy._next_init = None
    if hasattr(policy, "_env_steps") and policy._env_steps is not None:
        policy._env_steps = np.zeros(world.num_envs, dtype=np.int64)
    if hasattr(policy, "_task_artifacts") and policy._task_artifacts is not None:
        policy._task_artifacts = [[] for _ in range(world.num_envs)]

    task_info = load_task_info_json(task_dir)
    episode_idx = int(task_info["episode_idx"])
    start_step = int(task_info["start_step"])
    goal_offset_steps = int(config["eval_goal_offset_steps"])
    eval_budget = int(config["eval_budget"])

    ep_idx_arr = np.array([episode_idx])
    start_steps_arr = np.array([start_step])
    end_steps = start_steps_arr + goal_offset_steps
    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
    columns = dataset.column_names

    init_step_per_env: dict[str, list] = defaultdict(list)
    goal_step_per_env: dict[str, list] = defaultdict(list)
    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)

            if not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue

            init_data = ep[col][0]
            goal_data = ep[col][-1]
            init_data = init_data.numpy() if isinstance(init_data, torch.Tensor) else init_data
            goal_data = goal_data.numpy() if isinstance(goal_data, torch.Tensor) else goal_data
            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)

    init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}
    goal_step = {}
    for key, value in goal_step_per_env.items():
        out_key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[out_key] = np.stack(value)

    seeds = init_step.get("seed")
    vkey = "variation."
    variations_dict = {
        k.removeprefix(vkey): v
        for k, v in init_step.items()
        if k.startswith(vkey)
    }
    options = [{}]
    if len(variations_dict) > 0:
        options[0]["variation"] = list(variations_dict.keys())
        options[0]["variation_values"] = {k: v[0] for k, v in variations_dict.items()}

    init_step.update(deepcopy(goal_step))
    world.reset(seed=seeds, options=options)

    callables = config["callables"] or []
    for env in world.envs.unwrapped.envs:
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for args_name, args_data in args.items():
                value = args_data.get("value", None)
                is_in_dataset = args_data.get("in_dataset", True)
                if is_in_dataset:
                    if value not in init_step:
                        continue
                    prepared_args[args_name] = deepcopy(init_step[value][0])
                else:
                    prepared_args[args_name] = args_data.get("value")
            method(**prepared_args)

    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
        for k, v in init_step.items()
    }
    goal_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
        for k, v in goal_step.items()
    }
    world.infos.update(deepcopy(init_step))
    world.infos.update(deepcopy(goal_step))

    for step_idx in range(eval_budget):
        world.infos.update(deepcopy(goal_step))
        world.step()
        if bool(world.terminateds[0]):
            return min(step_idx + 2, eval_budget)
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))

    return None


def render_task(
    decoder,
    task_dir: Path,
    device: torch.device,
    fps: int,
    overwrite: bool,
    success_runtime: dict | None = None,
) -> None:
    imagined_video_path = task_dir / "imagined_trajectory.mp4"
    comparison_video_path = task_dir / "imagined_vs_real.mp4"
    grid_path = task_dir / "imagined_replans.png"
    summary_path = task_dir / "imagined_summary.pt"

    if (
        not overwrite
        and imagined_video_path.exists()
        and grid_path.exists()
        and summary_path.exists()
    ):
        return

    replans = load_replans(task_dir)
    action_block = get_action_block(replans, task_dir)
    executed_latents = build_executed_latent_sequence(replans)
    imagined_frames = decode_latents(decoder, executed_latents, device)

    decoded_replans = []
    for record in replans:
        decoded_replans.append(decode_latents(decoder, record["state_plan_latents"], device))

    real_frames = load_real_rollout_frames(task_dir)
    imagined_video_frames = expand_imagined_frames(
        imagined_frames,
        action_block=action_block,
        target_len=None if real_frames is None else len(real_frames),
    )

    success_frame_limit = None
    if success_runtime is not None:
        success_frame_limit = compute_success_frame_limit(task_dir, success_runtime)
        if success_frame_limit is not None:
            imagined_video_frames = imagined_video_frames[:success_frame_limit]
            if real_frames is not None:
                real_frames = real_frames[:success_frame_limit]

    write_video(imagined_video_path, imagined_video_frames, fps=fps)

    if real_frames is not None:
        comparison_frames = build_comparison_video(imagined_video_frames, real_frames)
        write_video(comparison_video_path, comparison_frames, fps=fps)

    grid = build_replan_grid_frames(replans, decoded_replans)
    save_image(grid, grid_path)

    torch.save(
        {
            "executed_state_plan_latents": executed_latents,
            "imagined_frame_count": int(imagined_video_frames.shape[0]),
            "latent_frame_count": int(imagined_frames.shape[0]),
            "real_action_block": int(action_block),
            "success_frame_limit": success_frame_limit,
            "replan_horizons": [int(record["state_plan_latents"].shape[0]) for record in replans],
            "executed_horizons": [executed_horizon(record) for record in replans],
        },
        summary_path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    decoder = load_decoder(args.decoder_checkpoint, device)
    success_runtime = build_success_runtime() if args.trim_successes else None

    task_dirs = find_task_dirs(args.input)
    if not task_dirs:
        raise FileNotFoundError(f"No task folders with planning artifacts found under: {args.input}")

    print(f"Found {len(task_dirs)} task folders.")
    for index, task_dir in enumerate(task_dirs, start=1):
        print(f"[{index}/{len(task_dirs)}] Rendering {task_dir}")
        render_task(
            decoder,
            task_dir,
            device,
            fps=args.fps,
            overwrite=args.overwrite,
            success_runtime=success_runtime,
        )


if __name__ == "__main__":
    main()
