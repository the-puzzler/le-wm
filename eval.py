import os

os.environ["MUJOCO_GL"] = "egl"

import json
import time
from collections import deque
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn import preprocessing
from torchvision import tv_tensors
try:
    from gymnasium.spaces import Box
except ModuleNotFoundError:  # pragma: no cover
    from gym.spaces import Box

import config as cfg
from module import ActionTranslator
from train import build_dataset, build_train_config


def resolve_object(dotted_path: str):
    module_name, attr_name = dotted_path.rsplit(".", 1)
    return getattr(import_module(module_name), attr_name)


def build_eval_config() -> dict:
    return {
        "cache_dir": str(cfg.CACHE_DIR),
        "seed": cfg.EVAL_SEED,
        "policy": cfg.EVAL_POLICY,
        "use_action_translator": cfg.EVAL_USE_ACTION_TRANSLATOR,
        "translator_checkpoint": cfg.EVAL_TRANSLATOR_CHECKPOINT,
        "world": {
            "env_name": cfg.EVAL_WORLD_ENV_NAME,
            "history_size": cfg.EVAL_WORLD_HISTORY_SIZE,
            "frame_skip": cfg.EVAL_WORLD_FRAME_SKIP,
            "task": cfg.EVAL_WORLD_TASK,
            "env_type": cfg.EVAL_WORLD_ENV_TYPE,
            "ob_type": cfg.EVAL_WORLD_OB_TYPE,
            "multiview": cfg.EVAL_WORLD_MULTIVIEW,
            "width": cfg.EVAL_WORLD_WIDTH,
            "height": cfg.EVAL_WORLD_HEIGHT,
            "visualize_info": cfg.EVAL_WORLD_VISUALIZE_INFO,
            "terminate_at_goal": cfg.EVAL_WORLD_TERMINATE_AT_GOAL,
        },
        "dataset": {
            "dataset_name": cfg.EVAL_DATASET_NAME,
            "keys_to_cache": list(cfg.EVAL_DATASET_KEYS_TO_CACHE),
        },
        "output_filename": cfg.EVAL_OUTPUT_FILENAME,
        "callables": list(cfg.EVAL_CALLABLES),
        "eval_num": cfg.EVAL_NUM,
        "eval_goal_offset_steps": cfg.EVAL_GOAL_OFFSET_STEPS,
        "eval_budget": cfg.EVAL_BUDGET,
        "eval_img_size": cfg.EVAL_IMG_SIZE,
        "plan": {
            "horizon": cfg.PLAN_HORIZON,
            "receding_horizon": cfg.PLAN_RECEDING_HORIZON,
            "action_block": cfg.PLAN_ACTION_BLOCK,
        },
        "solver": {
            "type": cfg.SOLVER_TYPE,
            "batch_size": cfg.SOLVER_BATCH_SIZE,
            "num_samples": cfg.SOLVER_NUM_SAMPLES,
            "n_steps": cfg.SOLVER_N_STEPS,
            "device": cfg.SOLVER_DEVICE,
            "seed": cfg.SOLVER_SEED,
            "var_scale": cfg.SOLVER_VAR_SCALE,
            "topk": cfg.SOLVER_TOPK,
            "action_noise": cfg.SOLVER_ACTION_NOISE,
            "optimizer_cls": cfg.SOLVER_OPTIMIZER_CLS,
            "optimizer_kwargs": cfg.SOLVER_OPTIMIZER_KWARGS,
        },
    }


def img_transform(config: dict):
    import stable_pretraining as spt
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=config["eval_img_size"]),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(config: dict):
    import stable_worldmodel as swm

    dataset_cfg = config["dataset"]
    dataset_path = Path(config["cache_dir"] or swm.data.utils.get_cache_dir())
    dataset_file = dataset_path / f"{dataset_cfg['dataset_name']}.h5"
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_file}\n"
            f"Set CACHE_DIR in config.py to the directory containing {dataset_cfg['dataset_name']}.h5."
        )

    return swm.data.HDF5Dataset(
        dataset_cfg["dataset_name"],
        keys_to_cache=dataset_cfg["keys_to_cache"],
        cache_dir=dataset_path,
    )


def build_solver(model, config: dict):
    import stable_worldmodel as swm

    solver_cfg = config["solver"]
    solver_type = solver_cfg["type"].lower()
    if solver_type == "cem":
        return swm.solver.CEMSolver(
            model=model,
            batch_size=solver_cfg["batch_size"],
            num_samples=solver_cfg["num_samples"],
            var_scale=solver_cfg["var_scale"],
            n_steps=solver_cfg["n_steps"],
            topk=solver_cfg["topk"],
            device=solver_cfg["device"],
            seed=solver_cfg["seed"],
        )
    if solver_type == "adam":
        optimizer_cls = resolve_object(solver_cfg["optimizer_cls"])
        return swm.solver.GradientSolver(
            model=model,
            n_steps=solver_cfg["n_steps"],
            batch_size=solver_cfg["batch_size"],
            num_samples=solver_cfg["num_samples"],
            action_noise=solver_cfg["action_noise"],
            device=solver_cfg["device"],
            seed=solver_cfg["seed"],
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=solver_cfg["optimizer_kwargs"],
        )
    raise ValueError(f"Unsupported solver type: {solver_cfg['type']}")


def load_cost_model(policy_name: str):
    import torch
    import stable_worldmodel as swm

    policy_path = Path(policy_name).expanduser()
    if policy_path.suffix == ".ckpt" and policy_path.exists():
        model = torch.load(policy_path, map_location="cpu", weights_only=False)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        results_path = policy_path.parent
        return model, results_path

    model = swm.policy.AutoCostModel(policy_name)
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    results_path = Path(swm.data.utils.get_cache_dir(), policy_name).parent
    return model, results_path


def find_latest_translator_checkpoint() -> Path:
    runs_dir = Path(cfg.RUNS_DIR)
    weight_candidates = sorted(
        runs_dir.glob("*/*_action_translator_weights.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if weight_candidates:
        return weight_candidates[-1]

    object_candidates = sorted(
        runs_dir.glob("*/*_action_translator_object.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if object_candidates:
        return object_candidates[-1]

    raise FileNotFoundError(f"No action translator checkpoints found under: {runs_dir}")


def get_action_chunk_stats(train_config: dict):
    import torch

    dataset = build_dataset(train_config)
    actions = torch.from_numpy(np.array(dataset.get_col_data("action")))
    actions = actions[~torch.isnan(actions).any(dim=1)]
    mean = actions.mean(0, keepdim=True).float()
    std = actions.std(0, keepdim=True).float()
    frameskip = int(train_config["dataset"]["frameskip"])
    mean = mean.repeat(1, frameskip).reshape(1, -1)
    std = std.repeat(1, frameskip).reshape(1, -1)
    return mean, std


def load_action_translator(device, train_config: dict, config: dict):
    import torch

    checkpoint_path = (
        Path(config["translator_checkpoint"])
        if config["translator_checkpoint"]
        else find_latest_translator_checkpoint()
    )
    action_dim = train_config["dataset"]["frameskip"] * train_config["wm"]["action_dim"]
    translator = ActionTranslator(
        num_codes=train_config["codebook"]["num_codes"],
        state_dim=train_config["wm"]["embed_dim"],
        action_dim=action_dim,
        hidden_dim=cfg.TRANSLATOR_HIDDEN_DIM,
    ).to(device)

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        translator.load_state_dict(payload)
    else:
        translator.load_state_dict(payload.state_dict())
    translator.eval()
    translator.requires_grad_(False)
    return translator, checkpoint_path


class LatentActionCostModel:
    def __init__(
        self,
        *,
        model,
        translator,
        history_size: int,
        num_codes: int,
        real_action_block: int,
        real_action_dim: int,
        action_chunk_mean,
        action_chunk_std,
    ):
        self.model = model
        self.translator = translator
        self.history_size = history_size
        self.num_codes = num_codes
        self.real_action_block = real_action_block
        self.real_action_dim = real_action_dim
        self.device = next(model.parameters()).device
        self.action_chunk_mean = action_chunk_mean.to(self.device)
        self.action_chunk_std = action_chunk_std.to(self.device)

    def _move_info(self, info_dict: dict):
        import torch

        moved = {}
        for key, value in info_dict.items():
                moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _select_sequence_view(self, value):
        ndim = getattr(value, "ndim", 0)
        if ndim >= 6:
            return value[:, 0]
        return value

    def _encode_goal(self, info_dict: dict):
        goal = {
            k: self._select_sequence_view(v)
            for k, v in info_dict.items()
            if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2
        }
        goal["pixels"] = goal["goal"]
        if getattr(goal["pixels"], "ndim", 0) == 4:
            goal["pixels"] = goal["pixels"][:, None]
        for key in list(goal.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        return self.model.encode(goal)["emb"][:, -1]

    def _encode_init(self, info_dict: dict):
        init = {
            k: self._select_sequence_view(v)
            for k, v in info_dict.items()
            if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2
        }
        if "pixels" in init and getattr(init["pixels"], "ndim", 0) == 4:
            init["pixels"] = init["pixels"][:, None]
        init.pop("action", None)
        return self.model.encode(init)["emb"]

    def _quantize_candidates(self, action_candidates):
        import torch

        if action_candidates.shape[-1] > 1:
            action_candidates = action_candidates[..., :1]
        code_indices = action_candidates.round().clamp(0, self.num_codes - 1)
        return code_indices.to(dtype=torch.long).squeeze(-1)

    def _rollout(self, info_dict: dict, code_indices, decode_actions: bool):
        import torch

        batch_size, num_samples, horizon = code_indices.shape
        init_emb = self._encode_init(info_dict)
        emb = init_emb.unsqueeze(1).expand(batch_size, num_samples, -1, -1)
        emb = emb.reshape(batch_size * num_samples, init_emb.size(1), init_emb.size(2)).clone()
        flat_codes = code_indices.reshape(batch_size * num_samples, horizon)

        decoded_chunks = []
        for t in range(horizon):
            current_code = flat_codes[:, t]
            if decode_actions:
                decoded = self.translator(emb[:, -1], current_code)
                decoded = decoded * self.action_chunk_std + self.action_chunk_mean
                decoded_chunks.append(decoded.view(-1, self.real_action_block, self.real_action_dim))

            code_hist = flat_codes[:, : t + 1]
            act_emb = self.model.quantizer.codebook(code_hist)
            seq_len = min(self.history_size, emb.size(1), act_emb.size(1))
            pred_emb = self.model.predict(emb[:, -seq_len:], act_emb[:, -seq_len:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

        final_emb = emb[:, -1].view(batch_size, num_samples, -1)
        decoded_plan = None
        if decode_actions:
            decoded_plan = torch.stack(decoded_chunks, dim=1)
            decoded_plan = decoded_plan.view(
                batch_size,
                num_samples,
                horizon,
                self.real_action_block,
                self.real_action_dim,
            )
        return final_emb, decoded_plan

    def get_cost(self, info_dict: dict, action_candidates):
        import torch
        import torch.nn.functional as F

        info_dict = self._move_info(info_dict)
        code_indices = self._quantize_candidates(action_candidates)
        goal_emb = self._encode_goal(info_dict).unsqueeze(1)
        final_emb, _ = self._rollout(info_dict, code_indices, decode_actions=False)
        goal_emb = goal_emb.expand_as(final_emb)
        return F.mse_loss(final_emb, goal_emb.detach(), reduction="none").sum(dim=-1)

    def decode_action_plan(self, info_dict: dict, latent_plan):
        import torch

        info_dict = self._move_info(info_dict)
        if latent_plan.ndim == 3:
            latent_plan = latent_plan.unsqueeze(1)
        if not torch.is_tensor(latent_plan):
            latent_plan = torch.as_tensor(latent_plan, device=self.device)
        code_indices = self._quantize_candidates(latent_plan.to(self.device))
        _, decoded_plan = self._rollout(info_dict, code_indices, decode_actions=True)
        return decoded_plan[:, 0]


class LatentActionWorldModelPolicy:
    def __init__(self, solver, config, model, process=None, transform=None):
        self.env = None
        self.type = "latent_world_model"
        self.cfg = config
        self.solver = solver
        self.model = model
        self.process = process or {}
        self.transform = transform or {}
        self.real_action_block = model.real_action_block
        self.real_action_dim = model.real_action_dim
        self._action_buffer = None
        self._next_init = None

    @property
    def flatten_receding_horizon(self) -> int:
        return self.cfg.receding_horizon * self.real_action_block

    def set_env(self, env):
        self.env = env
        self._action_buffer = deque(maxlen=self.flatten_receding_horizon)
        latent_action_space = Box(
            low=0.0,
            high=float(self.model.num_codes - 1),
            shape=(env.num_envs, 1),
            dtype=np.float32,
        )
        self.solver.configure(
            action_space=latent_action_space,
            n_envs=getattr(env, "num_envs", 1),
            config=self.cfg,
        )

    def _prepare_info(self, info_dict: dict):
        import torch

        for key, value in info_dict.items():
            is_numpy = isinstance(value, (np.ndarray, np.generic))

            if key in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{key}' in process, got {type(value)}"
                    )
                shape = value.shape
                if len(shape) > 2:
                    value = value.reshape(-1, *shape[2:])
                value = self.process[key].transform(value)
                value = value.reshape(shape)

            if key in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(value):
                    if value.ndim > 2:
                        shape = value.shape
                        value = value.reshape(-1, *shape[2:])
                if key.startswith("pixels") or key.startswith("goal"):
                    if is_numpy:
                        value = np.transpose(value, (0, 3, 1, 2))
                    else:
                        value = value.permute(0, 3, 1, 2)
                value = torch.stack(
                    [self.transform[key](tv_tensors.Image(x)) for x in value]
                )
                is_numpy = isinstance(value, (np.ndarray, np.generic))
                if shape is not None:
                    value = value.reshape(*shape[:2], *value.shape[1:])

            if is_numpy and value.dtype.kind not in "USO":
                value = torch.from_numpy(value)
            info_dict[key] = value
        return info_dict

    def get_action(self, info_dict: dict, **kwargs):
        assert self.env is not None, "Environment not set for policy"
        info_dict = self._prepare_info(info_dict)

        if len(self._action_buffer) == 0:
            outputs = self.solver(info_dict, init_action=self._next_init)
            latent_actions = outputs["actions"]
            keep_horizon = self.cfg.receding_horizon
            plan = latent_actions[:, :keep_horizon]
            rest = latent_actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None

            decoded = self.model.decode_action_plan(info_dict, plan)
            decoded = decoded.reshape(
                self.env.num_envs,
                self.flatten_receding_horizon,
                self.real_action_dim,
            )
            self._action_buffer.extend(decoded.transpose(0, 1).cpu())

        action = self._action_buffer.popleft().numpy()
        return action.reshape(*self.env.action_space.shape)


def main():
    import torch
    import stable_worldmodel as swm

    config = build_eval_config()
    if config["plan"]["horizon"] * config["plan"]["action_block"] > config["eval_budget"]:
        raise ValueError("Planning horizon must be smaller than or equal to eval_budget")

    world_kwargs = {
        key: value for key, value in config["world"].items() if value is not None
    }
    world_kwargs["num_envs"] = config["eval_num"]
    world_kwargs["max_episode_steps"] = 2 * config["eval_budget"]
    world = swm.World(**world_kwargs, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(config),
        "goal": img_transform(config),
    }

    dataset = get_dataset(config)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in config["dataset"]["keys_to_cache"]:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor

    train_config = build_train_config()
    policy_name = config["policy"]
    if policy_name != "random":
        model, results_path = load_cost_model(policy_name)
        if config["use_action_translator"]:
            if not train_config["wm"]["use_learned_actions"]:
                raise ValueError("Action translator eval requires USE_LEARNED_ACTIONS = True in config.py.")
            translator, translator_checkpoint = load_action_translator(
                device=torch.device("cuda"),
                train_config=train_config,
                config=config,
            )
            action_chunk_mean, action_chunk_std = get_action_chunk_stats(train_config)
            latent_model = LatentActionCostModel(
                model=model,
                translator=translator,
                history_size=train_config["wm"]["history_size"],
                num_codes=train_config["codebook"]["num_codes"],
                real_action_block=train_config["dataset"]["frameskip"],
                real_action_dim=train_config["wm"]["action_dim"],
                action_chunk_mean=action_chunk_mean,
                action_chunk_std=action_chunk_std,
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
            solver = build_solver(model=latent_model, config=latent_solver_config)
            policy = LatentActionWorldModelPolicy(
                solver=solver,
                config=latent_plan_config,
                model=latent_model,
                process={k: v for k, v in process.items() if k != "action"},
                transform=transform,
            )
            print(f"Using latent-action translator from: {translator_checkpoint}")
        else:
            plan_config = swm.PlanConfig(**config["plan"])
            solver = build_solver(model=model, config=config)
            policy = swm.policy.WorldModelPolicy(
                solver=solver,
                config=plan_config,
                process=process,
                transform=transform,
            )
    else:
        policy = swm.policy.RandomPolicy()
        results_path = Path(__file__).parent

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - config["eval_goal_offset_steps"] - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    generator = np.random.default_rng(config["seed"])
    random_episode_indices = generator.choice(
        len(valid_indices) - 1, size=config["eval_num"], replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < config["eval_num"]:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=config["eval_goal_offset_steps"],
        eval_budget=config["eval_budget"],
        episodes_idx=eval_episodes.tolist(),
        callables=config["callables"],
        video_path=results_path,
    )
    end_time = time.time()

    print(metrics)

    output_path = results_path / config["output_filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    main()
