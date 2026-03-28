import os

os.environ["MUJOCO_GL"] = "egl"

import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn import preprocessing

import config as cfg


def resolve_object(dotted_path: str):
    module_name, attr_name = dotted_path.rsplit(".", 1)
    return getattr(import_module(module_name), attr_name)


def build_eval_config() -> dict:
    return {
        "cache_dir": str(cfg.CACHE_DIR),
        "seed": cfg.EVAL_SEED,
        "policy": cfg.EVAL_POLICY,
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


def main():
    import stable_worldmodel as swm

    config = build_eval_config()
    if config["plan"]["horizon"] * config["plan"]["action_block"] > config["eval_budget"]:
        raise ValueError("Planning horizon must be smaller than or equal to eval_budget")

    world_kwargs = dict(config["world"])
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

    policy_name = config["policy"]
    if policy_name != "random":
        model = swm.policy.AutoCostModel(policy_name)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        plan_config = swm.PlanConfig(**config["plan"])
        solver = build_solver(model=model, config=config)
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan_config,
            process=process,
            transform=transform,
        )
        results_path = Path(swm.data.utils.get_cache_dir(), policy_name).parent
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
