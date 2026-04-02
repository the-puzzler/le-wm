from __future__ import annotations

from pathlib import Path
import json
import math
import sys
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm.auto import tqdm

import config as cfg
from script_utils import find_latest_object_checkpoint, set_source_model_mode
from train import build_dataset, build_train_config, move_batch_to_device, resolve_device


def create_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.RUNS_DIR) / f"{timestamp}-{cfg.TRANSLATOR_ANALYSIS_RUN_NAME}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_loader(train_config: dict):
    dataset = build_dataset(train_config)
    sample_count = min(len(dataset), int(cfg.TRANSLATOR_ANALYSIS_SAMPLES))
    generator = torch.Generator().manual_seed(cfg.SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=cfg.TRANSLATOR_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.TRANSLATOR_NUM_WORKERS,
        persistent_workers=cfg.TRANSLATOR_PERSISTENT_WORKERS and cfg.TRANSLATOR_NUM_WORKERS > 0,
        prefetch_factor=cfg.TRANSLATOR_PREFETCH_FACTOR if cfg.TRANSLATOR_NUM_WORKERS > 0 else None,
        pin_memory=cfg.TRANSLATOR_PIN_MEMORY,
    )

def summarize_codebook(source_model, loader, device, train_config: dict):
    ctx_len = train_config["wm"]["history_size"]
    num_codes = train_config["codebook"]["num_codes"]
    action_dim = train_config["dataset"]["frameskip"] * train_config["wm"]["action_dim"]

    counts = torch.zeros(num_codes, dtype=torch.long)
    sum_actions = torch.zeros(num_codes, action_dim)
    sum_sq_actions = torch.zeros(num_codes, action_dim)
    sum_dirs = torch.zeros(num_codes, 2)
    sum_dir_norm = torch.zeros(num_codes)
    octant_counts = torch.zeros(num_codes, 8, dtype=torch.long)
    sum_unit_dirs = torch.zeros(num_codes, 2)
    sum_nonzero_dirs = torch.zeros(num_codes, dtype=torch.long)

    with torch.no_grad():
        for batch in tqdm(loader, desc="analyze codes", leave=False):
            batch = move_batch_to_device(batch, device)
            batch = dict(batch)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

            output = source_model.encode(batch)
            emb = output["emb"]
            code_features = source_model.infer_action_codes(emb[:, : ctx_len + train_config["wm"]["num_preds"]])
            vq_output = source_model.quantize_action_codes(code_features[:, :ctx_len])

            code_indices = vq_output["indices"].reshape(-1).detach().cpu()
            actions = batch["action"][:, :ctx_len].reshape(-1, action_dim).detach().cpu()
            dirs = actions.view(actions.size(0), -1, 2).mean(dim=1)
            dir_norms = dirs.norm(dim=1)
            unit_dirs = torch.zeros_like(dirs)
            nonzero_mask = dir_norms > 1e-6
            unit_dirs[nonzero_mask] = dirs[nonzero_mask] / dir_norms[nonzero_mask].unsqueeze(-1)
            angles = torch.rad2deg(torch.atan2(dirs[:, 1], dirs[:, 0]))
            octant_indices = (((angles + 22.5) % 360) // 45).to(torch.long)

            for code in range(num_codes):
                mask = code_indices == code
                if not mask.any():
                    continue
                code_actions = actions[mask]
                code_dirs = dirs[mask]
                code_dir_norms = dir_norms[mask]
                code_unit_dirs = unit_dirs[mask]
                code_nonzero = nonzero_mask[mask]
                code_octants = octant_indices[mask]

                counts[code] += int(mask.sum())
                sum_actions[code] += code_actions.sum(dim=0)
                sum_sq_actions[code] += code_actions.pow(2).sum(dim=0)
                sum_dirs[code] += code_dirs.sum(dim=0)
                sum_dir_norm[code] += code_dir_norms.sum()
                if code_nonzero.any():
                    sum_unit_dirs[code] += code_unit_dirs[code_nonzero].sum(dim=0)
                    sum_nonzero_dirs[code] += int(code_nonzero.sum())
                octant_counts[code] += torch.bincount(code_octants, minlength=8)

    summaries = []
    total = int(counts.sum().item())
    octants = [
        "right",
        "up-right",
        "up",
        "up-left",
        "left",
        "down-left",
        "down",
        "down-right",
    ]
    for code in range(num_codes):
        count = int(counts[code].item())
        if count == 0:
            summaries.append(
                {
                    "code": code,
                    "count": 0,
                    "usage_fraction": 0.0,
                    "mean_action": None,
                    "std_action": None,
                    "mean_direction": None,
                    "mean_unit_direction": None,
                    "mean_direction_norm": None,
                    "octant": None,
                    "dominant_octant": None,
                    "dominant_octant_fraction": None,
                    "octant_histogram": None,
                }
            )
            continue

        mean_action = sum_actions[code] / count
        mean_sq = sum_sq_actions[code] / count
        std_action = (mean_sq - mean_action.pow(2)).clamp_min(0.0).sqrt()
        mean_direction = sum_dirs[code] / count
        mean_direction_norm = float(sum_dir_norm[code].item() / count)
        nonzero_count = int(sum_nonzero_dirs[code].item())
        mean_unit_direction = None
        if nonzero_count > 0:
            mean_unit_direction = (sum_unit_dirs[code] / nonzero_count).tolist()

        angle = math.degrees(math.atan2(float(mean_direction[1]), float(mean_direction[0])))
        octant_idx = int(((angle + 22.5) % 360) // 45)
        octant = octants[octant_idx]
        dominant_octant_idx = int(octant_counts[code].argmax().item())
        dominant_octant_count = int(octant_counts[code, dominant_octant_idx].item())
        dominant_octant = octants[dominant_octant_idx]
        octant_histogram = {
            octants[i]: int(octant_counts[code, i].item())
            for i in range(len(octants))
        }

        summaries.append(
            {
                "code": code,
                "count": count,
                "usage_fraction": count / max(1, total),
                "mean_action": mean_action.tolist(),
                "std_action": std_action.tolist(),
                "mean_direction": mean_direction.tolist(),
                "mean_unit_direction": mean_unit_direction,
                "mean_direction_norm": mean_direction_norm,
                "octant": octant,
                "dominant_octant": dominant_octant,
                "dominant_octant_fraction": dominant_octant_count / count,
                "octant_histogram": octant_histogram,
            }
        )

    return summaries


def main():
    train_config = build_train_config()
    if not train_config["wm"]["use_learned_actions"]:
        raise ValueError("This analysis only applies when USE_LEARNED_ACTIONS = True.")

    checkpoint_path = (
        Path(cfg.TRANSLATOR_SOURCE_CHECKPOINT)
        if cfg.TRANSLATOR_SOURCE_CHECKPOINT
        else find_latest_object_checkpoint(cfg.RUNS_DIR, exclude_name_substrings=("_decoder", "_translator"))
    )
    device = resolve_device(train_config)
    source_model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device)
    for param in source_model.parameters():
        param.requires_grad_(False)
    set_source_model_mode(
        source_model,
        cfg.TRANSLATOR_SOURCE_MODEL_MODE,
        mode_label="TRANSLATOR_SOURCE_MODEL_MODE",
    )

    loader = build_loader(train_config)
    summaries = summarize_codebook(source_model, loader, device, train_config)
    run_dir = create_run_dir()

    output = {
        "checkpoint": str(checkpoint_path),
        "dataset": train_config["dataset"]["name"],
        "num_samples": min(len(loader.dataset), int(cfg.TRANSLATOR_ANALYSIS_SAMPLES)),
        "history_size": train_config["wm"]["history_size"],
        "num_codes": train_config["codebook"]["num_codes"],
        "codes": summaries,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nsaved summary to {summary_path}")


if __name__ == "__main__":
    main()
