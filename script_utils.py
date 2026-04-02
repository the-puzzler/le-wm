from __future__ import annotations

from pathlib import Path

import torch.nn as nn


def find_latest_object_checkpoint(runs_dir: str | Path, *, exclude_name_substrings: tuple[str, ...] = ()) -> Path:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    candidates = sorted(
        [
            path
            for path in runs_dir.glob("*/*_object.ckpt")
            if not any(token in path.name for token in exclude_name_substrings)
        ],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints found under: {runs_dir}")
    return candidates[-1]


def set_source_model_mode(model, mode: str, *, mode_label: str = "SOURCE_MODEL_MODE") -> None:
    if mode == "train":
        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unsupported {mode_label}: {mode}")


def freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
