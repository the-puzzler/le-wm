from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

import config as cfg
from module import SIGReg
from train import build_train_config, build_dataset, lejepa_forward, move_batch_to_device, resolve_amp, resolve_device


# Optional override. Leave as None to auto-pick the newest object checkpoint
# from the newest run under RUNS_DIR.
CHECKPOINT_PATH = None


def find_latest_checkpoint() -> Path:
    runs_dir = Path(cfg.RUNS_DIR)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    candidates = sorted(runs_dir.glob("*/*_object.ckpt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints found under: {runs_dir}")
    return candidates[-1]


def build_val_loader(config: dict):
    import stable_pretraining as spt

    dataset = build_dataset(config)
    generator = torch.Generator().manual_seed(config["seed"])
    _, val_set = spt.data.random_split(
        dataset,
        lengths=[config["train_split"], 1 - config["train_split"]],
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": config["loader"]["batch_size"],
        "num_workers": config["loader"]["num_workers"],
        "persistent_workers": config["loader"]["persistent_workers"] and config["loader"]["num_workers"] > 0,
        "pin_memory": config["loader"]["pin_memory"],
    }
    if config["loader"]["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = config["loader"]["prefetch_factor"]

    return torch.utils.data.DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )


def evaluate_once(model, sigreg, loader, device, amp_dtype, config: dict, *, train_mode: bool) -> dict[str, float]:
    if train_mode:
        model.train()
        sigreg.train()
        desc = "val(train_mode)"
    else:
        model.eval()
        sigreg.eval()
        desc = "val(eval_mode)"

    totals = {
        "loss": 0.0,
        "pred_loss": 0.0,
        "sigreg_loss": 0.0,
        "codebook_loss": 0.0,
        "commitment_loss": 0.0,
    }
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = lejepa_forward(model, sigreg, batch, config)

            totals["loss"] += output["loss"].detach().item()
            totals["pred_loss"] += output["pred_loss"].detach().item()
            totals["sigreg_loss"] += output["sigreg_loss"].detach().item()
            totals["codebook_loss"] += output["codebook_loss"].detach().item()
            totals["commitment_loss"] += output["commitment_loss"].detach().item()
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Validation loader is empty.")

    return {key: value / total_batches for key, value in totals.items()}


def main():
    config = build_train_config()
    checkpoint_path = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else find_latest_checkpoint()

    device = resolve_device(config)
    amp_dtype, _ = resolve_amp(device, config["trainer"]["precision"])

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"precision: {config['trainer']['precision']}")

    model = torch.load(checkpoint_path, map_location=device)
    model = model.to(device)

    sigreg = SIGReg(
        knots=config["loss"]["sigreg"]["knots"],
        num_proj=config["loss"]["sigreg"]["num_proj"],
    ).to(device)

    val_loader = build_val_loader(config)

    eval_metrics = evaluate_once(
        model,
        sigreg,
        val_loader,
        device,
        amp_dtype,
        config,
        train_mode=False,
    )
    train_metrics = evaluate_once(
        model,
        sigreg,
        val_loader,
        device,
        amp_dtype,
        config,
        train_mode=True,
    )

    print("\n=== eval mode ===")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.6f}")

    print("\n=== train mode + no_grad ===")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.6f}")

    print("\n=== difference (train_mode - eval_mode) ===")
    for key in eval_metrics:
        print(f"{key}: {train_metrics[key] - eval_metrics[key]:+.6f}")


if __name__ == "__main__":
    main()
