from __future__ import annotations

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import config as cfg
from train import (
    build_train_config,
    build_dataset,
    build_scheduler,
    move_batch_to_device,
    resolve_amp,
    resolve_device,
)
from utils import JsonlLogger, ModelArtifactSaver, TsvLogger


class ActionTranslator(nn.Module):
    def __init__(self, num_codes: int, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.code_embedding = nn.Embedding(num_codes, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_emb: torch.Tensor, code_indices: torch.Tensor) -> torch.Tensor:
        code_emb = self.code_embedding(code_indices)
        features = torch.cat([state_emb, code_emb], dim=-1)
        return self.net(features)


def build_translator_config() -> dict:
    return {
        "run_name": cfg.TRANSLATOR_RUN_NAME,
        "runs_dir": Path(cfg.RUNS_DIR),
        "cache_dir": str(cfg.CACHE_DIR),
        "source_checkpoint": cfg.TRANSLATOR_SOURCE_CHECKPOINT,
        "source_model_mode": cfg.TRANSLATOR_SOURCE_MODEL_MODE,
        "seed": cfg.SEED,
        "train_split": cfg.TRAIN_SPLIT,
        "output_model_name": f"{cfg.OUTPUT_MODEL_NAME}_action_translator",
        "max_epochs": cfg.TRANSLATOR_MAX_EPOCHS,
        "batch_size": cfg.TRANSLATOR_BATCH_SIZE,
        "num_workers": cfg.TRANSLATOR_NUM_WORKERS,
        "persistent_workers": cfg.TRANSLATOR_PERSISTENT_WORKERS,
        "prefetch_factor": cfg.TRANSLATOR_PREFETCH_FACTOR,
        "pin_memory": cfg.TRANSLATOR_PIN_MEMORY,
        "lr": cfg.TRANSLATOR_LEARNING_RATE,
        "weight_decay": cfg.TRANSLATOR_WEIGHT_DECAY,
        "gradient_clip_val": cfg.TRANSLATOR_GRADIENT_CLIP_VAL,
        "hidden_dim": cfg.TRANSLATOR_HIDDEN_DIM,
        "console_every_steps": cfg.TRANSLATOR_CONSOLE_EVERY_STEPS,
        "write_every_steps": cfg.TRANSLATOR_WRITE_EVERY_STEPS,
        "plot_every_steps": cfg.TRANSLATOR_PLOT_EVERY_STEPS,
        "plot_every_epochs": cfg.TRANSLATOR_PLOT_EVERY_EPOCHS,
        "save_tsv": cfg.TRANSLATOR_SAVE_TSV,
        "num_vis_samples": cfg.TRANSLATOR_NUM_VIS_SAMPLES,
        "mse_weight": cfg.TRANSLATOR_MSE_WEIGHT,
    }


def create_run_dir(config: dict) -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config["runs_dir"] / f"{timestamp}-{config['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_config(config: dict, run_dir: Path) -> None:
    serializable = dict(config)
    serializable["runs_dir"] = str(serializable["runs_dir"])
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def find_latest_checkpoint() -> Path:
    runs_dir = Path(cfg.RUNS_DIR)
    candidates = sorted(
        [
            path
            for path in runs_dir.glob("*/*_object.ckpt")
            if "_decoder" not in path.name and "_translator" not in path.name
        ],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints found under: {runs_dir}")
    return candidates[-1]


def build_data_loaders(train_config: dict, translator_config: dict):
    import stable_pretraining as spt

    dataset = build_dataset(train_config)
    generator = torch.Generator().manual_seed(translator_config["seed"])
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[translator_config["train_split"], 1 - translator_config["train_split"]],
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": translator_config["batch_size"],
        "num_workers": translator_config["num_workers"],
        "persistent_workers": translator_config["persistent_workers"]
        and translator_config["num_workers"] > 0,
        "pin_memory": translator_config["pin_memory"],
    }
    if translator_config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = translator_config["prefetch_factor"]

    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def set_source_model_mode(model, mode: str):
    if mode == "train":
        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unsupported TRANSLATOR_SOURCE_MODEL_MODE: {mode}")


def freeze_source_model(model):
    for param in model.parameters():
        param.requires_grad_(False)


def extract_translator_targets(source_model, batch: dict, train_config: dict):
    batch = dict(batch)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    ctx_len = train_config["wm"]["history_size"]
    n_preds = train_config["wm"]["num_preds"]

    output = source_model.encode(batch)
    emb = output["emb"]
    code_features = source_model.infer_action_codes(emb[:, : ctx_len + n_preds])
    vq_output = source_model.quantize_action_codes(code_features[:, :ctx_len])

    state_emb = emb[:, :ctx_len].reshape(-1, emb.size(-1))
    code_indices = vq_output["indices"].reshape(-1)
    target_actions = batch["action"][:, :ctx_len].reshape(-1, batch["action"].size(-1))
    return state_emb, code_indices, target_actions


def compute_losses(pred_action: torch.Tensor, target_action: torch.Tensor, mse_weight: float) -> dict[str, torch.Tensor]:
    l1_loss = F.l1_loss(pred_action, target_action)
    mse_loss = F.mse_loss(pred_action, target_action)
    loss = l1_loss + mse_weight * mse_loss
    return {
        "loss": loss,
        "l1_loss": l1_loss,
        "mse_loss": mse_loss,
    }


def metrics_row(*, split: str, epoch: int, epoch_step: int, global_step: int, lr: float | None, metrics: dict):
    return {
        "split": split,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "lr": lr,
        "loss": metrics["loss"],
        "l1_loss": metrics["l1_loss"],
        "mse_loss": metrics["mse_loss"],
    }


def average_metrics(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(1, count) for key, value in totals.items()}


def save_action_visualization(
    source_model,
    translator,
    loader,
    device,
    amp_dtype,
    train_config: dict,
    translator_config: dict,
    output_path: Path,
):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            state_emb, code_indices, target_actions = extract_translator_targets(
                source_model,
                batch,
                train_config,
            )
            pred_actions = translator(state_emb, code_indices)

    num_samples = min(translator_config["num_vis_samples"], target_actions.size(0))
    target_actions = target_actions[:num_samples].detach().cpu()
    pred_actions = pred_actions[:num_samples].detach().cpu()
    code_indices = code_indices[:num_samples].detach().cpu()

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2.4 * num_samples), squeeze=False)
    for sample_idx in range(num_samples):
        ax = axes[sample_idx, 0]
        dims = list(range(target_actions.size(-1)))
        ax.plot(dims, target_actions[sample_idx].tolist(), marker="o", label="target")
        ax.plot(dims, pred_actions[sample_idx].tolist(), marker="x", label="pred")
        ax.set_title(f"Sample {sample_idx + 1} | code {int(code_indices[sample_idx])}")
        ax.set_xlabel("Action Dimension")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_translator_plots_with_visuals(history: list[dict], output_path: Path, visualization_path: Path | None):
    if not history:
        return

    train_rows = [row for row in history if row.get("split") == "train"]
    val_rows = [row for row in history if row.get("split") == "val"]
    if not train_rows and not val_rows:
        return

    train_steps = [row["global_step"] for row in train_rows]
    val_steps = [row["global_step"] for row in val_rows]
    if visualization_path is None or not visualization_path.exists():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.ravel()
        image_ax = None
    else:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.35])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
        image_ax = fig.add_subplot(gs[2, :])

    plots = [
        ("Total Loss", "loss"),
        ("L1 Loss", "l1_loss"),
        ("MSE Loss", "mse_loss"),
        ("Learning Rate", "lr"),
    ]

    for ax, (title, key) in zip(axes, plots):
        if train_rows:
            ax.plot(train_steps, [row[key] for row in train_rows], label=f"train/{key}", linewidth=1.5)
        if val_rows and key != "lr":
            ax.plot(val_steps, [row[key] for row in val_rows], label=f"val/{key}", linewidth=2.0, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

    if image_ax is not None:
        image = plt.imread(visualization_path)
        image_ax.imshow(image)
        image_ax.axis("off")
        image_ax.set_title("Action Translation Preview")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_one_epoch(
    *,
    source_model,
    translator,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    amp_dtype,
    train_config: dict,
    translator_config: dict,
    epoch: int,
    global_step_start: int,
    on_step_end,
):
    translator.train()
    totals = {"loss": 0.0, "l1_loss": 0.0, "mse_loss": 0.0}
    running = {"loss": 0.0, "l1_loss": 0.0, "mse_loss": 0.0}
    total_batches = 0
    running_batches = 0

    progress = tqdm(loader, desc=f"translator train {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        global_step = global_step_start + batch_idx
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            state_emb, code_indices, target_actions = extract_translator_targets(
                source_model,
                batch,
                train_config,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred_actions = translator(state_emb, code_indices)
            losses = compute_losses(pred_actions, target_actions, translator_config["mse_weight"])

        if scaler is not None:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            if translator_config["gradient_clip_val"] is not None:
                torch.nn.utils.clip_grad_norm_(translator.parameters(), translator_config["gradient_clip_val"])
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            if translator_config["gradient_clip_val"] is not None:
                torch.nn.utils.clip_grad_norm_(translator.parameters(), translator_config["gradient_clip_val"])
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        step_metrics = {key: value.detach().item() for key, value in losses.items()}
        for key, value in step_metrics.items():
            totals[key] += value
            running[key] += value
        total_batches += 1
        running_batches += 1

        on_step_end(
            metrics_row(
                split="train",
                epoch=epoch,
                epoch_step=batch_idx,
                global_step=global_step,
                lr=optimizer.param_groups[0]["lr"],
                metrics=step_metrics,
            ),
            batch_idx=batch_idx,
        )

        if batch_idx % max(1, translator_config["console_every_steps"]) == 0:
            progress.set_postfix(
                loss=f"{running['loss'] / running_batches:.4f}",
                l1=f"{running['l1_loss'] / running_batches:.4f}",
                mse=f"{running['mse_loss'] / running_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            running = {"loss": 0.0, "l1_loss": 0.0, "mse_loss": 0.0}
            running_batches = 0

    return average_metrics(totals, total_batches)


def evaluate_translator(
    *,
    source_model,
    translator,
    loader,
    device,
    amp_dtype,
    train_config: dict,
    translator_config: dict,
    epoch: int,
    global_step: int,
):
    translator.eval()
    totals = {"loss": 0.0, "l1_loss": 0.0, "mse_loss": 0.0}
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"translator val {epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                state_emb, code_indices, target_actions = extract_translator_targets(
                    source_model,
                    batch,
                    train_config,
                )
                pred_actions = translator(state_emb, code_indices)
                losses = compute_losses(pred_actions, target_actions, translator_config["mse_weight"])

            for key, value in losses.items():
                totals[key] += value.detach().item()
            total_batches += 1

    return metrics_row(
        split="val",
        epoch=epoch,
        epoch_step=0,
        global_step=global_step,
        lr=None,
        metrics=average_metrics(totals, total_batches),
    )


def main():
    torch.manual_seed(cfg.SEED)

    train_config = build_train_config()
    if not train_config["wm"]["use_learned_actions"]:
        raise ValueError("Action translator is only needed when USE_LEARNED_ACTIONS = True.")

    translator_config = build_translator_config()
    checkpoint_path = (
        Path(translator_config["source_checkpoint"])
        if translator_config["source_checkpoint"]
        else find_latest_checkpoint()
    )

    run_dir = create_run_dir(translator_config)
    save_run_config({**translator_config, "source_checkpoint": str(checkpoint_path)}, run_dir)

    device = resolve_device(train_config)
    amp_dtype, use_grad_scaler = resolve_amp(device, train_config["trainer"]["precision"])

    source_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_model = source_model.to(device)
    freeze_source_model(source_model)
    set_source_model_mode(source_model, translator_config["source_model_mode"])

    action_dim = train_config["dataset"]["frameskip"] * train_config["wm"]["action_dim"]
    translator = ActionTranslator(
        num_codes=train_config["codebook"]["num_codes"],
        state_dim=train_config["wm"]["embed_dim"],
        action_dim=action_dim,
        hidden_dim=translator_config["hidden_dim"],
    ).to(device)

    train_loader, val_loader = build_data_loaders(train_config, translator_config)

    optimizer = torch.optim.AdamW(
        translator.parameters(),
        lr=translator_config["lr"],
        weight_decay=translator_config["weight_decay"],
    )
    total_train_steps = translator_config["max_epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_train_steps=total_train_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    metrics_jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    metrics_tsv = TsvLogger(run_dir / "metrics.tsv") if translator_config["save_tsv"] else None
    plot_rows: list[dict] = []
    pending_rows: list[dict] = []
    artifact_saver = ModelArtifactSaver(
        dirpath=run_dir,
        filename=translator_config["output_model_name"],
        epoch_interval=1,
    )
    latest_vis_path = run_dir / "action_translation_latest.png"

    def flush_pending_rows():
        if not pending_rows:
            return
        for row in pending_rows:
            metrics_jsonl.log(row)
            if metrics_tsv is not None:
                metrics_tsv.log(row)
        pending_rows.clear()

    def persist_row(row: dict, *, update_plot: bool = False, flush: bool = False):
        pending_rows.append(row)
        plot_rows.append(row)
        if flush or len(pending_rows) >= max(1, translator_config["write_every_steps"]):
            flush_pending_rows()
        if update_plot:
            save_action_visualization(
                source_model,
                translator,
                val_loader,
                device,
                amp_dtype,
                train_config,
                translator_config,
                latest_vis_path,
            )
            save_translator_plots_with_visuals(
                plot_rows,
                run_dir / "training_curves.png",
                latest_vis_path,
            )

    global_step = 0
    for epoch in range(1, translator_config["max_epochs"] + 1):
        epoch_train_metrics = train_one_epoch(
            source_model=source_model,
            translator=translator,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            train_config=train_config,
            translator_config=translator_config,
            epoch=epoch,
            global_step_start=global_step,
            on_step_end=lambda row, batch_idx: persist_row(
                row,
                update_plot=batch_idx % max(1, translator_config["plot_every_steps"]) == 0,
                flush=batch_idx % max(1, translator_config["write_every_steps"]) == 0,
            ),
        )
        global_step += len(train_loader)

        val_row = evaluate_translator(
            source_model=source_model,
            translator=translator,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            train_config=train_config,
            translator_config=translator_config,
            epoch=epoch,
            global_step=global_step,
        )
        persist_row(val_row, update_plot=True, flush=True)

        artifact_saver.save_epoch(
            translator,
            epoch=epoch,
            max_epochs=translator_config["max_epochs"],
        )
        if epoch % max(1, translator_config["plot_every_epochs"]) == 0:
            save_action_visualization(
                source_model,
                translator,
                val_loader,
                device,
                amp_dtype,
                train_config,
                translator_config,
                run_dir / f"action_translation_epoch_{epoch}.png",
            )
            save_translator_plots_with_visuals(
                plot_rows,
                run_dir / "training_curves.png",
                latest_vis_path,
            )

        print(
            f"translator epoch {epoch}/{translator_config['max_epochs']} "
            f"train_loss={epoch_train_metrics['loss']:.6f} "
            f"val_loss={val_row['loss']:.6f} "
            f"step={global_step}"
        )

    flush_pending_rows()
    artifact_saver.save_final(translator)
    save_action_visualization(
        source_model,
        translator,
        val_loader,
        device,
        amp_dtype,
        train_config,
        translator_config,
        latest_vis_path,
    )
    save_translator_plots_with_visuals(
        plot_rows,
        run_dir / "training_curves.png",
        latest_vis_path,
    )
    print(f"saved action translator run to {run_dir}")


if __name__ == "__main__":
    main()
