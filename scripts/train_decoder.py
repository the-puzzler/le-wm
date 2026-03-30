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
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

import config as cfg
from module import VisualDecoder
from train import (
    build_train_config,
    build_dataset,
    build_scheduler,
    move_batch_to_device,
    resolve_amp,
    resolve_device,
)
from utils import JsonlLogger, ModelArtifactSaver, TsvLogger


def build_decoder_config() -> dict:
    return {
        "run_name": cfg.DECODER_RUN_NAME,
        "runs_dir": Path(cfg.RUNS_DIR),
        "cache_dir": str(cfg.CACHE_DIR),
        "source_checkpoint": cfg.DECODER_SOURCE_CHECKPOINT,
        "source_model_mode": cfg.DECODER_SOURCE_MODEL_MODE,
        "resume_checkpoint": cfg.DECODER_RESUME_CHECKPOINT,
        "seed": cfg.SEED,
        "img_size": cfg.IMG_SIZE,
        "train_split": cfg.TRAIN_SPLIT,
        "output_model_name": f"{cfg.OUTPUT_MODEL_NAME}_decoder",
        "embed_dim": cfg.DECODER_EMBED_DIM,
        "base_channels": cfg.DECODER_BASE_CHANNELS,
        "max_epochs": cfg.DECODER_MAX_EPOCHS,
        "batch_size": cfg.DECODER_BATCH_SIZE,
        "num_workers": cfg.DECODER_NUM_WORKERS,
        "persistent_workers": cfg.DECODER_PERSISTENT_WORKERS,
        "prefetch_factor": cfg.DECODER_PREFETCH_FACTOR,
        "pin_memory": cfg.DECODER_PIN_MEMORY,
        "lr": cfg.DECODER_LEARNING_RATE,
        "weight_decay": cfg.DECODER_WEIGHT_DECAY,
        "gradient_clip_val": cfg.DECODER_GRADIENT_CLIP_VAL,
        "console_every_steps": cfg.DECODER_CONSOLE_EVERY_STEPS,
        "write_every_steps": cfg.DECODER_WRITE_EVERY_STEPS,
        "plot_every_steps": cfg.DECODER_PLOT_EVERY_STEPS,
        "plot_every_epochs": cfg.DECODER_PLOT_EVERY_EPOCHS,
        "save_tsv": cfg.DECODER_SAVE_TSV,
        "num_vis_samples": cfg.DECODER_NUM_VIS_SAMPLES,
        "topk_fraction": cfg.DECODER_TOPK_FRACTION,
        "lpips_weight": cfg.DECODER_LPIPS_WEIGHT,
        "mse_weight": cfg.DECODER_MSE_WEIGHT,
        "topk_weight": cfg.DECODER_TOPK_WEIGHT,
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
            if "_decoder" not in path.name
        ],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints found under: {runs_dir}")
    return candidates[-1]


def build_data_loaders(train_config: dict, decoder_config: dict):
    import stable_pretraining as spt

    dataset = build_dataset(train_config)
    generator = torch.Generator().manual_seed(decoder_config["seed"])
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[decoder_config["train_split"], 1 - decoder_config["train_split"]],
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": decoder_config["batch_size"],
        "num_workers": decoder_config["num_workers"],
        "persistent_workers": decoder_config["persistent_workers"] and decoder_config["num_workers"] > 0,
        "pin_memory": decoder_config["pin_memory"],
    }
    if decoder_config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = decoder_config["prefetch_factor"]

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
        raise ValueError(f"Unsupported DECODER_SOURCE_MODEL_MODE: {mode}")


def freeze_source_model(model):
    for param in model.parameters():
        param.requires_grad_(False)


def denormalize_pixels(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def extract_latents(source_model, batch: dict, train_config: dict):
    output = source_model.encode(batch)
    emb = output["emb"]
    pred_emb = None

    if train_config["wm"]["use_learned_actions"]:
        ctx_len = train_config["wm"]["history_size"]
        n_preds = train_config["wm"]["num_preds"]
        code_features = source_model.infer_action_codes(emb[:, : ctx_len + n_preds])
        vq_output = source_model.quantize_action_codes(code_features[:, :ctx_len])
        ctx_act = vq_output["quantized"]
        pred_emb = source_model.predict(emb[:, :ctx_len], ctx_act)

    return emb, pred_emb


def compute_decoder_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    topk_fraction: float,
    mse_weight: float,
    topk_weight: float,
    lpips_model=None,
    lpips_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    per_pixel_mse = (recon - target).pow(2).reshape(recon.size(0), -1)
    k = max(1, int(per_pixel_mse.size(1) * topk_fraction))
    topk_mse_loss = per_pixel_mse.topk(k=k, dim=1).values.mean()
    mse_loss = per_pixel_mse.mean()
    lpips_loss = recon.new_zeros(())
    if lpips_model is not None and lpips_weight > 0.0:
        recon_for_lpips = denormalize_pixels(recon.float())
        target_for_lpips = denormalize_pixels(target.float())
        lpips_loss = lpips_model(recon_for_lpips, target_for_lpips)
    loss = mse_weight * mse_loss + topk_weight * topk_mse_loss + lpips_weight * lpips_loss
    return {
        "loss": loss,
        "topk_mse_loss": topk_mse_loss,
        "mse_loss": mse_loss,
        "lpips_loss": lpips_loss,
    }


def metrics_row(*, split: str, epoch: int, epoch_step: int, global_step: int, lr: float | None, metrics: dict):
    return {
        "split": split,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "lr": lr,
        "loss": metrics["loss"],
        "topk_mse_loss": metrics["topk_mse_loss"],
        "mse_loss": metrics["mse_loss"],
        "lpips_loss": metrics["lpips_loss"],
    }


def average_metrics(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(1, count) for key, value in totals.items()}


def save_decoder_plots(history: list[dict], output_path: Path):
    save_decoder_plots_with_visuals(history, output_path, visualization_path=None)


def render_reconstruction_grid(
    source_model,
    decoder,
    loader,
    device,
    amp_dtype,
    train_config: dict,
    decoder_config: dict,
):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    num_samples = min(decoder_config["num_vis_samples"], batch["pixels"].size(0))
    batch = {key: value[:num_samples] if torch.is_tensor(value) else value for key, value in batch.items()}

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            emb, pred_emb = extract_latents(source_model, batch, train_config)
            ctx_len = train_config["wm"]["history_size"]
            n_preds = train_config["wm"]["num_preds"]
            target_pixels = batch["pixels"][:, n_preds:]
            recon_target = decoder(emb[:, n_preds:].reshape(-1, emb.size(-1))).view_as(target_pixels)
            recon_pred = None
            if pred_emb is not None:
                recon_pred = decoder(pred_emb.reshape(-1, pred_emb.size(-1))).view_as(target_pixels)

    rows = []
    labels = []
    for sample_idx in range(num_samples):
        rows.append(denormalize_pixels(target_pixels[sample_idx]))
        labels.append("target")
        rows.append(denormalize_pixels(recon_target[sample_idx]))
        labels.append("recon(real z)")
        if recon_pred is not None:
            rows.append(denormalize_pixels(recon_pred[sample_idx]))
            labels.append("recon(pred z)")

    flat = torch.cat(rows, dim=0)
    nrow = target_pixels.size(1)
    grid = make_grid(flat, nrow=nrow, padding=2)
    return grid, labels, nrow


def save_decoder_plots_with_visuals(history: list[dict], output_path: Path, visualization_path: Path | None):
    if not history:
        return

    train_rows = [row for row in history if row.get("split") == "train"]
    val_rows = [row for row in history if row.get("split") == "val"]
    if not train_rows and not val_rows:
        return

    train_steps = [row["global_step"] for row in train_rows]
    val_steps = [row["global_step"] for row in val_rows]
    if visualization_path is None or not visualization_path.exists():
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        axes = axes.ravel()
        image_ax = None
    else:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3])
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
        ]
        image_ax = fig.add_subplot(gs[2, 1])

    plots = [
        ("Total Loss", "loss"),
        ("Top-k MSE Loss", "topk_mse_loss"),
        ("MSE Loss", "mse_loss"),
        ("LPIPS Loss", "lpips_loss"),
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
        image_ax.set_title("Reconstruction Preview")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_reconstruction_visualization(
    source_model,
    decoder,
    loader,
    device,
    amp_dtype,
    train_config: dict,
    decoder_config: dict,
    output_path: Path,
):
    grid, _, _ = render_reconstruction_grid(
        source_model,
        decoder,
        loader,
        device,
        amp_dtype,
        train_config,
        decoder_config,
    )
    save_image(grid, output_path)


def train_one_epoch(
    *,
    source_model,
    decoder,
    loader,
    optimizer,
    scheduler,
    scaler,
    lpips_model,
    device,
    amp_dtype,
    train_config: dict,
    decoder_config: dict,
    epoch: int,
    global_step_start: int,
    on_step_end,
):
    decoder.train()
    totals = {"loss": 0.0, "topk_mse_loss": 0.0, "mse_loss": 0.0, "lpips_loss": 0.0}
    running = {"loss": 0.0, "topk_mse_loss": 0.0, "mse_loss": 0.0, "lpips_loss": 0.0}
    total_batches = 0
    running_batches = 0

    progress = tqdm(loader, desc=f"decoder train {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        global_step = global_step_start + batch_idx
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            emb, _ = extract_latents(source_model, batch, train_config)

        latent = emb.reshape(-1, emb.size(-1))
        target = batch["pixels"].reshape(-1, *batch["pixels"].shape[2:])

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            recon = decoder(latent)
            losses = compute_decoder_losses(
                recon,
                target,
                decoder_config["topk_fraction"],
                decoder_config["mse_weight"],
                decoder_config["topk_weight"],
                lpips_model=lpips_model,
                lpips_weight=decoder_config["lpips_weight"],
            )

        if scaler is not None:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            if decoder_config["gradient_clip_val"] is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), decoder_config["gradient_clip_val"])
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            if decoder_config["gradient_clip_val"] is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), decoder_config["gradient_clip_val"])
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

        if batch_idx % max(1, decoder_config["console_every_steps"]) == 0:
            progress.set_postfix(
                loss=f"{running['loss'] / running_batches:.4f}",
                topk=f"{running['topk_mse_loss'] / running_batches:.4f}",
                mse=f"{running['mse_loss'] / running_batches:.4f}",
                lpips=f"{running['lpips_loss'] / running_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            running = {"loss": 0.0, "topk_mse_loss": 0.0, "mse_loss": 0.0, "lpips_loss": 0.0}
            running_batches = 0

    return average_metrics(totals, total_batches)


def evaluate_decoder(
    *,
    source_model,
    decoder,
    loader,
    device,
    amp_dtype,
    lpips_model,
    train_config: dict,
    decoder_config: dict,
    epoch: int,
    global_step: int,
):
    decoder.eval()
    totals = {"loss": 0.0, "topk_mse_loss": 0.0, "mse_loss": 0.0, "lpips_loss": 0.0}
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"decoder val {epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                emb, _ = extract_latents(source_model, batch, train_config)
                latent = emb.reshape(-1, emb.size(-1))
                target = batch["pixels"].reshape(-1, *batch["pixels"].shape[2:])
                recon = decoder(latent)
                losses = compute_decoder_losses(
                recon,
                target,
                decoder_config["topk_fraction"],
                decoder_config["mse_weight"],
                decoder_config["topk_weight"],
                lpips_model=lpips_model,
                lpips_weight=decoder_config["lpips_weight"],
            )

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
    decoder_config = build_decoder_config()
    checkpoint_path = Path(decoder_config["source_checkpoint"]) if decoder_config["source_checkpoint"] else find_latest_checkpoint()

    run_dir = create_run_dir(decoder_config)
    save_run_config(
        {
            **decoder_config,
            "source_checkpoint": str(checkpoint_path),
            "resume_checkpoint": (
                str(decoder_config["resume_checkpoint"])
                if decoder_config["resume_checkpoint"] is not None
                else None
            ),
        },
        run_dir,
    )

    device = resolve_device(train_config)
    amp_dtype, use_grad_scaler = resolve_amp(device, train_config["trainer"]["precision"])

    lpips_model = None
    if decoder_config["lpips_weight"] > 0.0:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        lpips_model.eval()
        lpips_model.requires_grad_(False)

    source_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_model = source_model.to(device)
    freeze_source_model(source_model)
    set_source_model_mode(source_model, decoder_config["source_model_mode"])

    decoder = VisualDecoder(
        embed_dim=decoder_config["embed_dim"],
        base_channels=decoder_config["base_channels"],
    ).to(device)
    if decoder_config["resume_checkpoint"] is not None:
        resume_path = Path(decoder_config["resume_checkpoint"])
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            decoder.load_state_dict(state)
        else:
            decoder.load_state_dict(state.state_dict())
        print(f"Loaded decoder weights from {resume_path}")

    train_loader, val_loader = build_data_loaders(train_config, decoder_config)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=decoder_config["lr"],
        weight_decay=decoder_config["weight_decay"],
    )
    total_train_steps = decoder_config["max_epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_train_steps=total_train_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    metrics_jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    metrics_tsv = TsvLogger(run_dir / "metrics.tsv") if decoder_config["save_tsv"] else None
    plot_rows: list[dict] = []
    pending_rows: list[dict] = []
    artifact_saver = ModelArtifactSaver(
        dirpath=run_dir,
        filename=decoder_config["output_model_name"],
        epoch_interval=1,
    )
    latest_vis_path = run_dir / "reconstructions_latest.png"

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
        if flush or len(pending_rows) >= max(1, decoder_config["write_every_steps"]):
            flush_pending_rows()
        if update_plot:
            save_reconstruction_visualization(
                source_model,
                decoder,
                val_loader,
                device,
                amp_dtype,
                train_config,
                decoder_config,
                latest_vis_path,
            )
            save_decoder_plots_with_visuals(
                plot_rows,
                run_dir / "training_curves.png",
                latest_vis_path,
            )

    global_step = 0
    for epoch in range(1, decoder_config["max_epochs"] + 1):
        epoch_train_metrics = train_one_epoch(
            source_model=source_model,
            decoder=decoder,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            lpips_model=lpips_model,
            device=device,
            amp_dtype=amp_dtype,
            train_config=train_config,
            decoder_config=decoder_config,
            epoch=epoch,
            global_step_start=global_step,
            on_step_end=lambda row, batch_idx: persist_row(
                row,
                update_plot=batch_idx % max(1, decoder_config["plot_every_steps"]) == 0,
                flush=batch_idx % max(1, decoder_config["write_every_steps"]) == 0,
            ),
        )
        global_step += len(train_loader)

        val_row = evaluate_decoder(
            source_model=source_model,
            decoder=decoder,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            lpips_model=lpips_model,
            train_config=train_config,
            decoder_config=decoder_config,
            epoch=epoch,
            global_step=global_step,
        )
        persist_row(val_row, update_plot=True, flush=True)

        artifact_saver.save_epoch(decoder, epoch=epoch, max_epochs=decoder_config["max_epochs"])
        if epoch % max(1, decoder_config["plot_every_epochs"]) == 0:
            save_reconstruction_visualization(
                source_model,
                decoder,
                val_loader,
                device,
                amp_dtype,
                train_config,
                decoder_config,
                run_dir / f"reconstructions_epoch_{epoch}.png",
            )
            save_decoder_plots_with_visuals(
                plot_rows,
                run_dir / "training_curves.png",
                latest_vis_path,
            )

        print(
            f"decoder epoch {epoch}/{decoder_config['max_epochs']} "
            f"train_loss={epoch_train_metrics['loss']:.6f} "
            f"val_loss={val_row['loss']:.6f} "
            f"step={global_step}"
        )

    flush_pending_rows()
    artifact_saver.save_final(decoder)
    save_reconstruction_visualization(
        source_model,
        decoder,
        val_loader,
        device,
        amp_dtype,
        train_config,
        decoder_config,
        latest_vis_path,
    )
    save_decoder_plots_with_visuals(
        plot_rows,
        run_dir / "training_curves.png",
        latest_vis_path,
    )
    print(f"saved decoder run to {run_dir}")


if __name__ == "__main__":
    main()
