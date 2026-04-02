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
from module import TAEAdapter
from script_utils import find_latest_object_checkpoint, set_source_model_mode
from train import (
    build_train_config,
    build_dataset,
    build_scheduler,
    move_batch_to_device,
    resolve_amp,
    resolve_device,
)
from utils import JsonlLogger, ModelArtifactSaver, TsvLogger


def build_adapter_config() -> dict:
    return {
        "run_name": cfg.TAESD_RUN_NAME,
        "runs_dir": Path(cfg.RUNS_DIR),
        "cache_dir": str(cfg.CACHE_DIR),
        "source_checkpoint": cfg.TAESD_SOURCE_CHECKPOINT,
        "source_model_mode": cfg.TAESD_SOURCE_MODEL_MODE,
        "taesd_model_name": cfg.TAESD_MODEL_NAME,
        "seed": cfg.SEED,
        "img_size": cfg.IMG_SIZE,
        "train_split": cfg.TRAIN_SPLIT,
        "output_model_name": f"{cfg.OUTPUT_MODEL_NAME}_taesd_adapter",
        "max_epochs": cfg.TAESD_MAX_EPOCHS,
        "batch_size": cfg.TAESD_BATCH_SIZE,
        "num_workers": cfg.TAESD_NUM_WORKERS,
        "persistent_workers": cfg.TAESD_PERSISTENT_WORKERS,
        "prefetch_factor": cfg.TAESD_PREFETCH_FACTOR,
        "pin_memory": cfg.TAESD_PIN_MEMORY,
        "lr": cfg.TAESD_LEARNING_RATE,
        "weight_decay": cfg.TAESD_WEIGHT_DECAY,
        "gradient_clip_val": cfg.TAESD_GRADIENT_CLIP_VAL,
        "latent_channels": cfg.TAESD_LATENT_CHANNELS,
        "latent_size": cfg.TAESD_LATENT_SIZE,
        "adapter_hidden_dim": cfg.TAESD_ADAPTER_HIDDEN_DIM,
        "console_every_steps": cfg.TAESD_CONSOLE_EVERY_STEPS,
        "write_every_steps": cfg.TAESD_WRITE_EVERY_STEPS,
        "plot_every_steps": cfg.TAESD_PLOT_EVERY_STEPS,
        "plot_every_epochs": cfg.TAESD_PLOT_EVERY_EPOCHS,
        "save_tsv": cfg.TAESD_SAVE_TSV,
        "num_vis_samples": cfg.TAESD_NUM_VIS_SAMPLES,
        "latent_loss_weight": cfg.TAESD_LATENT_LOSS_WEIGHT,
        "pixel_loss_weight": cfg.TAESD_PIXEL_LOSS_WEIGHT,
        "topk_fraction": cfg.TAESD_TOPK_FRACTION,
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


def build_data_loaders(train_config: dict, adapter_config: dict):
    import stable_pretraining as spt

    dataset = build_dataset(train_config)
    generator = torch.Generator().manual_seed(adapter_config["seed"])
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[adapter_config["train_split"], 1 - adapter_config["train_split"]],
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": adapter_config["batch_size"],
        "num_workers": adapter_config["num_workers"],
        "persistent_workers": adapter_config["persistent_workers"] and adapter_config["num_workers"] > 0,
        "pin_memory": adapter_config["pin_memory"],
    }
    if adapter_config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = adapter_config["prefetch_factor"]

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


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad_(False)


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def to_taesd_range(x: torch.Tensor) -> torch.Tensor:
    return denormalize_imagenet(x) * 2.0 - 1.0


def from_taesd_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def extract_source_latents(source_model, batch: dict, train_config: dict):
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


def encode_taesd(taesd, images: torch.Tensor) -> torch.Tensor:
    encoded = taesd.encode(images)
    return encoded.latents if hasattr(encoded, "latents") else encoded[0]


def decode_taesd(taesd, latents: torch.Tensor) -> torch.Tensor:
    decoded = taesd.decode(latents)
    return decoded.sample if hasattr(decoded, "sample") else decoded[0]


def compute_losses(
    pred_latents: torch.Tensor,
    target_latents: torch.Tensor,
    recon_images: torch.Tensor,
    target_images: torch.Tensor,
    latent_loss_weight: float,
    pixel_loss_weight: float,
    topk_fraction: float,
) -> dict[str, torch.Tensor]:
    latent_mse_loss = F.mse_loss(pred_latents, target_latents)
    per_pixel_mse = (recon_images - target_images).pow(2).reshape(recon_images.size(0), -1)
    k = max(1, int(per_pixel_mse.size(1) * topk_fraction))
    topk_pixel_mse_loss = per_pixel_mse.topk(k=k, dim=1).values.mean()
    pixel_mse_loss = per_pixel_mse.mean()
    loss = latent_loss_weight * latent_mse_loss + pixel_loss_weight * topk_pixel_mse_loss
    return {
        "loss": loss,
        "latent_mse_loss": latent_mse_loss,
        "topk_pixel_mse_loss": topk_pixel_mse_loss,
        "pixel_mse_loss": pixel_mse_loss,
    }


def metrics_row(*, split: str, epoch: int, epoch_step: int, global_step: int, lr: float | None, metrics: dict):
    return {
        "split": split,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "lr": lr,
        "loss": metrics["loss"],
        "latent_mse_loss": metrics["latent_mse_loss"],
        "topk_pixel_mse_loss": metrics["topk_pixel_mse_loss"],
        "pixel_mse_loss": metrics["pixel_mse_loss"],
    }


def average_metrics(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(1, count) for key, value in totals.items()}


def render_reconstruction_grid(
    source_model,
    taesd,
    adapter,
    loader,
    device,
    amp_dtype,
    train_config: dict,
    adapter_config: dict,
):
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    num_samples = min(adapter_config["num_vis_samples"], batch["pixels"].size(0))
    batch = {key: value[:num_samples] if torch.is_tensor(value) else value for key, value in batch.items()}

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            emb, pred_emb = extract_source_latents(source_model, batch, train_config)
            n_preds = train_config["wm"]["num_preds"]
            target_pixels = batch["pixels"][:, n_preds:]
            target_images = to_taesd_range(target_pixels.reshape(-1, *target_pixels.shape[2:]))

            target_taesd_latents = encode_taesd(taesd, target_images)
            recon_target = decode_taesd(
                taesd,
                adapter(emb[:, n_preds:].reshape(-1, emb.size(-1))),
            )
            ref_target = decode_taesd(taesd, target_taesd_latents)
            recon_pred = None
            if pred_emb is not None:
                recon_pred = decode_taesd(
                    taesd,
                    adapter(pred_emb.reshape(-1, pred_emb.size(-1))),
                )

    ref_target = from_taesd_range(ref_target).view_as(target_pixels)
    recon_target = from_taesd_range(recon_target).view_as(target_pixels)
    if recon_pred is not None:
        recon_pred = from_taesd_range(recon_pred).view_as(target_pixels)
    target_pixels = denormalize_imagenet(target_pixels.reshape(-1, *target_pixels.shape[2:])).view_as(target_pixels)

    rows = []
    for sample_idx in range(num_samples):
        rows.append(target_pixels[sample_idx])
        rows.append(ref_target[sample_idx])
        rows.append(recon_target[sample_idx])
        if recon_pred is not None:
            rows.append(recon_pred[sample_idx])

    flat = torch.cat(rows, dim=0)
    nrow = target_pixels.size(1)
    return make_grid(flat, nrow=nrow, padding=2)


def save_plots_with_visuals(history: list[dict], output_path: Path, visualization_path: Path | None):
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
        ("Latent MSE", "latent_mse_loss"),
        ("Top-k Pixel MSE", "topk_pixel_mse_loss"),
        ("Pixel MSE", "pixel_mse_loss"),
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
        image_ax.imshow(plt.imread(visualization_path))
        image_ax.axis("off")
        image_ax.set_title("TAESD Adapter Preview")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_one_epoch(
    *,
    source_model,
    taesd,
    adapter,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    amp_dtype,
    train_config: dict,
    adapter_config: dict,
    epoch: int,
    global_step_start: int,
    on_step_end,
):
    adapter.train()
    totals = {"loss": 0.0, "latent_mse_loss": 0.0, "topk_pixel_mse_loss": 0.0, "pixel_mse_loss": 0.0}
    running = {"loss": 0.0, "latent_mse_loss": 0.0, "topk_pixel_mse_loss": 0.0, "pixel_mse_loss": 0.0}
    total_batches = 0
    running_batches = 0

    progress = tqdm(loader, desc=f"taesd train {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        global_step = global_step_start + batch_idx
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            emb, _ = extract_source_latents(source_model, batch, train_config)

        target_pixels = batch["pixels"][:, train_config["wm"]["num_preds"]:]
        target_images = to_taesd_range(target_pixels.reshape(-1, *target_pixels.shape[2:]))
        latent_input = emb[:, train_config["wm"]["num_preds"] :].reshape(-1, emb.size(-1))

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            with torch.no_grad():
                target_taesd_latents = encode_taesd(taesd, target_images)
            pred_taesd_latents = adapter(latent_input)
            recon_images = decode_taesd(taesd, pred_taesd_latents)
            losses = compute_losses(
                pred_taesd_latents,
                target_taesd_latents,
                recon_images,
                target_images,
                adapter_config["latent_loss_weight"],
                adapter_config["pixel_loss_weight"],
                adapter_config["topk_fraction"],
            )

        if scaler is not None:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            if adapter_config["gradient_clip_val"] is not None:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), adapter_config["gradient_clip_val"])
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            if adapter_config["gradient_clip_val"] is not None:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), adapter_config["gradient_clip_val"])
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

        if batch_idx % max(1, adapter_config["console_every_steps"]) == 0:
            progress.set_postfix(
                loss=f"{running['loss'] / running_batches:.4f}",
                latent=f"{running['latent_mse_loss'] / running_batches:.4f}",
                topk=f"{running['topk_pixel_mse_loss'] / running_batches:.4f}",
                pixel=f"{running['pixel_mse_loss'] / running_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            running = {"loss": 0.0, "latent_mse_loss": 0.0, "topk_pixel_mse_loss": 0.0, "pixel_mse_loss": 0.0}
            running_batches = 0

    return average_metrics(totals, total_batches)


def evaluate_adapter(
    *,
    source_model,
    taesd,
    adapter,
    loader,
    device,
    amp_dtype,
    train_config: dict,
    adapter_config: dict,
    epoch: int,
    global_step: int,
):
    adapter.eval()
    totals = {"loss": 0.0, "latent_mse_loss": 0.0, "topk_pixel_mse_loss": 0.0, "pixel_mse_loss": 0.0}
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"taesd val {epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                emb, _ = extract_source_latents(source_model, batch, train_config)
                target_pixels = batch["pixels"][:, train_config["wm"]["num_preds"]:]
                target_images = to_taesd_range(target_pixels.reshape(-1, *target_pixels.shape[2:]))
                latent_input = emb[:, train_config["wm"]["num_preds"] :].reshape(-1, emb.size(-1))
                target_taesd_latents = encode_taesd(taesd, target_images)
                pred_taesd_latents = adapter(latent_input)
                recon_images = decode_taesd(taesd, pred_taesd_latents)
                losses = compute_losses(
                    pred_taesd_latents,
                    target_taesd_latents,
                    recon_images,
                    target_images,
                    adapter_config["latent_loss_weight"],
                    adapter_config["pixel_loss_weight"],
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
    try:
        from diffusers import AutoencoderKL, AutoencoderTiny
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "diffusers is required for TAESD adapter training. Install dependencies first."
        ) from exc

    torch.manual_seed(cfg.SEED)

    train_config = build_train_config()
    adapter_config = build_adapter_config()
    checkpoint_path = (
        Path(adapter_config["source_checkpoint"])
        if adapter_config["source_checkpoint"]
        else find_latest_object_checkpoint(cfg.RUNS_DIR, exclude_name_substrings=("_decoder",))
    )

    run_dir = create_run_dir(adapter_config)
    save_run_config({**adapter_config, "source_checkpoint": str(checkpoint_path)}, run_dir)

    device = resolve_device(train_config)
    amp_dtype, use_grad_scaler = resolve_amp(device, train_config["trainer"]["precision"])

    source_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_model = source_model.to(device)
    freeze_module(source_model)
    set_source_model_mode(
        source_model,
        adapter_config["source_model_mode"],
        mode_label="TAESD_SOURCE_MODEL_MODE",
    )

    try:
        taesd = AutoencoderTiny.from_pretrained(adapter_config["taesd_model_name"]).to(device)
    except Exception:
        taesd = AutoencoderKL.from_pretrained(adapter_config["taesd_model_name"]).to(device)
    taesd.eval()
    freeze_module(taesd)

    adapter = TAEAdapter(
        input_dim=train_config["wm"]["embed_dim"],
        latent_channels=adapter_config["latent_channels"],
        latent_size=adapter_config["latent_size"],
        hidden_dim=adapter_config["adapter_hidden_dim"],
    ).to(device)

    train_loader, val_loader = build_data_loaders(train_config, adapter_config)

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=adapter_config["lr"],
        weight_decay=adapter_config["weight_decay"],
    )
    total_train_steps = adapter_config["max_epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_train_steps=total_train_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    metrics_jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    metrics_tsv = TsvLogger(run_dir / "metrics.tsv") if adapter_config["save_tsv"] else None
    plot_rows: list[dict] = []
    pending_rows: list[dict] = []
    artifact_saver = ModelArtifactSaver(
        dirpath=run_dir,
        filename=adapter_config["output_model_name"],
        epoch_interval=1,
    )
    latest_vis_path = run_dir / "taesd_adapter_latest.png"

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
        if flush or len(pending_rows) >= max(1, adapter_config["write_every_steps"]):
            flush_pending_rows()
        if update_plot:
            grid = render_reconstruction_grid(
                source_model,
                taesd,
                adapter,
                val_loader,
                device,
                amp_dtype,
                train_config,
                adapter_config,
            )
            save_image(grid, latest_vis_path)
            save_plots_with_visuals(plot_rows, run_dir / "training_curves.png", latest_vis_path)

    global_step = 0
    for epoch in range(1, adapter_config["max_epochs"] + 1):
        epoch_train_metrics = train_one_epoch(
            source_model=source_model,
            taesd=taesd,
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            train_config=train_config,
            adapter_config=adapter_config,
            epoch=epoch,
            global_step_start=global_step,
            on_step_end=lambda row, batch_idx: persist_row(
                row,
                update_plot=batch_idx % max(1, adapter_config["plot_every_steps"]) == 0,
                flush=batch_idx % max(1, adapter_config["write_every_steps"]) == 0,
            ),
        )
        global_step += len(train_loader)

        val_row = evaluate_adapter(
            source_model=source_model,
            taesd=taesd,
            adapter=adapter,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            train_config=train_config,
            adapter_config=adapter_config,
            epoch=epoch,
            global_step=global_step,
        )
        persist_row(val_row, update_plot=True, flush=True)

        artifact_saver.save_epoch(adapter, epoch=epoch, max_epochs=adapter_config["max_epochs"])
        if epoch % max(1, adapter_config["plot_every_epochs"]) == 0:
            grid = render_reconstruction_grid(
                source_model,
                taesd,
                adapter,
                val_loader,
                device,
                amp_dtype,
                train_config,
                adapter_config,
            )
            save_image(grid, run_dir / f"taesd_adapter_epoch_{epoch}.png")
            save_plots_with_visuals(plot_rows, run_dir / "training_curves.png", latest_vis_path)

        print(
            f"taesd epoch {epoch}/{adapter_config['max_epochs']} "
            f"train_loss={epoch_train_metrics['loss']:.6f} "
            f"val_loss={val_row['loss']:.6f} "
            f"step={global_step}"
        )

    flush_pending_rows()
    artifact_saver.save_final(adapter)
    grid = render_reconstruction_grid(
        source_model,
        taesd,
        adapter,
        val_loader,
        device,
        amp_dtype,
        train_config,
        adapter_config,
    )
    save_image(grid, latest_vis_path)
    save_plots_with_visuals(plot_rows, run_dir / "training_curves.png", latest_vis_path)
    print(f"saved TAESD adapter run to {run_dir}")


if __name__ == "__main__":
    main()
