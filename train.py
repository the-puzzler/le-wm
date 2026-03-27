from pathlib import Path

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from tqdm.auto import tqdm

from jepa import JEPA
from module import ARPredictor, Embedder, InverseDynamicsTransformer, MLP, SIGReg, VectorQuantizer
from utils import (
    JsonlLogger,
    ModelArtifactSaver,
    TsvLogger,
    get_column_normalizer,
    get_img_preprocessor,
    save_training_plots,
)


def lejepa_forward(model, sigreg, batch, cfg):
    """Encode observations, predict next states, and compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch = dict(batch)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = model.encode(batch)

    emb = output["emb"]

    ctx_emb = emb[:, :ctx_len]
    if cfg.wm.use_learned_actions:
        code_features = model.infer_action_codes(emb[:, : ctx_len + n_preds])
        vq_output = model.quantize_action_codes(code_features[:, :ctx_len])
        ctx_act = vq_output["quantized"]
        output["code_indices"] = vq_output["indices"]
        output["codebook_loss"] = cfg.loss.vq.codebook_weight * vq_output["codebook_loss"]
        output["commitment_loss"] = cfg.loss.vq.commitment_weight * vq_output["commitment_loss"]
    else:
        act_emb = output["act_emb"]
        ctx_act = act_emb[:, :ctx_len]
        output["codebook_loss"] = emb.new_zeros(())
        output["commitment_loss"] = emb.new_zeros(())

    tgt_emb = emb[:, n_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = sigreg(emb.transpose(0, 1))
    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + output["codebook_loss"]
        + output["commitment_loss"]
    )
    return output


def resolve_device(cfg):
    accelerator = str(cfg.trainer.get("accelerator", "auto")).lower()
    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        devices = cfg.trainer.get("devices", "auto")
        if isinstance(devices, int):
            return torch.device("cuda:0")
        if isinstance(devices, str) and devices not in {"auto", ""}:
            return torch.device(f"cuda:{int(devices.split(',')[0])}")
        if isinstance(devices, (list, tuple)) and devices:
            return torch.device(f"cuda:{int(devices[0])}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_amp(device, precision):
    precision = str(precision).lower()
    if device.type != "cuda":
        return None, False
    if precision == "bf16":
        return torch.bfloat16, False
    if precision in {"16", "fp16", "float16"}:
        return torch.float16, True
    return None, False


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def maybe_step_scheduler(scheduler):
    if scheduler is not None:
        scheduler.step()


def evaluate(model, sigreg, loader, device, amp_dtype, cfg):
    model.eval()
    sigreg.eval()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    total_codebook_loss = 0.0
    total_commitment_loss = 0.0
    total_batches = 0

    progress = tqdm(loader, desc="val", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = lejepa_forward(model, sigreg, batch, cfg)

            total_loss += output["loss"].detach().item()
            total_pred_loss += output["pred_loss"].detach().item()
            total_sigreg_loss += output["sigreg_loss"].detach().item()
            total_codebook_loss += output["codebook_loss"].detach().item()
            total_commitment_loss += output["commitment_loss"].detach().item()
            total_batches += 1

            if batch_idx % max(1, int(cfg.logging.get("log_interval", 1))) == 0:
                progress.set_postfix(
                    loss=f"{total_loss / total_batches:.4f}",
                    pred=f"{total_pred_loss / total_batches:.4f}",
                    sigreg=f"{total_sigreg_loss / total_batches:.4f}",
                    codebook=f"{total_codebook_loss / total_batches:.4f}",
                    commit=f"{total_commitment_loss / total_batches:.4f}",
                )

    if total_batches == 0:
        return {
            "loss": 0.0,
            "pred_loss": 0.0,
            "sigreg_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
        }

    return {
        "loss": total_loss / total_batches,
        "pred_loss": total_pred_loss / total_batches,
        "sigreg_loss": total_sigreg_loss / total_batches,
        "codebook_loss": total_codebook_loss / total_batches,
        "commitment_loss": total_commitment_loss / total_batches,
    }


def train_one_epoch(model, sigreg, loader, optimizer, scaler, device, amp_dtype, cfg):
    model.train()
    sigreg.train()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    total_codebook_loss = 0.0
    total_commitment_loss = 0.0
    total_batches = 0

    grad_clip = cfg.trainer.get("gradient_clip_val")

    progress = tqdm(loader, desc="train", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            output = lejepa_forward(model, sigreg, batch, cfg)
            loss = output["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += output["loss"].detach().item()
        total_pred_loss += output["pred_loss"].detach().item()
        total_sigreg_loss += output["sigreg_loss"].detach().item()
        total_codebook_loss += output["codebook_loss"].detach().item()
        total_commitment_loss += output["commitment_loss"].detach().item()
        total_batches += 1

        if batch_idx % max(1, int(cfg.logging.get("log_interval", 1))) == 0:
            progress.set_postfix(
                loss=f"{total_loss / total_batches:.4f}",
                pred=f"{total_pred_loss / total_batches:.4f}",
                sigreg=f"{total_sigreg_loss / total_batches:.4f}",
                codebook=f"{total_codebook_loss / total_batches:.4f}",
                commit=f"{total_commitment_loss / total_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    if total_batches == 0:
        return {
            "loss": 0.0,
            "pred_loss": 0.0,
            "sigreg_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
        }

    return {
        "loss": total_loss / total_batches,
        "pred_loss": total_pred_loss / total_batches,
        "sigreg_loss": total_sigreg_loss / total_batches,
        "codebook_loss": total_codebook_loss / total_batches,
        "commitment_loss": total_commitment_loss / total_batches,
    }


def build_scheduler(optimizer, max_epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, cache_dir=cfg.cache_dir, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    inverse_dynamics = InverseDynamicsTransformer(
        num_frames=cfg.wm.history_size + cfg.wm.num_preds,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=embed_dim,
        **cfg.inverse_dynamics,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    quantizer = VectorQuantizer(
        num_codes=cfg.codebook.num_codes,
        code_dim=embed_dim,
        beta=cfg.codebook.beta,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        inverse_dynamics=inverse_dynamics,
        quantizer=quantizer,
    )
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    device = resolve_device(cfg)
    amp_dtype, use_grad_scaler = resolve_amp(device, cfg.trainer.get("precision", "32"))
    model = model.to(device)
    sigreg = sigreg.to(device)

    optimizer_cls = getattr(torch.optim, cfg.optimizer.type)
    optimizer_kwargs = {
        key: OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        for key, value in cfg.optimizer.items()
        if key != "type"
    }
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    scheduler = build_scheduler(optimizer, max_epochs=cfg.trainer.max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    run_id = cfg.get("subdir") or ""
    run_dir = Path(cfg.cache_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    artifact_saver = ModelArtifactSaver(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1
    )
    metrics_logger = JsonlLogger(run_dir / "metrics.jsonl")
    tsv_logger = TsvLogger(run_dir / "metrics.tsv") if cfg.logging.get("save_tsv", True) else None
    history = []

    max_epochs = int(cfg.trainer.max_epochs)
    log_interval = max(1, int(cfg.logging.get("log_interval", 1)))
    plot_interval = max(1, int(cfg.logging.get("plot_interval", 5)))
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            sigreg=sigreg,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            cfg=cfg,
        )
        val_metrics = evaluate(
            model=model,
            sigreg=sigreg,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            cfg=cfg,
        )
        maybe_step_scheduler(scheduler)

        current_lr = optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch,
            "lr": current_lr,
            "train/loss": train_metrics["loss"],
            "train/pred_loss": train_metrics["pred_loss"],
            "train/sigreg_loss": train_metrics["sigreg_loss"],
            "train/codebook_loss": train_metrics["codebook_loss"],
            "train/commitment_loss": train_metrics["commitment_loss"],
            "val/loss": val_metrics["loss"],
            "val/pred_loss": val_metrics["pred_loss"],
            "val/sigreg_loss": val_metrics["sigreg_loss"],
            "val/codebook_loss": val_metrics["codebook_loss"],
            "val/commitment_loss": val_metrics["commitment_loss"],
        }
        history.append(metrics)
        metrics_logger.log(metrics)
        if tsv_logger is not None:
            tsv_logger.log(metrics)
        artifact_saver.save_epoch(model=model, epoch=epoch, max_epochs=max_epochs)
        if epoch % plot_interval == 0 or epoch == 1 or epoch == max_epochs:
            save_training_plots(history, run_dir / "training_curves.png")

        if epoch % log_interval == 0 or epoch == 1 or epoch == max_epochs:
            print(
                f"epoch {epoch}/{max_epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"lr={current_lr:.6e}"
            )

    artifact_saver.save_final(model=model)


if __name__ == "__main__":
    run()
