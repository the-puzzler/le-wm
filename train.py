from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import config as cfg
from utils import (
    JsonlLogger,
    ModelArtifactSaver,
    TsvLogger,
    get_column_normalizer,
    get_img_preprocessor,
    save_training_plots,
)


def build_train_config() -> dict:
    return {
        "dataset_preset": cfg.TRAIN_DATASET,
        "dataset": {
            "format": cfg.DATASET_FORMAT,
            "name": cfg.DATASET_NAME,
            "frameskip": cfg.DATASET_FRAMESKIP,
            "keys_to_load": list(cfg.DATASET_KEYS_TO_LOAD),
            "keys_to_cache": list(cfg.DATASET_KEYS_TO_CACHE),
            "keys_to_merge": dict(cfg.DATASET_KEYS_TO_MERGE),
        },
        "output_model_name": cfg.OUTPUT_MODEL_NAME,
        "runs_dir": Path(cfg.RUNS_DIR),
        "cache_dir": str(cfg.CACHE_DIR),
        "run_name": cfg.RUN_NAME,
        "train_split": cfg.TRAIN_SPLIT,
        "seed": cfg.SEED,
        "img_size": cfg.IMG_SIZE,
        "patch_size": cfg.PATCH_SIZE,
        "encoder_scale": cfg.ENCODER_SCALE,
        "trainer": {
            "max_epochs": cfg.MAX_EPOCHS,
            "accelerator": cfg.ACCELERATOR,
            "devices": cfg.DEVICES,
            "precision": cfg.PRECISION,
            "gradient_clip_val": cfg.GRADIENT_CLIP_VAL,
            "use_torch_compile": cfg.USE_TORCH_COMPILE,
        },
        "logging": {
            "console_every_steps": cfg.CONSOLE_EVERY_STEPS,
            "write_every_steps": cfg.WRITE_EVERY_STEPS,
            "plot_every_steps": cfg.PLOT_EVERY_STEPS,
            "plot_every_epochs": cfg.PLOT_EVERY_EPOCHS,
            "save_tsv": cfg.SAVE_TSV,
        },
        "loader": {
            "batch_size": cfg.BATCH_SIZE,
            "num_workers": cfg.NUM_WORKERS,
            "persistent_workers": cfg.PERSISTENT_WORKERS,
            "prefetch_factor": cfg.PREFETCH_FACTOR,
            "pin_memory": cfg.PIN_MEMORY,
        },
        "optimizer": {
            "name": cfg.OPTIMIZER_NAME,
            "lr": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
        },
        "wm": {
            "history_size": cfg.HISTORY_SIZE,
            "num_preds": cfg.NUM_PREDS,
            "embed_dim": cfg.EMBED_DIM,
            "action_dim": getattr(cfg, "ACTION_DIM", None),
            "use_learned_actions": cfg.USE_LEARNED_ACTIONS,
        },
        "codebook": {
            "num_codes": cfg.NUM_CODES,
            "beta": cfg.CODEBOOK_BETA,
        },
        "predictor": {
            "depth": cfg.PREDICTOR_DEPTH,
            "heads": cfg.PREDICTOR_HEADS,
            "mlp_dim": cfg.PREDICTOR_MLP_DIM,
            "dim_head": cfg.PREDICTOR_DIM_HEAD,
            "dropout": cfg.PREDICTOR_DROPOUT,
            "emb_dropout": cfg.PREDICTOR_EMB_DROPOUT,
        },
        "inverse_dynamics": {
            "depth": cfg.INVERSE_DYNAMICS_DEPTH,
            "heads": cfg.INVERSE_DYNAMICS_HEADS,
            "mlp_dim": cfg.INVERSE_DYNAMICS_MLP_DIM,
            "dim_head": cfg.INVERSE_DYNAMICS_DIM_HEAD,
            "dropout": cfg.INVERSE_DYNAMICS_DROPOUT,
            "emb_dropout": cfg.INVERSE_DYNAMICS_EMB_DROPOUT,
        },
        "loss": {
            "sigreg": {
                "weight": cfg.SIGREG_WEIGHT,
                "knots": cfg.SIGREG_KNOTS,
                "num_proj": cfg.SIGREG_NUM_PROJ,
            },
            "vq": {
                "codebook_weight": cfg.CODEBOOK_LOSS_WEIGHT,
                "commitment_weight": cfg.COMMITMENT_LOSS_WEIGHT,
            },
        },
    }


def create_run_dir(config: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{config['run_name']}" if config["run_name"] else f"-{config['dataset_preset']}"
    run_dir = config["runs_dir"] / f"{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_config(config: dict, run_dir: Path) -> None:
    serializable = dict(config)
    serializable["runs_dir"] = str(serializable["runs_dir"])
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def lejepa_forward(model, sigreg, batch, config: dict):
    import torch

    ctx_len = config["wm"]["history_size"]
    n_preds = config["wm"]["num_preds"]
    sigreg_weight = config["loss"]["sigreg"]["weight"]

    batch = dict(batch)
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    elif not config["wm"]["use_learned_actions"]:
        raise ValueError("Batch is missing 'action', but USE_LEARNED_ACTIONS is False.")

    output = model.encode(batch)
    emb = output["emb"]
    ctx_emb = emb[:, :ctx_len]

    if config["wm"]["use_learned_actions"]:
        code_features = model.infer_action_codes(emb[:, : ctx_len + n_preds])
        vq_output = model.quantize_action_codes(code_features[:, :ctx_len])
        ctx_act = vq_output["quantized"]
        output["code_indices"] = vq_output["indices"]
        output["codebook_loss"] = config["loss"]["vq"]["codebook_weight"] * vq_output["codebook_loss"]
        output["commitment_loss"] = (
            config["loss"]["vq"]["commitment_weight"] * vq_output["commitment_loss"]
        )
    else:
        ctx_act = output["act_emb"][:, :ctx_len]
        output["codebook_loss"] = emb.new_zeros(())
        output["commitment_loss"] = emb.new_zeros(())

    tgt_emb = emb[:, n_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = sigreg(emb.transpose(0, 1))
    output["loss"] = (
        output["pred_loss"]
        + sigreg_weight * output["sigreg_loss"]
        + output["codebook_loss"]
        + output["commitment_loss"]
    )
    return output


def resolve_device(config: dict):
    import torch

    accelerator = str(config["trainer"]["accelerator"]).lower()
    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        devices = config["trainer"]["devices"]
        if isinstance(devices, int):
            return torch.device(f"cuda:{devices}")
        if isinstance(devices, str) and devices not in {"auto", ""}:
            return torch.device(f"cuda:{int(devices.split(',')[0])}")
        if isinstance(devices, (list, tuple)) and devices:
            return torch.device(f"cuda:{int(devices[0])}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_amp(device, precision: str):
    import torch

    precision = str(precision).lower()
    if device.type != "cuda":
        return None, False
    if precision == "bf16":
        return torch.bfloat16, False
    if precision in {"16", "fp16", "float16"}:
        return torch.float16, True
    return None, False


def move_batch_to_device(batch, device):
    import torch

    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_dataset(config: dict):
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from video_dataset import MarioFrameSequenceDataset

    dataset_cfg = config["dataset"]
    if dataset_cfg["format"] == "mario_frames":
        dataset_root = Path(config["cache_dir"]) / dataset_cfg["name"]
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Mario dataset directory not found: {dataset_root}\n"
                f"Download and extract mario_data into that directory or change DATASET_NAME/CACHE_DIR in config.py."
            )
        return MarioFrameSequenceDataset(
            root=dataset_root,
            num_steps=config["wm"]["history_size"] + config["wm"]["num_preds"],
            img_size=config["img_size"],
        )

    cache_dir = config["cache_dir"] or str(swm.data.utils.get_cache_dir())
    dataset_path = Path(cache_dir) / f"{dataset_cfg['name']}.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Set CACHE_DIR in config.py to the directory containing {dataset_cfg['name']}.h5, "
            f"or change TRAIN_DATASET to a dataset you already have."
        )

    dataset = swm.data.HDF5Dataset(
        name=dataset_cfg["name"],
        num_steps=config["wm"]["history_size"] + config["wm"]["num_preds"],
        frameskip=dataset_cfg["frameskip"],
        keys_to_load=dataset_cfg["keys_to_load"],
        keys_to_cache=dataset_cfg["keys_to_cache"],
        keys_to_merge=dataset_cfg["keys_to_merge"],
        cache_dir=cache_dir,
        transform=None,
    )

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=config["img_size"])]
    for col in dataset_cfg["keys_to_load"]:
        if col.startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
        config["wm"][f"{col}_dim"] = dataset.get_dim(col)

    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def build_model(config: dict):
    import stable_pretraining as spt
    import torch

    from jepa import JEPA
    from module import ARPredictor, Embedder, InverseDynamicsTransformer, MLP, SIGReg, VectorQuantizer

    encoder = spt.backbone.utils.vit_hf(
        config["encoder_scale"],
        patch_size=config["patch_size"],
        image_size=config["img_size"],
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = config["wm"]["embed_dim"] or hidden_dim
    has_real_actions = "action" in config["dataset"]["keys_to_load"]
    effective_act_dim = None
    if has_real_actions:
        if config["wm"]["action_dim"] is None:
            raise ValueError("ACTION_DIM must be set when training with real action inputs.")
        effective_act_dim = config["dataset"]["frameskip"] * config["wm"]["action_dim"]

    predictor = ARPredictor(
        num_frames=config["wm"]["history_size"],
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **config["predictor"],
    )
    inverse_dynamics = InverseDynamicsTransformer(
        num_frames=config["wm"]["history_size"] + config["wm"]["num_preds"],
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=embed_dim,
        **config["inverse_dynamics"],
    )
    action_encoder = (
        Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        if has_real_actions
        else torch.nn.Identity()
    )
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
        num_codes=config["codebook"]["num_codes"],
        code_dim=embed_dim,
        beta=config["codebook"]["beta"],
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
    sigreg = SIGReg(
        knots=config["loss"]["sigreg"]["knots"],
        num_proj=config["loss"]["sigreg"]["num_proj"],
    )
    return model, sigreg


def build_optimizer(model, config: dict):
    import torch

    optimizer_cls = getattr(torch.optim, config["optimizer"]["name"])
    return optimizer_cls(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )


def build_scheduler(optimizer, total_train_steps: int):
    import torch

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_train_steps),
    )


def maybe_compile_model(model, config: dict):
    import torch

    if not config["trainer"].get("use_torch_compile", False):
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this PyTorch build; continuing without compile.")
        return model
    try:
        return torch.compile(model)
    except Exception as exc:
        print(f"torch.compile failed, continuing without compile: {exc}")
        return model


def empty_metrics() -> dict[str, float]:
    return {
        "loss": 0.0,
        "pred_loss": 0.0,
        "sigreg_loss": 0.0,
        "codebook_loss": 0.0,
        "commitment_loss": 0.0,
    }


def average_metrics(totals: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        return empty_metrics()
    return {key: value / count for key, value in totals.items()}


def metrics_row(
    *,
    split: str,
    epoch: int,
    epoch_step: int,
    global_step: int,
    lr: float | None,
    metrics: dict[str, float],
) -> dict[str, float | int | str | None]:
    return {
        "split": split,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "lr": lr,
        "loss": metrics["loss"],
        "pred_loss": metrics["pred_loss"],
        "sigreg_loss": metrics["sigreg_loss"],
        "codebook_loss": metrics["codebook_loss"],
        "commitment_loss": metrics["commitment_loss"],
    }


def evaluate(model, sigreg, loader, device, amp_dtype, config: dict, epoch: int, global_step: int):
    import torch
    from tqdm.auto import tqdm

    model.eval()
    sigreg.eval()

    totals = empty_metrics()
    total_batches = 0

    progress = tqdm(loader, desc=f"val {epoch}", leave=False)
    with torch.no_grad():
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = lejepa_forward(model, sigreg, batch, config)

            totals["loss"] += output["loss"].detach().item()
            totals["pred_loss"] += output["pred_loss"].detach().item()
            totals["sigreg_loss"] += output["sigreg_loss"].detach().item()
            totals["codebook_loss"] += output["codebook_loss"].detach().item()
            totals["commitment_loss"] += output["commitment_loss"].detach().item()
            total_batches += 1

    return metrics_row(
        split="val",
        epoch=epoch,
        epoch_step=0,
        global_step=global_step,
        lr=None,
        metrics=average_metrics(totals, total_batches),
    )


def train_one_epoch(
    *,
    model,
    sigreg,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    amp_dtype,
    config: dict,
    epoch: int,
    global_step_start: int,
    on_step_end,
):
    import torch
    from tqdm.auto import tqdm

    model.train()
    sigreg.train()

    running_totals = empty_metrics()
    epoch_totals = empty_metrics()
    running_batches = 0
    epoch_batches = 0
    grad_clip = config["trainer"]["gradient_clip_val"]

    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        global_step = global_step_start + batch_idx
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            output = lejepa_forward(model, sigreg, batch, config)
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
        if scheduler is not None:
            scheduler.step()

        step_metrics = {
            "loss": output["loss"].detach().item(),
            "pred_loss": output["pred_loss"].detach().item(),
            "sigreg_loss": output["sigreg_loss"].detach().item(),
            "codebook_loss": output["codebook_loss"].detach().item(),
            "commitment_loss": output["commitment_loss"].detach().item(),
        }

        for key, value in step_metrics.items():
            running_totals[key] += value
            epoch_totals[key] += value

        running_batches += 1
        epoch_batches += 1

        row = metrics_row(
            split="train",
            epoch=epoch,
            epoch_step=batch_idx,
            global_step=global_step,
            lr=optimizer.param_groups[0]["lr"],
            metrics=step_metrics,
        )
        on_step_end(row, batch_idx=batch_idx)

        if batch_idx % max(1, config["logging"]["console_every_steps"]) == 0:
            averaged = average_metrics(running_totals, running_batches)
            progress.set_postfix(
                loss=f"{averaged['loss']:.4f}",
                pred=f"{averaged['pred_loss']:.4f}",
                sigreg=f"{averaged['sigreg_loss']:.4f}",
                codebook=f"{averaged['codebook_loss']:.4f}",
                commit=f"{averaged['commitment_loss']:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            running_totals = empty_metrics()
            running_batches = 0

    return average_metrics(epoch_totals, epoch_batches)


def main():
    import stable_pretraining as spt
    import torch

    config = build_train_config()
    run_dir = create_run_dir(config)
    save_run_config(config, run_dir)

    torch.manual_seed(config["seed"])
    dataset = build_dataset(config)

    generator = torch.Generator().manual_seed(config["seed"])
    train_set, val_set = spt.data.random_split(
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

    model, sigreg = build_model(config)
    model = maybe_compile_model(model, config)
    device = resolve_device(config)
    amp_dtype, use_grad_scaler = resolve_amp(device, config["trainer"]["precision"])
    model = model.to(device)
    sigreg = sigreg.to(device)

    optimizer = build_optimizer(model, config)
    total_train_steps = config["trainer"]["max_epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_train_steps=total_train_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    metrics_jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    metrics_tsv = TsvLogger(run_dir / "metrics.tsv") if config["logging"]["save_tsv"] else None
    plot_rows: list[dict] = []
    pending_rows: list[dict] = []
    artifact_saver = ModelArtifactSaver(
        dirpath=run_dir,
        filename=config["output_model_name"],
        epoch_interval=1,
    )

    def flush_pending_rows():
        if not pending_rows:
            return
        for pending_row in pending_rows:
            metrics_jsonl.log(pending_row)
            if metrics_tsv is not None:
                metrics_tsv.log(pending_row)
        pending_rows.clear()

    def persist_row(row: dict, *, update_plot: bool = False, flush: bool = False):
        pending_rows.append(row)
        plot_rows.append(row)
        if flush or len(pending_rows) >= max(1, config["logging"]["write_every_steps"]):
            flush_pending_rows()
        if update_plot:
            save_training_plots(plot_rows, run_dir / "training_curves.png")

    global_step = 0
    for epoch in range(1, config["trainer"]["max_epochs"] + 1):
        epoch_train_metrics = train_one_epoch(
            model=model,
            sigreg=sigreg,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            config=config,
            epoch=epoch,
            global_step_start=global_step,
            on_step_end=lambda row, batch_idx: persist_row(
                row,
                update_plot=batch_idx % max(1, config["logging"]["plot_every_steps"]) == 0,
                flush=batch_idx % max(1, config["logging"]["write_every_steps"]) == 0,
            ),
        )
        global_step += len(train_loader)

        val_row = evaluate(
            model=model,
            sigreg=sigreg,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )
        persist_row(val_row, update_plot=True, flush=True)

        artifact_saver.save_epoch(model=model, epoch=epoch, max_epochs=config["trainer"]["max_epochs"])
        if epoch % max(1, config["logging"]["plot_every_epochs"]) == 0:
            save_training_plots(plot_rows, run_dir / "training_curves.png")

        print(
            f"epoch {epoch}/{config['trainer']['max_epochs']} "
            f"train_loss={epoch_train_metrics['loss']:.6f} "
            f"val_loss={val_row['loss']:.6f} "
            f"step={global_step}"
        )

    flush_pending_rows()
    artifact_saver.save_final(model=model)
    save_training_plots(plot_rows, run_dir / "training_curves.png")
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
