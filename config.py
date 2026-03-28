from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    name: str
    frameskip: int
    keys_to_load: list[str]
    keys_to_cache: list[str]
    keys_to_merge: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    accelerator: str = "gpu"
    devices: str | int | list[int] = "auto"
    precision: str = "bf16"
    gradient_clip_val: float | None = 1.0


@dataclass
class LoggingConfig:
    console_every_steps: int = 20
    write_every_steps: int = 100
    plot_every_steps: int = 100
    plot_every_epochs: int = 1
    save_tsv: bool = True


@dataclass
class LoaderConfig:
    batch_size: int = 16
    num_workers: int = 6
    persistent_workers: bool = True
    prefetch_factor: int = 3
    pin_memory: bool = True


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    lr: float = 5e-5
    weight_decay: float = 1e-3


@dataclass
class WorldModelConfig:
    history_size: int = 3
    num_preds: int = 1
    embed_dim: int = 192
    action_dim: int = 2
    use_learned_actions: bool = True


@dataclass
class CodebookConfig:
    num_codes: int = 8
    beta: float = 0.25


@dataclass
class PredictorConfig:
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0


@dataclass
class InverseDynamicsConfig:
    depth: int = 4
    heads: int = 8
    mlp_dim: int = 1024
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0


@dataclass
class SigRegConfig:
    weight: float = 0.09
    knots: int = 17
    num_proj: int = 1024


@dataclass
class VQLossConfig:
    codebook_weight: float = 1.0
    commitment_weight: float = 1.0


@dataclass
class LossConfig:
    sigreg: SigRegConfig = field(default_factory=SigRegConfig)
    vq: VQLossConfig = field(default_factory=VQLossConfig)


@dataclass
class TrainConfig:
    dataset_preset: str = "pusht"
    output_model_name: str = "lewm"
    runs_dir: str = "runs"
    cache_dir: str | None = None
    run_name: str | None = None
    train_split: float = 0.9
    seed: int = 3072
    img_size: int = 224
    patch_size: int = 14
    encoder_scale: str = "tiny"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wm: WorldModelConfig = field(default_factory=WorldModelConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    inverse_dynamics: InverseDynamicsConfig = field(default_factory=InverseDynamicsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig | None = None


def dataset_presets(num_steps: int) -> dict[str, DatasetConfig]:
    return {
        "pusht": DatasetConfig(
            name="pusht_expert_train",
            frameskip=5,
            keys_to_load=["pixels", "action", "proprio", "state"],
            keys_to_cache=["action", "proprio", "state"],
        ),
        "tworoom": DatasetConfig(
            name="tworoom",
            frameskip=5,
            keys_to_load=["pixels", "action", "proprio"],
            keys_to_cache=["action", "proprio"],
        ),
        "dmc": DatasetConfig(
            name="reacher",
            frameskip=5,
            keys_to_load=["pixels", "action", "observation"],
            keys_to_cache=["action", "observation"],
        ),
        "ogb": DatasetConfig(
            name="ogbench/cube_single_expert",
            frameskip=5,
            keys_to_load=["pixels", "action", "observation"],
            keys_to_cache=["action", "observation"],
            keys_to_merge={"proprio": "proprio"},
        ),
    }


def build_config(dataset_preset: str = "pusht") -> TrainConfig:
    config = TrainConfig(dataset_preset=dataset_preset)
    num_steps = config.wm.history_size + config.wm.num_preds
    presets = dataset_presets(num_steps=num_steps)
    if dataset_preset not in presets:
        valid = ", ".join(sorted(presets))
        raise ValueError(f"Unknown dataset preset '{dataset_preset}'. Expected one of: {valid}")
    config.dataset = presets[dataset_preset]
    return config


CONFIG = build_config()
CONFIG.cache_dir = str(Path(__file__).resolve().parent / "data" / "stablewm")
