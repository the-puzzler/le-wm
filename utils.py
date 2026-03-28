import os
import json
import csv
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lets CLI load before deps are installed.
    torch = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    from stable_pretraining import data as dt

    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    from stable_pretraining import data as dt
    import torch

    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelArtifactSaver:
    """Persist model objects and state dicts during plain PyTorch training."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def save_epoch(self, model, epoch: int, max_epochs: int):
        if (epoch % self.epoch_interval) != 0 and epoch != max_epochs:
            return

        self._dump_model(model, self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt")
        self._dump_state(model, self.dirpath / f"{self.filename}_epoch_{epoch}_weights.ckpt")

    def save_final(self, model):
        self._dump_model(model, self.dirpath / f"{self.filename}_object.ckpt")
        self._dump_state(model, self.dirpath / f"{self.filename}_weights.ckpt")

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")

    def _dump_state(self, model, path):
        try:
            torch.save(model.state_dict(), path)
        except Exception as e:
            print(f"Error saving model weights: {e}")


class JsonlLogger:
    """Append newline-delimited JSON metrics to disk."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict):
        serializable = {}
        for key, value in payload.items():
            if torch is not None and isinstance(value, torch.Tensor):
                serializable[key] = float(value.detach().cpu().item())
            elif isinstance(value, np.generic):
                serializable[key] = value.item()
            else:
                serializable[key] = value

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable) + "\n")


class TsvLogger:
    """Append tab-separated metrics to disk with a single header row."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = None

    def log(self, payload: dict):
        serializable = {}
        for key, value in payload.items():
            if torch is not None and isinstance(value, torch.Tensor):
                serializable[key] = float(value.detach().cpu().item())
            elif isinstance(value, np.generic):
                serializable[key] = value.item()
            else:
                serializable[key] = value

        incoming_fields = list(serializable.keys())
        if self._fieldnames is None:
            self._fieldnames = incoming_fields
            if self.path.exists() and self.path.stat().st_size > 0:
                with self.path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f, delimiter="\t")
                    self._fieldnames = next(reader)
        elif any(field not in self._fieldnames for field in incoming_fields):
            self._fieldnames = self._fieldnames + [
                field for field in incoming_fields if field not in self._fieldnames
            ]
            self._rewrite_with_fieldnames()

        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, delimiter="\t")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({k: serializable.get(k, "") for k in self._fieldnames})

    def _rewrite_with_fieldnames(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return

        with self.path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})


def save_training_plots(history: list[dict], output_path):
    """Save training curves from per-step train rows and per-epoch validation rows."""
    if not history:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in history if row.get("split") == "train"]
    val_rows = [row for row in history if row.get("split") == "val"]
    if not train_rows and not val_rows:
        return

    train_steps = [row["global_step"] for row in train_rows]
    val_steps = [row["global_step"] for row in val_rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()

    plots = [
        ("Total Loss", "loss"),
        ("Prediction Loss", "pred_loss"),
        ("SIGReg Loss", "sigreg_loss"),
        ("Codebook Loss", "codebook_loss"),
        ("Commitment Loss", "commitment_loss"),
        ("Learning Rate", "lr", None),
    ]

    for ax, plot_spec in zip(axes, plots):
        title = plot_spec[0]
        metric_key = plot_spec[1]

        if train_rows:
            ax.plot(
                train_steps,
                [row[metric_key] for row in train_rows],
                label=f"train/{metric_key}",
                linewidth=1.5,
            )
        if val_rows and metric_key != "lr":
            ax.plot(
                val_steps,
                [row[metric_key] for row in val_rows],
                label=f"val/{metric_key}",
                linewidth=2.0,
                marker="o",
            )
        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
