import json
import csv
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
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
            if isinstance(value, torch.Tensor):
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
            if isinstance(value, torch.Tensor):
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
    """Save a compact training dashboard with loss subplots."""
    if not history:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [row.get("global_step", row["epoch"]) for row in history]
    x_label = "Global Step" if any("global_step" in row for row in history) else "Epoch"
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()

    plots = [
        ("Total Loss", "train/loss", "val/loss"),
        ("Prediction Loss", "train/pred_loss", "val/pred_loss"),
        ("SIGReg Loss", "train/sigreg_loss", "val/sigreg_loss"),
        ("Codebook Loss", "train/codebook_loss", "val/codebook_loss"),
        ("Commitment Loss", "train/commitment_loss", "val/commitment_loss"),
        ("Learning Rate", "lr", None),
    ]

    for ax, (title, train_key, val_key) in zip(axes, plots):
        train_points = [(x, row[train_key]) for x, row in zip(x_values, history) if row.get(train_key) is not None]
        if train_points:
            ax.plot(
                [x for x, _ in train_points],
                [value for _, value in train_points],
                label=train_key,
                linewidth=2,
            )
        if val_key is not None:
            val_points = [(x, row[val_key]) for x, row in zip(x_values, history) if row.get(val_key) is not None]
            if val_points:
                ax.plot(
                    [x for x, _ in val_points],
                    [value for _, value in val_points],
                    label=val_key,
                    linewidth=2,
                )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
