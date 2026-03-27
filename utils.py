import numpy as np
import torch
import json
from pathlib import Path
from stable_pretraining import data as dt

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
