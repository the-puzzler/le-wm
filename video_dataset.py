from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


_FRAME_RE = re.compile(r"f(\d+)")


class MarioFrameSequenceDataset(Dataset):
    def __init__(self, root: str | Path, num_steps: int, img_size: int | tuple[int, int] = 224):
        self.root = Path(root)
        self.num_steps = num_steps
        if isinstance(img_size, int):
            resize_size = (img_size, img_size)
        else:
            resize_size = tuple(img_size)
        self.img_size = resize_size
        self.transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(self.img_size),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.sequences = self._build_sequences()
        if not self.sequences:
            raise RuntimeError(f"No frame sequences of length {num_steps} found under {self.root}")

    def _extract_frame_index(self, path: Path) -> int | None:
        match = _FRAME_RE.search(path.stem)
        if match is None:
            return None
        return int(match.group(1))

    def _build_sequences(self) -> list[list[Path]]:
        image_paths = sorted(
            list(self.root.rglob("*.png")) + list(self.root.rglob("*.jpg")) + list(self.root.rglob("*.jpeg"))
        )
        groups: dict[Path, list[tuple[int, Path]]] = {}
        for path in image_paths:
            frame_idx = self._extract_frame_index(path)
            if frame_idx is None:
                continue
            groups.setdefault(path.parent, []).append((frame_idx, path))

        sequences: list[list[Path]] = []
        for items in groups.values():
            items.sort(key=lambda item: item[0])
            frame_indices = [idx for idx, _ in items]
            paths = [path for _, path in items]
            for start in range(0, len(paths) - self.num_steps + 1):
                window_indices = frame_indices[start : start + self.num_steps]
                if window_indices[-1] - window_indices[0] != self.num_steps - 1:
                    continue
                if any((b - a) != 1 for a, b in zip(window_indices[:-1], window_indices[1:])):
                    continue
                sequences.append(paths[start : start + self.num_steps])
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frames = []
        for path in self.sequences[index]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                frame = self.transform(np.array(image))
            frames.append(frame)
        pixels = torch.stack(frames, dim=0)
        return {"pixels": pixels}
