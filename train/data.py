"""The two dataset readers needed by the three paper training steps."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset"
ANGLES = (0, 45, 90, 135)


def pack_raw(raw):
    if raw.ndim != 2 or raw.shape[0] % 4 or raw.shape[1] % 4:
        raise ValueError(f"RAW shape must be H×W and divisible by 4; got {raw.shape}")
    return torch.from_numpy(np.stack([raw[row::4, col::4] for row in range(4) for col in range(4)]).astype(np.float32))


def read_raw(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def read_labels(label_dir, scene):
    labels = []
    for angle in ANGLES:
        path = label_dir / f"{scene}_{angle}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        labels.append(torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255)
    return torch.stack(labels)


def crop_arrays(arrays, crop_size, random_crop):
    if crop_size is None:
        return arrays
    if crop_size < 64 or crop_size % 4:
        raise ValueError("crop size must be a multiple of 4 and at least 64")
    height, width = arrays[0].shape[-2:]
    if crop_size > min(height, width):
        raise ValueError(f"crop {crop_size} exceeds image {height}×{width}")
    row = random.randrange((height - crop_size) // 4 + 1) * 4 if random_crop else ((height - crop_size) // 8) * 4
    col = random.randrange((width - crop_size) // 4 + 1) * 4 if random_crop else ((width - crop_size) // 8) * 4
    return [array[..., row:row + crop_size, col:col + crop_size] for array in arrays]


class NoisyRawDataset(Dataset):
    """FinalDataHigh: noisy RAW, clean RAW, and four RGB labels."""

    def __init__(self, split, crop_size=128, include_labels=True):
        if split not in ("train", "valid", "test"):
            raise ValueError(split)
        self.root = DATA_ROOT / "FinalDataHigh"
        self.split = split
        self.crop_size = crop_size
        self.include_labels = include_labels
        self.files = sorted((self.root / f"input_{split}").glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {self.root / f'input_{split}'}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        scene = path.stem.split("_")[0]
        noisy = torch.from_numpy(read_raw(path))
        clean = torch.from_numpy(read_raw(self.root / "label" / f"{scene}.npy"))
        if noisy.shape != clean.shape:
            raise ValueError(f"Mismatched RAW shapes for {path.name}")
        if self.include_labels:
            labels = read_labels(self.root / "label", scene)
            if noisy.shape != labels.shape[-2:]:
                raise ValueError(f"Mismatched RAW/RGB shapes for {path.name}")
            noisy, clean, labels = crop_arrays([noisy, clean, labels], self.crop_size, self.split == "train")
            return pack_raw(noisy.numpy()), pack_raw(clean.numpy()), labels
        noisy, clean = crop_arrays([noisy, clean], self.crop_size, self.split == "train")
        return pack_raw(noisy.numpy()), pack_raw(clean.numpy())


class CleanRawDataset(Dataset):
    """FinalDmData provides split/labels; FinalDataHigh provides aligned clean RAW."""

    def __init__(self, split, crop_size=128):
        if split not in ("train", "valid", "test"):
            raise ValueError(split)
        self.dm_root = DATA_ROOT / "FinalDmData"
        self.raw_root = DATA_ROOT / "FinalDataHigh" / "label"
        self.split = split
        self.crop_size = crop_size
        self.scenes = sorted(path.stem for path in (self.dm_root / split).glob("*.npy"))
        if not self.scenes:
            raise FileNotFoundError(f"No .npy files in {self.dm_root / split}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.scenes[index]
        clean = torch.from_numpy(read_raw(self.raw_root / f"{scene}.npy"))
        labels = read_labels(self.dm_root / "label", scene)
        if clean.shape != labels.shape[-2:]:
            raise ValueError(f"Mismatched RAW/RGB shapes for {scene}")
        clean, labels = crop_arrays([clean, labels], self.crop_size, self.split == "train")
        return pack_raw(clean.numpy()), labels
