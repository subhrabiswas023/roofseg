"""Module for loading and processing the dataset."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

type ImageArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type MaskArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]


class PatchedDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        get_mask_path_from_image_path: Callable[[Path], Path],
        image_width: int,
        image_height: int,
        color_threshold: int = 128,
        patch_size: int = 256,
    ) -> None:
        self.image_paths = sorted(image_paths)
        self.mask_fn = get_mask_path_from_image_path
        self.patch_size = patch_size
        self.color_threshold = color_threshold

        self.patches_per_row = image_width // self.patch_size
        self.patches_per_col = image_height // self.patch_size

        self.patches_per_image = self.patches_per_row * self.patches_per_col
        self.total_patches = self.patches_per_image * len(self.image_paths)

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_idx, row_idx, col_idx = get_indices(
            idx, self.patches_per_image, self.patches_per_row
        )

        image_path = self.image_paths[image_idx]
        mask_path = self.mask_fn(image_path)

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        image = (image / 255.0).astype(np.float32)
        mask = (mask > self.color_threshold).astype(np.int64)

        image = patchify(image, self.patch_size, row_idx, col_idx)
        mask = patchify(mask, self.patch_size, row_idx, col_idx)

        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long)

        image = image.permute(2, 0, 1)  # Convert (H, W, C) to (C, H, W) format

        return image, mask

    def _load_image(self, path: Path) -> ImageArray:
        with Image.open(path) as img:
            image = img.convert("RGB")
        image = np.array(image, dtype=np.float32)
        return image

    def _load_mask(self, path: Path) -> MaskArray:
        with Image.open(path) as msk:
            mask = msk.convert("L")
        mask = np.array(mask, dtype=np.int64)
        return mask


def get_indices(
    idx: int, patches_per_image: int, patches_per_row: int
) -> tuple[int, int, int]:
    image_idx, offset = divmod(idx, patches_per_image)
    row_idx, col_idx = divmod(offset, patches_per_row)  # Row major order
    return image_idx, row_idx, col_idx


def patchify(
    image: np.ndarray, patch_size: int, row_idx: int, col_idx: int
) -> np.ndarray:
    row_start = row_idx * patch_size
    row_end = row_start + patch_size

    col_start = col_idx * patch_size
    col_end = col_start + patch_size

    image = image[row_start:row_end, col_start:col_end]

    return image
