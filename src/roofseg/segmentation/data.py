"""Module for loading and processing the dataset."""

from typing import override
from functools import lru_cache
from collections.abc import Callable
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

from roofseg.common.data import get_indices, patchify
from roofseg.segmentation.typing import ImageArray, MaskArray, ImageTensor, MaskTensor


class PatchedDataset(Dataset[tuple[ImageTensor, MaskTensor]]):
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

    @override
    def __getitem__(self, idx: int) -> tuple[ImageTensor, MaskTensor]:
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

    @lru_cache(maxsize=1)
    def _load_image(self, path: Path) -> ImageArray:
        with Image.open(path) as img:
            image = img.convert("RGB")
        image = np.array(image, dtype=np.float32)
        return image

    @lru_cache(maxsize=1)
    def _load_mask(self, path: Path) -> MaskArray:
        with Image.open(path) as msk:
            mask = msk.convert("L")
        mask = np.array(mask, dtype=np.int64)
        return mask
