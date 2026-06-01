import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from roofseg.common.data import get_indices, patchify


def test_dataset(dataset: Dataset):
    image_tensor, mask_tensor = dataset[0]

    assert image_tensor.shape == (3, 256, 256)  # C, H, W
    assert mask_tensor.shape == (256, 256)  # H, W

    assert image_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.int64


@pytest.mark.parametrize(
    "idx, expected",
    [
        (0, (0, 0, 0)),
        (3, (0, 0, 3)),
        (4, (0, 1, 0)),
        (11, (0, 2, 3)),
        (12, (0, 3, 0)),
        (25, (1, 2, 1)),
    ],
)
def test_get_indices_valid(idx, expected):
    assert get_indices(idx, patches_per_image=16, patches_per_row=4) == expected


@pytest.mark.parametrize(
    "idx, expected",
    [
        (7, (0, 1, 2)),
        (14, (0, 2, 4)),
        (15, (1, 0, 0)),
        (29, (1, 2, 4)),
    ],
)
def test_get_indices_rectangular(idx, expected):
    assert get_indices(idx, patches_per_image=15, patches_per_row=5) == expected


@pytest.mark.parametrize(
    "row_idx, col_idx, expected",
    [
        (0, 0, np.array([[0, 1], [4, 5]])),
        (0, 1, np.array([[2, 3], [6, 7]])),
        (1, 0, np.array([[8, 9], [12, 13]])),
        (1, 1, np.array([[10, 11], [14, 15]])),
    ],
)
def test_patchify(row_idx, col_idx, expected):
    image = np.arange(16).reshape(4, 4)
    patch_size = 2

    patch = patchify(image, patch_size, row_idx, col_idx)
    assert np.array_equal(patch, expected)
