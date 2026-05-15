import numpy as np

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