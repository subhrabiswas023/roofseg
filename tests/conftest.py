import numpy as np
import pytest
import torch

from pathlib import Path
from PIL import Image

from roofseg.segmentation.data import PatchedDataset
from roofseg.segmentation.model import build_model
from roofseg.segmentation.losses import CombinedLoss

# Some data

@pytest.fixture(scope="session")
def random_images():
    return torch.rand(2, 3, 256, 256)

@pytest.fixture(scope="session")
def random_masks():
    return torch.randint(0, 2, (2, 256, 256))

@pytest.fixture(scope="session")
def images():
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    image[:, :, 0, 0] = 1.0
    return image

@pytest.fixture(scope="session")
def masks():
    mask = torch.zeros((1, 2, 2), dtype=torch.int64)
    mask[:, 0, 0] = 1
    return mask

@pytest.fixture
def dataset(tmp_path: Path) -> PatchedDataset:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"

    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask = np.zeros((1024, 1024), dtype=np.uint8)

    Image.fromarray(image, mode="RGB").save(image_dir / "sample.jpg")
    Image.fromarray(mask, mode="L").save(mask_dir / "sample.jpg")

    dataset = PatchedDataset(
        image_paths=[image_dir / "sample.jpg"],
        get_mask_path_from_image_path=lambda p: mask_dir / p.name,
        image_width=1024,
        image_height=1024,
        color_threshold=128,
        patch_size=256,
    )
    return dataset

# training fixtures

@pytest.fixture
def model():
    return build_model("mobilenet_v2", "imagenet", 2)

@pytest.fixture
def optimizer(model: torch.nn.Module):
    return torch.optim.Adam(model.parameters(), lr=0.001)

@pytest.fixture(scope="session")
def criterion():
    return CombinedLoss(alpha=0.5, mode="multiclass")
    
