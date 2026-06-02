import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset

from roofseg.factories import build_criterion, build_model, build_optimizer
from roofseg.segmentation.config import Config, EnvironmentConfig, TrainingConfig
from roofseg.segmentation.data import PatchedDataset

# Some data

@pytest.fixture(scope="session")
def random_images():
    return torch.rand(2, 3, 256, 256)

@pytest.fixture(scope="session")
def random_masks():
    return torch.randint(0, 2, (2, 256, 256))

@pytest.fixture(scope="session")
def mock_images():
    image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    image[:, :, 0, 0] = 1.0
    return image

@pytest.fixture(scope="session")
def mock_masks():
    mask = torch.zeros((1, 2, 2), dtype=torch.int64)
    mask[:, 0, 0] = 1
    return mask

@pytest.fixture
def mock_dataset(tmp_path) -> Dataset:
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

@pytest.fixture(scope="session")
def mock_config() -> Config:
    return Config(
       environment=EnvironmentConfig(
           device="cpu"
       ),
       training=TrainingConfig(
           batch_size=1,
           num_epochs=1
       )
    )
    
@pytest.fixture(scope="session")
def mock_criterion(mock_config) -> nn.Module:
    return build_criterion(mock_config)

# NOTE: Stale model and optimizer are used to save time because there is no test for the parameter values
    
@pytest.fixture(scope="session")
def mock_model(mock_config) -> nn.Module:
    return build_model(mock_config)

@pytest.fixture(scope="session")
def mock_optimizer(mock_config, mock_model) -> optim.Optimizer:
    return build_optimizer(mock_config, mock_model)
