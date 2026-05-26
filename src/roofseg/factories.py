from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import kornia.augmentation as K
import segmentation_models_pytorch as smp

from roofseg.common.training import Phase
from roofseg.segmentation.config import Config
from roofseg.segmentation.data import PatchedDataset
from roofseg.segmentation.transforms import SyncedImageMaskTransform
from roofseg.segmentation.losses import CombinedLoss
from roofseg.segmentation.typing import PairedTensor


def setup_environment(config: Config) -> torch.device:
    torch.manual_seed(config.environment.seed)
    np.random.seed(config.environment.seed)

    device = torch.device(config.environment.device)
    return device


def _build_loader(phase: Phase, config: Config) -> DataLoader[PairedTensor]:
    DeployableDataset = partial(
        PatchedDataset,
        image_width=config.dataset.image_width,
        image_height=config.dataset.image_height,
        color_threshold=config.dataset.color_threshold,
        patch_size=config.dataset.patch_size,
    )

    phase_path = Path(config.dataset.root_dir) / phase
    image_dir = phase_path / config.dataset.image_dir
    mask_dir = phase_path / config.dataset.mask_dir

    return DataLoader(
        DeployableDataset(
            image_paths=list(image_dir.iterdir()),
            get_mask_path_from_image_path=lambda p: mask_dir / p.name,
        ),
        batch_size=config.training.batch_size,
        shuffle=phase == Phase.TRAIN,
        pin_memory=(config.environment.device == "cuda"),
    )


build_train_loader = partial(_build_loader, phase=Phase.TRAIN)
build_val_loader = partial(_build_loader, phase=Phase.VAL)


def build_transformer(config: Config) -> torch.nn.Module:
    return SyncedImageMaskTransform(
        spatial_transform=torch.nn.Sequential(
            K.RandomHorizontalFlip(p=config.augmentation.horizontal_flip_prob),
            K.RandomVerticalFlip(p=config.augmentation.vertical_flip_prob),
        )
    )


def build_model(config: Config) -> torch.nn.Module:
    return smp.Unet(
        config.model.encoder_name,
        encoder_weights=config.model.encoder_weights,
        classes=config.dataset.num_classes,
    )


def build_criterion(config: Config) -> torch.nn.Module:
    return CombinedLoss(alpha=config.criterion.loss_alpha, mode="multiclass")


def build_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)
