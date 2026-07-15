from functools import partial
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from roofseg.common.training import Phase
from roofseg.segmentation.config import Config
from roofseg.segmentation.data import PatchedDataset
from roofseg.segmentation.losses import CombinedLoss, L1Regularizer
from roofseg.segmentation.attentions import (
    CoordinateAttention,
    EfficientChannelAttention,
)
from roofseg.segmentation.tracking import LocalTracker, LocalRestorer, PathContext
from roofseg.segmentation.transforms import (
    ScaleImage,
    SyncedImageMaskTransform,
    ImageTransform,
)
from roofseg.segmentation.typing import PairedTensor


def build_tracker():
    return LocalTracker(paths=PathContext(root_dir=Path("out")))


def build_restorer():
    return LocalRestorer(paths=PathContext(root_dir=Path(".")))


def setup_environment(config: Config) -> torch.device:
    torch.manual_seed(config.environment.seed)
    np.random.seed(config.environment.seed)

    device = torch.device(config.environment.device)
    return device


def _build_loader(config: Config, phase: Phase) -> DataLoader[PairedTensor]:
    phase_path = Path(config.environment.input_dir) / config.dataset.root_dir / phase
    image_dir = phase_path / config.dataset.image_dir
    mask_dir = phase_path / config.dataset.mask_dir

    return DataLoader(
        PatchedDataset(
            image_paths=list(image_dir.iterdir()),
            get_mask_path_from_image_path=lambda p: mask_dir / p.name,
            image_width=config.dataset.image_width,
            image_height=config.dataset.image_height,
            color_threshold=config.dataset.color_threshold,
            patch_size=config.dataset.patch_size,
        ),
        batch_size=config.training.batch_size,
        shuffle=phase == Phase.TRAIN,
        pin_memory=(config.environment.device == "cuda"),
    )


build_train_loader = partial(_build_loader, phase=Phase.TRAIN)
build_val_loader = partial(_build_loader, phase=Phase.VAL)


def _build_transformer(
    config: Config, synced_image_mask_tranform: torch.nn.Module
) -> torch.nn.Module:
    return ImageTransform(
        ScaleImage(),
        synced_image_mask_tranform,
        K.Normalize(
            mean=torch.tensor(config.transformation.normalization_mean),
            std=torch.tensor(config.transformation.normalization_std),
        ),
    )


def build_train_transformer(config: Config) -> torch.nn.Module:
    return _build_transformer(
        config,
        SyncedImageMaskTransform(
            spatial_transform=torch.nn.Sequential(
                K.RandomHorizontalFlip(p=config.transformation.horizontal_flip_prob),
                K.RandomVerticalFlip(p=config.transformation.vertical_flip_prob),
            )
        ),
    )


def build_val_transformer(config: Config) -> torch.nn.Module:
    return _build_transformer(config, torch.nn.Identity())


def build_model(config: Config) -> torch.nn.Module:
    model = smp.UnetPlusPlus(
        config.model.encoder_name,
        encoder_weights=config.model.encoder_weights,
        classes=config.dataset.num_classes,
    )

    in_channels = model.segmentation_head[0].in_channels

    model.segmentation_head = torch.nn.Sequential(  # type: ignore
        EfficientChannelAttention(channels=in_channels),  # type: ignore
        CoordinateAttention(channels=in_channels),  # type: ignore
        torch.nn.Dropout2d(p=config.regularization.dropout),
        model.segmentation_head,
    )

    return model


def build_criterion(config: Config) -> torch.nn.Module:
    return CombinedLoss(alpha=config.criterion.loss_alpha, mode="multiclass")


def build_loss_regularizer(config: Config):
    return L1Regularizer(config.criterion.l1_lambda)


def build_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
