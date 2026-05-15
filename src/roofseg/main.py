from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import kornia.augmentation as K

from roofseg.common.training import Trainer, Phase
from roofseg.segmentation.config import (
    Config,
    EnvironmentConfig,
    DatasetConfig,
    AugmentationConfig,
    ModelConfig,
    CriterionConfig,
    OptimizerConfig,
    TrainingConfig,
)
from roofseg.segmentation.data import PatchedDataset
from roofseg.segmentation.transforms import SyncedImageMaskTransform
from roofseg.segmentation.losses import CombinedLoss
from roofseg.segmentation.model import build_model
from roofseg.segmentation.training import Module
from roofseg.segmentation.tracking import Tracker

config = Config(
    EnvironmentConfig(seed=42, device="cuda"),
    DatasetConfig(
        root_dir="data",  # currently doesn't exist
        image_dir="images",
        mask_dir="masks",
        image_height=1024,
        image_width=1024,
        num_classes=2,
        color_threshold=128,
        patch_size=256,
    ),
    AugmentationConfig(horizontal_flip_prob=0.5, vertical_flip_prob=0.5),
    ModelConfig(encoder_name="mobilenet_v2", encoder_weights="imagenet"),
    CriterionConfig(criterion="CombinedLoss", loss_alpha=0.5),
    OptimizerConfig(optimizer="Adam", learning_rate=0.001),
    TrainingConfig(batch_size=16, num_epochs=10),
)


def main():
    tracker = Tracker(Path("out"))

    # Saving the config
    tracker.save_config(config)

    # Environment setup
    torch.manual_seed(config.environment.seed)
    np.random.seed(config.environment.seed)

    DEVICE = torch.device(config.environment.device)

    # Dataset and dataloader
    DeployableDataset = partial(
        PatchedDataset,
        image_width=config.dataset.image_width,
        image_height=config.dataset.image_height,
        color_threshold=config.dataset.color_threshold,
        patch_size=config.dataset.patch_size,
    )

    train_path = Path(config.dataset.root_dir) / Phase.TRAIN
    train_image_dir = train_path / config.dataset.image_dir
    train_mask_dir = train_path / config.dataset.mask_dir
    train_dataset = DeployableDataset(
        image_paths=list(train_image_dir.iterdir()),
        get_mask_path_from_image_path=lambda p: train_mask_dir / p.name,
    )

    val_path = Path(config.dataset.root_dir) / Phase.VAL
    val_image_dir = val_path / config.dataset.image_dir
    val_mask_dir = val_path / config.dataset.mask_dir
    val_dataset = DeployableDataset(
        image_paths=list(val_image_dir.iterdir()),
        get_mask_path_from_image_path=lambda p: val_mask_dir / p.name,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Augmentation setup
    transform = SyncedImageMaskTransform(
        spatial_transform=torch.nn.Sequential(
            K.RandomHorizontalFlip(p=config.augmentation.horizontal_flip_prob),
            K.RandomVerticalFlip(p=config.augmentation.vertical_flip_prob),
        )
    ).to(DEVICE)

    # Model setup
    model = build_model(
        config.model.encoder_name,
        config.model.encoder_weights,
        config.dataset.num_classes,
    ).to(DEVICE)

    # Training setup
    criterion = CombinedLoss(alpha=config.criterion.loss_alpha, mode="multiclass")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)

    trainer = Trainer(
        Module(model, transform, criterion, optimizer),
        tracker,
        DEVICE,
    )

    trainer.fit(train_loader, val_loader, config.training.num_epochs)


if __name__ == "__main__":
    main()
