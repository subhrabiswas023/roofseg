from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import kornia.augmentation as K

from config import Config
from dataset import PatchedDataset
from model import build_model
from training import SegmentationModule, Trainer, CombinedLoss, SyncedImageMaskTransform

config = Config(
    seed=42,
    device="cuda",
    root_dir="data",  # currently doesn't exist
    image_dir="images",
    mask_dir="masks",
    image_height=1024,
    image_width=1024,
    num_classes=2,
    color_threshold=128,
    patch_size=256,
    horizontal_flip_prob=0.5,
    vertical_flip_prob=0.5,
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    batch_size=16,
    num_epochs=10,
    optimizer="Adam",
    learning_rate=0.001,
    criterion="CombinedLoss",
    loss_alpha=0.5,
)


def main():
    # Environment setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    DEVICE = torch.device(config.device)

    # Dataset and dataloader
    DeployableDataset = partial(
        PatchedDataset,
        image_width=config.image_width,
        image_height=config.image_height,
        color_threshold=config.color_threshold,
        patch_size=config.patch_size,
    )
    
    train_path = Path(config.root_dir) / "train"
    train_image_dir = train_path / config.image_dir
    train_mask_dir = train_path / config.mask_dir
    train_dataset = DeployableDataset(
        image_paths=list(train_image_dir.iterdir()),
        get_mask_path_from_image_path=lambda p: train_mask_dir / p.name,
    )

    val_path = Path(config.root_dir) / "val"
    val_image_dir = val_path / config.image_dir
    val_mask_dir = val_path / config.mask_dir
    val_dataset = DeployableDataset(
        image_paths=list(val_image_dir.iterdir()),
        get_mask_path_from_image_path=lambda p: val_mask_dir / p.name,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=(DEVICE.type == "cuda")
    )
    
    # Augmentation setup
    transform = SyncedImageMaskTransform(
        spatial_transform=torch.nn.Sequential(
            K.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            K.RandomVerticalFlip(p=config.vertical_flip_prob),
        )
    ).to(DEVICE)

    # Model setup
    model = build_model(
        config.encoder_name, config.encoder_weights, config.num_classes
    ).to(DEVICE)
    
    # Training setup
    criterion = CombinedLoss(alpha=config.loss_alpha, mode="multiclass")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    trainer = Trainer(
        SegmentationModule(model, criterion, optimizer, DEVICE),
        transform,
        DEVICE,
    )

    trainer.fit(
        train_loader,
        val_loader,
        config.num_epochs
    )

if __name__ == "__main__":
    main()
