from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from common.training import Trainer
from segmentation.transforms import SyncedImageMaskTransform
from segmentation.training import Module
from segmentation.tracking import Tracker


def test_main(
    dataset: Dataset,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    tmp_path: Path
):
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = "cpu"
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)
    print(f"Using device: {DEVICE}")

    trainer = Trainer(
        Module(
            model, SyncedImageMaskTransform(torch.nn.Identity()), criterion, optimizer
        ),
        Tracker(
            tmp_path
        ),
        DEVICE,
    )

    trainer.fit(DataLoader(dataset, batch_size=1), DataLoader(dataset, batch_size=1), 1)
