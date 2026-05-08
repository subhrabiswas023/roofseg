import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from training import Trainer, SegmentationModule, SyncedImageMaskTransform


def test_main(
    dataset: Dataset,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
):
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = "cpu"
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)
    print(f"Using device: {DEVICE}")
    
    trainer = Trainer(
        SegmentationModule(model, criterion, optimizer, DEVICE),
        SyncedImageMaskTransform(torch.nn.Identity()),
        DEVICE,
    )

    trainer.fit(
        DataLoader(dataset, batch_size=1),
        DataLoader(dataset, batch_size=1),
        1
    )
