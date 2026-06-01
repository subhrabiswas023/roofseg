from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from roofseg.common.training import train
from roofseg.factories import setup_environment
from roofseg.pipeline import run_training_pipeline
from roofseg.segmentation.config import Config
from roofseg.segmentation.tracking import LocalTracker
from roofseg.segmentation.training import Module
from roofseg.segmentation.transforms import SyncedImageMaskTransform


def test_main(
    config: Config,
    dataset: Dataset,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    tmp_path: Path
):
    pass
