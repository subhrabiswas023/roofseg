from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from roofseg.common.training import train
from roofseg.segmentation.config import Config
from roofseg.segmentation.training import Module
from roofseg.segmentation.tracking import Tracker

import roofseg.factories as factories
from roofseg.segmentation.typing import ImageTensor, MaskTensor


def run_training_pipeline[C: Config](
    config: C,
    tracker: Tracker,
    environment_setter: Callable[[C], torch.device],
    dataloader_factory: Callable[
        [C, torch.device],
        tuple[
            DataLoader[tuple[ImageTensor, MaskTensor]],
            DataLoader[tuple[ImageTensor, MaskTensor]],
        ],
    ],
    transformer_factory: Callable[[C], torch.nn.Module],
    model_factory: Callable[[C], torch.nn.Module],
    criterion_factory: Callable[[C], torch.nn.Module],
    optimizer_factory: Callable[[C, torch.nn.Module], torch.optim.Optimizer],
) -> None:
    tracker.save_config(config)
    device = environment_setter(config)
    train_loader, val_loader = dataloader_factory(config, device)

    model = model_factory(config)
    train(
        Module(
            model,
            transformer_factory(config),
            criterion_factory(config),
            optimizer_factory(config, model),
        ),
        tracker,
        device,
        train_loader,
        val_loader,
        config.training.num_epochs,
    )


def run_default_training_pipeline() -> None:
    run_training_pipeline(
        Config(),
        Tracker(Path("out")),
        factories.setup_environment,
        factories.build_dataloaders,
        factories.build_transformer,
        factories.build_model,
        factories.build_criterion,
        factories.build_optimizer,
    )
