from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from roofseg.common.training import train
from roofseg.segmentation.config import Config
from roofseg.segmentation.training import Module
from roofseg.segmentation.tracking import Tracker

import roofseg.factories as factories
from roofseg.segmentation.typing import PairedTensor


def run_training_pipeline[C: Config](
    config: C,
    tracker: Tracker,
    environment_setter: Callable[[C], torch.device],
    train_loader_factory: Callable[[C], DataLoader[PairedTensor]],
    val_loader_factory: Callable[[C], DataLoader[PairedTensor]],
    transformer_factory: Callable[[C], torch.nn.Module],
    model_factory: Callable[[C], torch.nn.Module],
    criterion_factory: Callable[[C], torch.nn.Module],
    optimizer_factory: Callable[[C, torch.nn.Module], torch.optim.Optimizer],
) -> None:
    tracker.save_config(config)
    device = environment_setter(config)

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
        train_loader_factory(config),
        val_loader_factory(config),
        config.training.num_epochs,
    )


def run_default_training_pipeline() -> None:
    run_training_pipeline(
        Config(),
        Tracker(Path("out")),
        factories.setup_environment,
        factories.build_train_loader,
        factories.build_val_loader,
        factories.build_transformer,
        factories.build_model,
        factories.build_criterion,
        factories.build_optimizer,
    )
