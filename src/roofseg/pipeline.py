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
    tracker.save_config(config.to_dict())
    device = environment_setter(config)

    model = model_factory(config)

    model_state = tracker.restore_model()
    if model_state:
        model.load_state_dict(model_state)

    optimizer = optimizer_factory(config, model)
    optimizer_state = tracker.restore_optimizer()
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    train(
        module=Module(
            model,
            transformer_factory(config),
            criterion_factory(config),
            optimizer,
        ),
        device=device,
        train_loader=train_loader_factory(config),
        val_loader=val_loader_factory(config),
        num_epochs=config.training.num_epochs,
        metric_tracker=tracker,
        metric_restorer=tracker,
        artifact_tracker=tracker,
    )


def run_default_training_pipeline() -> None:
    run_training_pipeline(
        Config(),
        Tracker(
            target_root_dir=Path("out"), restoration_root_dir=Path("/kaggle/input")
        ),
        factories.setup_environment,
        factories.build_train_loader,
        factories.build_val_loader,
        factories.build_transformer,
        factories.build_model,
        factories.build_criterion,
        factories.build_optimizer,
    )
