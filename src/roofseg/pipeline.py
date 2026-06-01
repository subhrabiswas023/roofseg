from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

import roofseg.factories as factories
from roofseg.common.training import train
from roofseg.segmentation.config import Config
from roofseg.segmentation.tracking import LocalRestorer, LocalTracker, PathContext
from roofseg.segmentation.training import Module
from roofseg.segmentation.typing import PairedTensor


def run_training_pipeline[C: Config](
    config: C,
    tracker: LocalTracker,
    restorer: LocalRestorer,
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

    if model_state := restorer.restore_model():
        model.load_state_dict(model_state)

    optimizer = optimizer_factory(config, model)

    if optimizer_state := restorer.restore_optimizer():
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
        metric_restorer=restorer,
        artifact_tracker=tracker,
        transaction=tracker.prepare_transaction(),
    )


def run_default_training_pipeline() -> None:
    run_training_pipeline(
        config=Config(),
        tracker=LocalTracker(paths=PathContext(root_dir=Path("out"))),
        restorer=LocalRestorer(paths=PathContext(root_dir=Path("."))),
        environment_setter=factories.setup_environment,
        train_loader_factory=factories.build_train_loader,
        val_loader_factory=factories.build_val_loader,
        transformer_factory=factories.build_transformer,
        model_factory=factories.build_model,
        criterion_factory=factories.build_criterion,
        optimizer_factory=factories.build_optimizer,
    )
