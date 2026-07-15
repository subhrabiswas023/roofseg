from typing import Callable

import torch
from torch.utils.data import DataLoader

from roofseg import factories
from roofseg.common.training import train
from roofseg.segmentation.config import Config
from roofseg.segmentation.tracking import LocalRestorer, LocalTracker
from roofseg.segmentation.training import Module
from roofseg.segmentation.typing import PairedTensor


def run_training_pipeline[C: Config](
    config: C,
    tracker_factory: Callable[[], LocalTracker] = factories.build_tracker,
    restorer_factory: Callable[[], LocalRestorer] = factories.build_restorer,
    environment_setter: Callable[[C], torch.device] = factories.setup_environment,
    train_loader_factory: Callable[[C], DataLoader[PairedTensor]] = factories.build_train_loader,
    val_loader_factory: Callable[[C], DataLoader[PairedTensor]] = factories.build_val_loader,
    train_transformer_factory: Callable[[C], torch.nn.Module] = factories.build_train_transformer,
    val_transformer_factory: Callable[[C], torch.nn.Module] = factories.build_val_transformer,
    model_factory: Callable[[C], torch.nn.Module] = factories.build_model,
    criterion_factory: Callable[[C], torch.nn.Module] = factories.build_criterion,
    loss_regurlarizer_factory: Callable[[C], torch.nn.Module] = factories.build_loss_regularizer,
    optimizer_factory: Callable[[C, torch.nn.Module], torch.optim.Optimizer] = factories.build_optimizer,
) -> None:
    tracker = tracker_factory()
    restorer = restorer_factory()
    
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
            model=model,
            train_transform=train_transformer_factory(config),
            val_transform=val_transformer_factory(config),
            criterion=criterion_factory(config),
            loss_regularizer=loss_regurlarizer_factory(config),
            optimizer=optimizer,
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
        config=Config()
    )
