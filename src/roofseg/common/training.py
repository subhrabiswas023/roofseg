from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Self

import torch
from torch.utils.data import DataLoader

from roofseg.common.transaction import Transaction
from roofseg.common.typing import Dataclass, JsonDict, StateDict
from roofseg.common.tracking import ArtifactTracker, MetricRestorer, MetricTracker


class Phase(StrEnum):
    TRAIN = "train"
    VAL = "val"


@dataclass(frozen=True)
class TrainMetrics(Dataclass):
    epoch: int
    batch: int
    phase: Phase
    metrics: Dataclass


class Module[InputT: torch.Tensor, LabelT: torch.Tensor, MetricT: Dataclass](Protocol):
    def model_state_dict(self) -> StateDict: ...
    def optimizer_state_dict(self) -> StateDict: ...
    def to(self, device: torch.device) -> Self: ...
    def train_step(self, inputs: InputT, labels: LabelT) -> MetricT: ...
    def validation_step(self, inputs: InputT, labels: LabelT) -> MetricT: ...


def train[InputT: torch.Tensor, LabelT: torch.Tensor, MetricT: Dataclass](
    module: Module[InputT, LabelT, MetricT],
    device: torch.device,
    train_loader: DataLoader[tuple[InputT, LabelT]],
    val_loader: DataLoader[tuple[InputT, LabelT]],
    num_epochs: int,
    metric_tracker: MetricTracker[JsonDict],
    metric_restorer: MetricRestorer[JsonDict],
    artifact_tracker: ArtifactTracker[StateDict],
    transaction: Transaction,
):
    module = module.to(device)

    last_metrics = metric_restorer.restore_last_metrics()
    start_epoch = TrainMetrics.from_dict(last_metrics).epoch + 1 if last_metrics else 0

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            last_metrics = module.train_step(inputs.to(device), labels.to(device))

            metric_tracker.log_metrics(
                TrainMetrics(
                    epoch=epoch,
                    batch=batch_idx,
                    phase=Phase.TRAIN,
                    metrics=last_metrics,
                ).to_dict()
            )

        for batch_idx, (inputs, labels) in enumerate(val_loader):
            last_metrics = module.validation_step(inputs.to(device), labels.to(device))

            metric_tracker.log_metrics(
                TrainMetrics(
                    epoch=epoch,
                    batch=batch_idx,
                    phase=Phase.VAL,
                    metrics=last_metrics,
                ).to_dict()
            )

        artifact_tracker.save_model(module.model_state_dict())
        artifact_tracker.save_optimizer(module.optimizer_state_dict())
        
        transaction.stage()
        transaction.commit()
        
