from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Any, Self

import torch
from torch.utils.data import DataLoader

from roofseg.common.tracking import Tracker
from roofseg.common.typing import Dataclass


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
    def state_dict(self) -> dict[str, Any]: ...
    def to(self, device: torch.device) -> Self: ...
    def train_step(self, inputs: InputT, labels: LabelT) -> MetricT: ...
    def validation_step(self, inputs: InputT, labels: LabelT) -> MetricT: ...


def train[InputT: torch.Tensor, LabelT: torch.Tensor, MetricT: Dataclass](
    module: Module[InputT, LabelT, MetricT],
    tracker: Tracker,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
):
    module = module.to(device)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            metrics = module.train_step(inputs, labels)

            tracker.log_metrics(
                TrainMetrics(
                    epoch=epoch, batch=batch_idx, phase=Phase.TRAIN, metrics=metrics
                )
            )

        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            metrics = module.validation_step(inputs, labels)

            tracker.log_metrics(
                TrainMetrics(
                    epoch=epoch,
                    batch=batch_idx,
                    phase=Phase.VAL,
                    metrics=metrics,
                )
            )

        torch.save(
            module.state_dict(),
            tracker.artifacts_path / "checkpoint.pth",
        )
