from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Any, Self

import torch
from torch.utils.data import DataLoader

from .tracking import Tracker
from .typing import Dataclass


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


class Trainer[InputT: torch.Tensor, LabelT: torch.Tensor, MetricT: Dataclass]:
    def __init__(
        self,
        module: Module[InputT, LabelT, MetricT],
        tracker: Tracker,
        device: torch.device,
    ):
        self._module = module.to(device)
        self._tracker = tracker
        self._device = device

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        for epoch in range(num_epochs):
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                metrics = self._module.train_step(inputs, labels)

                self._tracker.log_metrics(
                    TrainMetrics(
                        epoch=epoch, batch=batch_idx, phase=Phase.TRAIN, metrics=metrics
                    )
                )

            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                metrics = self._module.validation_step(inputs, labels)

                self._tracker.log_metrics(
                    TrainMetrics(
                        epoch=epoch,
                        batch=batch_idx,
                        phase=Phase.VAL,
                        metrics=metrics,
                    )
                )

            torch.save(
                self._module.state_dict(),
                self._tracker.artifacts_path / "checkpoint.pth",
            )
