from typing import Protocol, Any, Self

import torch
from torch.utils.data import DataLoader

from tracking import Tracker

class Module(Protocol):
    def state_dict(self) -> dict[str, Any]: ...
    def to(self, device: torch.device) -> Self: ...
    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]: ...
    def validation_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]: ...
    
class Trainer:
    def __init__(self, module: Module, tracker: Tracker, device: torch.device):
        self._module = module.to(device)
        self._tracker = tracker
        self._device = device

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        for epoch in range(num_epochs):
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                metrics = self._module.train_step(inputs, labels)
                self._tracker.log_metrics({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "phase": "train",
                    "metrics": metrics,
                })
                
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                metrics = self._module.validation_step(inputs, labels)
                self._tracker.log_metrics({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "phase": "val",
                    "metrics": metrics,
                })
                
            torch.save(self._module.state_dict(), self._tracker.artifacts_path / "checkpoint.pth") 