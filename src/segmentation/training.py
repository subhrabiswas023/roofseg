from typing import Self, Any

import torch
from torch import nn, optim

from ..common import training

class Module(training.Module):
    def __init__(
        self,
        model: nn.Module,
        tranform: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ):
        self._model = model
        self._transform = tranform
        self._criterion = criterion
        self._optimizer = optimizer

    def to(self, device: torch.device) -> Self:
        self._model = self._model.to(device)
        self._transform = self._transform.to(device)
        self._criterion = self._criterion.to(device)
        return self
    
    def state_dict(self):
        return self._model.state_dict()

    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, Any]:
        self._model.train()

        inputs, labels = self._transform(inputs, labels)

        self._optimizer.zero_grad()

        outputs = self._model(inputs)
        loss = self._criterion(outputs, labels)

        loss.backward()
        self._optimizer.step()

        return {
            "train_loss": loss.item()
        }

    def validation_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str,Any]:
        self._model.eval()

        with torch.inference_mode():
            outputs = self._model(inputs)
            loss = self._criterion(outputs, labels)

        return {
            "val_loss": loss.item()
        }
