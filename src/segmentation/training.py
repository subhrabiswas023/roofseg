from typing import Self, override

from dataclasses import dataclass
import torch
from torch import nn, optim

from ..common import training
from ..common.typing import Dataclass, BatchedRGBTensor, BatchedGrayscaleTensor

from metrics import ConfusionMatrix

@dataclass(frozen=True)
class BatchMetrics(Dataclass):
    loss: float
    confusion_matrices: list[ConfusionMatrix]


class Module(training.Module[BatchedRGBTensor, BatchedGrayscaleTensor, BatchMetrics]):
    def __init__(
        self,
        model: nn.Module,
        transform: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ):
        self._model = model
        self._transform = transform
        self._criterion = criterion
        self._optimizer = optimizer

    @override
    def to(self, device: torch.device) -> Self:
        self._model = self._model.to(device)
        self._transform = self._transform.to(device)
        self._criterion = self._criterion.to(device)
        return self

    @override
    def state_dict(self):
        return self._model.state_dict()

    @override
    def train_step(
        self,
        inputs: BatchedRGBTensor,
        labels: BatchedGrayscaleTensor,
    ) -> BatchMetrics:
        self._model.train()

        inputs, labels = self._transform(inputs, labels)

        self._optimizer.zero_grad()

        logits = self._model(inputs)
        loss = self._criterion(logits, labels)

        loss.backward()
        self._optimizer.step()
        
        preds = torch.argmax(logits, dim=1)

        return BatchMetrics(
            loss=loss.item(),
            confusion_matrices=[
                ConfusionMatrix.from_tensors(pred, target)
                for pred, target in zip(preds, labels)
            ]
        )

    @override
    def validation_step(
        self,
        inputs: BatchedRGBTensor,
        labels: BatchedGrayscaleTensor,
    ) -> BatchMetrics:
        self._model.eval()

        with torch.inference_mode():
            logits = self._model(inputs)
            loss = self._criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1)

        return BatchMetrics(
            loss=loss.item(),
            confusion_matrices=[
                ConfusionMatrix.from_tensors(pred, label)
                for pred, label in zip(preds, labels)
            ]
        )
