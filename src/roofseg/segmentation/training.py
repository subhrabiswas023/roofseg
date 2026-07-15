from dataclasses import dataclass
from typing import Self, override

import torch
from torch import nn, optim

from roofseg.common import training
from roofseg.common.typing import Dataclass, StateDict
from roofseg.segmentation.metrics import ConfusionMatrix
from roofseg.segmentation.typing import BatchedImageTensor, BatchedMaskTensor


@dataclass(frozen=True)
class BatchMetrics(Dataclass):
    loss: float
    confusion_matrices: list[ConfusionMatrix]


class Module(training.Module[BatchedImageTensor, BatchedMaskTensor, BatchMetrics]):
    def __init__(
        self,
        model: nn.Module,
        train_transform: nn.Module,
        val_transform: nn.Module,
        criterion: nn.Module,
        loss_regularizer: nn.Module,
        optimizer: optim.Optimizer,
    ):
        self._model = model
        self._train_transform = train_transform
        self._val_transform = val_transform
        self._criterion = criterion
        self._loss_regularizer = loss_regularizer
        self._optimizer = optimizer

    @override
    def to(self, device: torch.device) -> Self:
        self._model = self._model.to(device)
        self._train_transform = self._train_transform.to(device)
        self._val_transform = self._val_transform.to(device)
        self._criterion = self._criterion.to(device)
        self._loss_regularizer = self._loss_regularizer.to(device)
        return self

    @override
    def model_state_dict(self) -> StateDict:
        return self._model.state_dict()

    @override
    def optimizer_state_dict(self) -> StateDict:
        return self._optimizer.state_dict()

    @override
    def train_step(
        self,
        inputs: BatchedImageTensor,
        labels: BatchedMaskTensor,
    ) -> BatchMetrics:
        self._model.train()

        inputs, labels = self._train_transform((inputs, labels))

        self._optimizer.zero_grad()

        logits = self._model(inputs)
        loss = self._criterion(logits, labels)
        
        trainable_parameters = (p for p in self._model.parameters() if p.requires_grad)
        penalty = self._loss_regularizer(trainable_parameters)
        train_loss = loss + penalty

        train_loss.backward()
        self._optimizer.step()

        preds = torch.argmax(logits, dim=1)

        return BatchMetrics(
            loss=loss.item(),
            confusion_matrices=[
                ConfusionMatrix.from_tensors(pred, target)
                for pred, target in zip(preds, labels)
            ],
        )

    @override
    def validation_step(
        self,
        inputs: BatchedImageTensor,
        labels: BatchedMaskTensor,
    ) -> BatchMetrics:
        self._model.eval()

        with torch.inference_mode():
            inputs, labels = self._val_transform((inputs, labels))
            
            logits = self._model(inputs)
            loss = self._criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

        return BatchMetrics(
            loss=loss.item(),
            confusion_matrices=[
                ConfusionMatrix.from_tensors(pred, label)
                for pred, label in zip(preds, labels)
            ],
        )
