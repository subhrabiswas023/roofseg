from dataclasses import dataclass
from typing import Self

import torch

from roofseg.common.typing import Dataclass

def count_pixels(
    pred: torch.Tensor,
    label: torch.Tensor,
    pred_class: int,
    label_class: int,
) -> int:
    return int(((pred == pred_class) & (label == label_class)).sum().item())


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def from_tensors(cls, pred: torch.Tensor, label: torch.Tensor) -> Self:
        return cls(
            tp=count_pixels(pred, label, 1, 1),
            fp=count_pixels(pred, label, 1, 0),
            tn=count_pixels(pred, label, 0, 0),
            fn=count_pixels(pred, label, 0, 1),
        )
        