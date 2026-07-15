from collections.abc import Iterable
from typing import override

import segmentation_models_pytorch as smp
import torch


class CombinedLoss(torch.nn.Module):
    """Focal and dice loss added with the proportion of alpha"""

    def __init__(self, mode: str, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.focal_loss_fn = smp.losses.FocalLoss(mode=mode)
        self.dice_loss_fn = smp.losses.DiceLoss(mode=mode)

    @override
    def forward(self, preds, targets):
        focal_loss = self.focal_loss_fn(preds, targets)
        dice_loss = self.dice_loss_fn(preds, targets)

        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss
    
class L1Regularizer(torch.nn.Module):
    """Generates the L1 loss penalty to be added to the base loss

    Args:
        l1_lambda (float): The constant to multiply with the L1 penalty
    """
    def __init__(self, l1_lambda: float):
        super().__init__()
        self.l1_lambda = l1_lambda
        
    @override
    def forward(self, parameters: Iterable[torch.nn.Parameter]):
        l1_penalty = sum(torch.sum(torch.abs(p)) for p in parameters)
        return 0.5 * self.l1_lambda * l1_penalty
