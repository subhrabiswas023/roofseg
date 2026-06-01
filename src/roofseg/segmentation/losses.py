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
