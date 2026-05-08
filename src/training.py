from typing import Protocol

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

class Module(Protocol):
    def train_step(self, inputs, labels) -> float: ...
    def validation_step(self, inputs, labels) -> float: ...
    
class SegmentationModule(Module):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self.model.train()
        
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self.model.eval()
        
        with torch.inference_mode():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

        return loss.item()
    
class Trainer:
    def __init__(
        self,
        module: Module,
        transform: nn.Module,
        device: torch.device
    ):
        self.training_module = module
        self.transform = transform.to(device)
        self.device = device
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        for epoch in range(num_epochs):
            avg_train_loss = self._train_epoch(train_loader)
            avg_validation_loss = self._validate_epoch(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")

    def _validate_epoch(self, loader: DataLoader) -> float:
        running_validation_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss = self.training_module.validation_step(inputs, labels)
            running_validation_loss += loss
        avg_validation_loss = running_validation_loss / len(loader)
        return avg_validation_loss

    def _train_epoch(self, loader: DataLoader) -> float:
        running_train_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = self.transform(inputs.to(self.device), labels.to(self.device))
            loss = self.training_module.train_step(inputs, labels)
            running_train_loss += loss
        avg_train_loss = running_train_loss / len(loader)
        return avg_train_loss
            

class SyncedImageMaskTransform(torch.nn.Module):
    """Stacks image and mask tensor for performing random augmentation together, then returns the final image and mask"""
    def __init__(self, spatial_transform):
        super().__init__()
        self.spatial_transform = spatial_transform

    def forward(self, image, mask):
        mask = mask.unsqueeze(1).float()
        stacked = torch.cat([image, mask], dim=1)
        
        stacked = self.spatial_transform(stacked)
        
        image_channels = image.shape[1]
        mask_channels = mask.shape[1]
        
        image, mask = torch.split(stacked, [image_channels, mask_channels], dim=1)
        mask = mask.squeeze(1).long()
        
        return image, mask
    
class CombinedLoss(torch.nn.Module):
    '''Focal and dice loss added with the proportion of alpha'''
    def __init__(self, mode: str, alpha: float=0.5):
        super().__init__()
        self.alpha = alpha
        self.focal_loss_fn = smp.losses.FocalLoss(mode=mode)
        self.dice_loss_fn = smp.losses.DiceLoss(mode=mode)

    def forward(self, preds, targets):
        focal_loss = self.focal_loss_fn(preds, targets)
        dice_loss = self.dice_loss_fn(preds, targets)
        
        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss