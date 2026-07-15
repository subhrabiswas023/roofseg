from typing import override

import torch
from torch import nn

from roofseg.segmentation.typing import BatchedImageTensor, PairedBatchedTensor


class ScaleImage(nn.Module):
    def __init__(self, factor: float = 255):
        super().__init__()
        self.factor = factor

    @override
    def forward(self, input: BatchedImageTensor):
        return input / self.factor


class SyncedImageMaskTransform(nn.Module):
    """Stacks image and mask tensor for performing random augmentation together, then returns the final image and mask"""

    def __init__(self, spatial_transform: nn.Module):
        super().__init__()
        self.spatial_transform = spatial_transform

    @override
    def forward(self, paired_instance: PairedBatchedTensor):
        image, mask = paired_instance

        mask = mask.unsqueeze(1).float()
        stacked = torch.cat([image, mask], dim=1)

        stacked = self.spatial_transform(stacked)

        image_channels = image.shape[1]
        mask_channels = mask.shape[1]

        image, mask = torch.split(stacked, [image_channels, mask_channels], dim=1)
        mask = mask.squeeze(1).long()

        return image, mask


class ImageTransform(nn.Module):
    def __init__(
        self,
        scale_image: nn.Module,
        synced_image_mask_transform: nn.Module,
        normalize: nn.Module,
    ):
        super().__init__()
        self.scale_image = scale_image
        self.inner_transformer = synced_image_mask_transform
        self.normalize = normalize

    def forward(self, paired_instance: PairedBatchedTensor):
        image, mask = paired_instance
        image = self.scale_image(image)
        image, mask = self.inner_transformer((image, mask))
        image = self.normalize(image)

        return image, mask
