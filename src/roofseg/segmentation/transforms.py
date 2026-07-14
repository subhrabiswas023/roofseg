from typing import override

import torch

from roofseg.segmentation.typing import BatchedImageTensor, BatchedMaskTensor

class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
    @override 
    def forward(self, input: BatchedImageTensor):
        return input / 255.0

class SyncedImageMaskTransform(torch.nn.Module):
    """Stacks image and mask tensor for performing random augmentation together, then returns the final image and mask"""

    def __init__(self, spatial_transform: torch.nn.Module):
        super().__init__()
        self.spatial_transform = spatial_transform

    @override
    def forward(self, image: BatchedImageTensor, mask: BatchedMaskTensor):
        mask = mask.unsqueeze(1).float()
        stacked = torch.cat([image, mask], dim=1)

        stacked = self.spatial_transform(stacked)

        image_channels = image.shape[1]
        mask_channels = mask.shape[1]

        image, mask = torch.split(stacked, [image_channels, mask_channels], dim=1)
        mask = mask.squeeze(1).long()

        return image, mask