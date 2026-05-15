import torch

from roofseg.segmentation.transforms import SyncedImageMaskTransform
import kornia.augmentation as K


class TestSyncedImageMaskTransform:
    def test_horizontal_flip(self, images, masks):
        transform = SyncedImageMaskTransform(K.RandomHorizontalFlip(p=1.0))

        transformed_image, transformed_mask = transform(images, masks)

        expected_image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        expected_image[:, :, 0, 1] = 1.0
        expected_mask = torch.zeros((1, 2, 2), dtype=torch.int64)
        expected_mask[:, 0, 1] = 1

        assert torch.equal(transformed_image, expected_image)
        assert torch.equal(transformed_mask, expected_mask)
