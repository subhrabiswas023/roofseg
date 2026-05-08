import pytest
import torch

from training import SyncedImageMaskTransform, train_step
import kornia.augmentation as K


@pytest.mark.slow
def test_training_step(model, optimizer, criterion, random_images, random_masks):
    loss = train_step(model, optimizer, criterion, random_images, random_masks)

    assert isinstance(loss, float)
    assert loss > 0


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
