from dataclasses import dataclass

from roofseg.common.typing import Dataclass


# Environment parameters
@dataclass(frozen=True)
class EnvironmentConfig:
    seed: int = 42
    device: str = "cuda"


# Dataset parameters
@dataclass(frozen=True)
class DatasetConfig:
    root_dir: str = "/kaggle/input/inria-rooftop-segmentation-dataset-1024x1024-png"  # FIX ME: hardcoded path for now. Only depends on kaggle environment
    image_dir: str = "images"
    mask_dir: str = "masks"

    image_height: int = 1024
    image_width: int = 1024

    num_classes: int = 2
    color_threshold: int = 128
    patch_size: int = 256


# Augmentation parameters
@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5


# Model parameters
@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str = "mobilenet_v2"
    encoder_weights: str = "imagenet"


# Loss function parameters
@dataclass(frozen=True)
class CriterionConfig:
    criterion: str = "CombinedLoss"
    loss_alpha: float = 0.5


# Optimizer parameters
@dataclass(frozen=True)
class OptimizerConfig:
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3


# Training parameters
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 1


@dataclass(frozen=True)
class Config(Dataclass):
    environment: EnvironmentConfig = EnvironmentConfig()
    dataset: DatasetConfig = DatasetConfig()
    augmentation: AugmentationConfig = AugmentationConfig()

    model: ModelConfig = ModelConfig()

    criterion: CriterionConfig = CriterionConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
