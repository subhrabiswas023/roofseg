from dataclasses import dataclass

from roofseg.common.typing import Dataclass


# Environment parameters
@dataclass(frozen=True)
class EnvironmentConfig:
    seed: int
    device: str


# Dataset parameters
@dataclass(frozen=True)
class DatasetConfig:
    root_dir: str
    image_dir: str
    mask_dir: str

    image_height: int
    image_width: int

    num_classes: int
    color_threshold: int
    patch_size: int


@dataclass(frozen=True)
class AugmentationConfig:
    # Augmentation parameters
    horizontal_flip_prob: float
    vertical_flip_prob: float


# Model parameters
@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str
    encoder_weights: str


# Loss function parameters
@dataclass(frozen=True)
class CriterionConfig:
    criterion: str
    loss_alpha: float


# Optimizer parameters
@dataclass(frozen=True)
class OptimizerConfig:
    optimizer: str
    learning_rate: float


# Training parameters
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    num_epochs: int


@dataclass(frozen=True)
class Config(Dataclass):
    environment: EnvironmentConfig
    dataset: DatasetConfig
    augmentation: AugmentationConfig

    model: ModelConfig

    criterion: CriterionConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
