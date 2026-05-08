from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Environment parameters
    seed: int
    device: str
    
    # Dataset parameters
    root_dir: str
    image_dir: str
    mask_dir: str
    image_height: int
    image_width: int
    num_classes: int
    color_threshold: int
    patch_size: int
     
    # Augmentation parameters
    horizontal_flip_prob: float
    vertical_flip_prob: float
    
    # Model parameters
    encoder_name: str
    encoder_weights: str
    
    # Training parameters
    batch_size: int
    num_epochs: int
    
    optimizer: str
    learning_rate: float
    
    criterion: str
    loss_alpha: float