from dataclasses import dataclass
from typing import Tuple


@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing parameters"""
    # Spatial parameters
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    target_size: Tuple[int, int, int] = (256, 256, 300)  # voxels
    
    # Intensity parameters
    intensity_range: Tuple[float, float] = (-100, 400)  # HU units for CT
    normalize_method: str = "zscore"  # "zscore", "minmax", "robust"
    
    # Augmentation parameters (for training)
    apply_augmentation: bool = True
    rotation_degrees: float = 10.0
    translation_range: float = 10.0
    elastic_deformation: bool = False
    
    # Dataset split
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Processing parameters
    num_workers: int = 4
    batch_size: int = 2