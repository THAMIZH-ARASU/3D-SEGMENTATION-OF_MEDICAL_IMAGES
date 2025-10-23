"""
Configuration class for CT scan data preprocessing parameters. Defines spatial, intensity, augmentation, and dataset splitting parameters used in the preprocessing pipeline.

This configuration is used by the DataPreprocessor and CTPreprocessingPipeline to control:
- Spatial transformations (resampling, resizing)
- Intensity normalization and clipping
- Data augmentation for training
- Dataset splitting ratios
- Processing parameters (workers, batch size)
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class PreprocessingConfig:
    """
    Configuration class for preprocessing parameters used in CT scan segmentation.

    This class defines all parameters needed for preprocessing CT scan datasets,
    including spatial transformations, intensity normalization, data augmentation,
    and dataset splitting.

    Attributes:
        target_spacing (Tuple[float, float, float]): Target voxel spacing in mm.
            Default: (1.0, 1.0, 1.0) for isotropic 1mm spacing.
        target_size (Tuple[int, int, int]): Target image size in voxels.
            Default: (256, 256, 256) for 256³ volumes.
        intensity_range (Tuple[float, float]): Intensity clipping range in HU units.
            Default: (-100, 400) for typical CT liver window.
        normalize_method (str): Normalization method to apply.
            Options: "zscore", "minmax", "robust". Default: "zscore".
        apply_augmentation (bool): Whether to apply data augmentation during training.
            Default: True.
        rotation_degrees (float): Maximum rotation angle in degrees for augmentation.
            Default: 10.0 degrees.
        translation_range (float): Maximum translation range in voxels for augmentation.
            Default: 10.0 voxels.
        elastic_deformation (bool): Whether to apply elastic deformation augmentation.
            Default: False.
        train_ratio (float): Proportion of data for training. Default: 0.7 (70%).
        val_ratio (float): Proportion of data for validation. Default: 0.2 (20%).
        test_ratio (float): Proportion of data for testing. Default: 0.1 (10%).
        num_workers (int): Number of workers for data loading. Default: 4.
        batch_size (int): Batch size for processing. Default: 2.
    """
    # Spatial parameters
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    target_size: Tuple[int, int, int] = (256, 256, 256)  # voxels
    
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