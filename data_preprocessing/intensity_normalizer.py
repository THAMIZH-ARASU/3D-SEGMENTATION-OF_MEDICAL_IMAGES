"""
intensity_normalizer.py

Implements intensity normalization strategies for CT scan images, including z-score, min-max, and robust normalization. Used in preprocessing pipelines to standardize image intensities for segmentation models.
"""
from typing import Dict, Optional, Tuple
import torch


class IntensityNormalizer:
    """
    Handles different intensity normalization methods for CT scan images.
    """
    
    @staticmethod
    def zscore_normalize(image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply z-score normalization to the image.

        Args:
            image (torch.Tensor): Input image tensor.
            mask (Optional[torch.Tensor]): Optional mask to restrict normalization to foreground.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Normalized image and statistics (mean, std).
        """
        if mask is not None:
            masked_values = image[mask > 0]
            mean_val = masked_values.mean()
            std_val = masked_values.std()
        else:
            mean_val = image.mean()
            std_val = image.std()
            
        normalized = (image - mean_val) / (std_val + 1e-8)
        stats = {"mean": float(mean_val), "std": float(std_val)}
        return normalized, stats
    
    @staticmethod
    def minmax_normalize(image: torch.Tensor, intensity_range: Tuple[float, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply min-max normalization with intensity clipping.

        Args:
            image (torch.Tensor): Input image tensor.
            intensity_range (Tuple[float, float]): (min, max) intensity values for clipping.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Normalized image and statistics (min, max).
        """
        min_val, max_val = intensity_range
        clipped = torch.clamp(image, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        stats = {"min": min_val, "max": max_val}
        return normalized, stats
    
    @staticmethod
    def robust_normalize(image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply robust normalization using percentiles (1st and 99th).

        Args:
            image (torch.Tensor): Input image tensor.
            mask (Optional[torch.Tensor]): Optional mask to restrict normalization to foreground.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Normalized image and statistics (p01, p99).
        """
        if mask is not None:
            masked_values = image[mask > 0]
        else:
            masked_values = image.flatten()
            
        p01, p99 = torch.quantile(masked_values, torch.tensor([0.01, 0.99]))
        clipped = torch.clamp(image, p01, p99)
        normalized = (clipped - p01) / (p99 - p01 + 1e-8)
        stats = {"p01": float(p01), "p99": float(p99)}
        return normalized, stats