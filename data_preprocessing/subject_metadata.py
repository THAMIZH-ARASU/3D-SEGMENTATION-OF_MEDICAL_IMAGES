from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SubjectMetadata:
    """Metadata to retain for post-processing"""
    subject_id: str
    original_spacing: Tuple[float, float, float]
    original_shape: Tuple[int, int, int]
    original_origin: Tuple[float, float, float]
    original_direction: Tuple[Tuple[float, float, float], ...]
    affine_matrix: np.ndarray
    intensity_stats: Dict[str, float]
    dataset_type: str