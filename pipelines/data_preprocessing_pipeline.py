from typing import Dict, Tuple
import torch
import torchio as tio

from cofigs.data_preprocessing_config import PreprocessingConfig
from data_preprocessing.intensity_normalizer import IntensityNormalizer
from data_preprocessing.subject_metadata import SubjectMetadata

class CTPreprocessingPipeline:
    """Main preprocessing pipeline for CT scans"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.normalizer = IntensityNormalizer()
        
    def create_transforms(self, is_training: bool = False) -> tio.Compose:
        """Create TorchIO transforms pipeline"""
        transforms = [
            # Spatial preprocessing
            tio.Resample(self.config.target_spacing),
            tio.Resize(self.config.target_size),
            
            # Intensity preprocessing
            tio.Clamp(out_min=self.config.intensity_range[0], 
                     out_max=self.config.intensity_range[1]),
        ]
        
        # Add augmentations for training
        if is_training and self.config.apply_augmentation:
            transforms.extend([
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=self.config.rotation_degrees,
                    translation=self.config.translation_range,
                    p=0.5
                ),
                tio.RandomElasticDeformation(p=0.3) if self.config.elastic_deformation else tio.Lambda(lambda x: x),
                tio.RandomFlip(axes=(0,), p=0.5),
                tio.RandomNoise(p=0.3),
                tio.RandomGamma(p=0.3),
            ])
            
        return tio.Compose(transforms)
    
    def extract_metadata(self, subject: tio.Subject) -> SubjectMetadata:
        """Extract metadata for post-processing"""
        import numpy as np
        image = subject['image']
        
        # Get original properties
        spacing = image.spacing
        shape = image.shape[1:]  # Remove channel dimension
        origin = image.origin
        # Robustly handle direction
        direction = image.direction
        if hasattr(direction, "numpy"):
            direction = direction.numpy()
        if isinstance(direction, np.ndarray):
            if direction.ndim == 0:  # scalar
                direction = ((float(direction),),)
            elif direction.ndim == 1:  # vector
                direction = tuple((float(x),) for x in direction)
            else:  # matrix
                direction = tuple(tuple(float(x) for x in row) for row in direction)
        elif isinstance(direction, (float, int, np.floating, np.integer)):
            direction = ((float(direction),),)
        elif isinstance(direction, (list, tuple)):
            if all(isinstance(x, (float, int, np.floating, np.integer)) for x in direction):
                direction = tuple((float(x),) for x in direction)
            else:
                direction = tuple(tuple(float(xx) for xx in x) for x in direction)
        else:
            direction = ((0.0,),)  # fallback
        # Robustly handle affine
        if hasattr(image.affine, "numpy"):
            affine = image.affine.numpy()
        else:
            affine = np.array(image.affine)
        
        # Calculate intensity statistics
        image_data = image.data.squeeze()
        intensity_stats = {
            "min": float(image_data.min()),
            "max": float(image_data.max()),
            "mean": float(image_data.mean()),
            "std": float(image_data.std()),
            "median": float(image_data.median())
        }
        
        return SubjectMetadata(
            subject_id=subject['subject_id'],
            original_spacing=tuple(spacing),
            original_shape=tuple(shape),
            original_origin=tuple(origin),
            original_direction=direction,
            affine_matrix=affine,
            intensity_stats=intensity_stats,
            dataset_type=subject.get('dataset_type', 'unknown')
        )
    
    def create_combined_mask(self, liver_mask: torch.Tensor, tumor_mask: torch.Tensor) -> torch.Tensor:
        """Create combined mask from separate liver and tumor masks"""
        combined = torch.zeros_like(liver_mask)
        combined[liver_mask > 0] = 1  # Liver label
        combined[tumor_mask > 0] = 2  # Tumor label (overwrites liver in tumor regions)
        return combined
    
    def process_subject(self, subject_info: Dict[str, str], transform: tio.Compose) -> Tuple[tio.Subject, SubjectMetadata]:
        """Process a single subject"""
        # Load image
        image = tio.ScalarImage(subject_info['image'])
        
        # Handle different label formats
        if subject_info['combined_label']:
            # Medical Decathlon format - combined mask
            label = tio.LabelMap(subject_info['combined_label'])
        else:
            # JIPMER format - separate masks
            liver_mask = tio.LabelMap(subject_info['liver_label'])
            tumor_mask = tio.LabelMap(subject_info['tumor_label'])
            
            # Combine masks
            combined_mask_data = self.create_combined_mask(
                liver_mask.data.squeeze(), 
                tumor_mask.data.squeeze()
            ).unsqueeze(0)  # Add channel dimension
            
            # Create combined label map
            label = tio.LabelMap(tensor=combined_mask_data, affine=liver_mask.affine)
        
        # Create subject
        subject = tio.Subject(
            image=image,
            label=label,
            subject_id=subject_info['subject_id'],
            dataset_type=subject_info.get('dataset_type', 'unknown')
        )
        
        # Extract metadata before transformation
        metadata = self.extract_metadata(subject)
        
        # Apply transforms
        transformed_subject = transform(subject)
        
        # Apply intensity normalization
        image_data = transformed_subject['image'].data.squeeze()
        label_data = transformed_subject['label'].data.squeeze()
        
        if self.config.normalize_method == "zscore":
            normalized_image, norm_stats = self.normalizer.zscore_normalize(
                image_data, label_data > 0)
        elif self.config.normalize_method == "minmax":
            normalized_image, norm_stats = self.normalizer.minmax_normalize(
                image_data, self.config.intensity_range)
        elif self.config.normalize_method == "robust":
            normalized_image, norm_stats = self.normalizer.robust_normalize(
                image_data, label_data > 0)
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalize_method}")
        
        # Update metadata with normalization stats
        metadata.intensity_stats.update(norm_stats)
        
        # Update subject with normalized image
        transformed_subject['image'] = tio.ScalarImage(
            tensor=normalized_image.unsqueeze(0),
            affine=transformed_subject['image'].affine
        )
        
        return transformed_subject, metadata