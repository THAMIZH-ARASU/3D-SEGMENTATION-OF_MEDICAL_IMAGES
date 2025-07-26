"""
data.py

Data loading utilities for model prediction/inference. Provides functions to create data loaders for prediction tasks using preprocessed medical images.

This module handles the creation of prediction data loaders that work with
TorchIO for loading and preprocessing medical images during inference.
"""

import os
import torchio as tio
from typing import List, Dict, Any

def get_prediction_loader(input_dir: str, batch_size: int = 1, num_workers: int = 2):
    """
    Create a data loader for prediction tasks from preprocessed images.

    This function scans the input directory for preprocessed image files and
    creates a TorchIO data loader suitable for model prediction. It automatically
    detects image files with the '_image.nii.gz' suffix and creates corresponding
    subject metadata.

    Args:
        input_dir (str): Directory containing preprocessed image files.
            Expected format: {subject_id}_image.nii.gz
        batch_size (int): Batch size for the data loader.
            Default: 1 (recommended for prediction to avoid memory issues).
        num_workers (int): Number of worker processes for data loading.
            Default: 2.

    Returns:
        tuple: (loader, subjects) where:
            - loader: TorchIO SubjectsLoader for batch processing
            - subjects: List of subject metadata dictionaries containing:
                - 'subject_id': Unique identifier for each subject
                - 'image': Path to the image file

    Example:
        >>> loader, subjects = get_prediction_loader("data_preprocessed/test")
        >>> for batch, subject in zip(loader, subjects):
        ...     # Process batch for prediction
        ...     pass

    Note:
        This function expects preprocessed images in NIfTI format with the
        naming convention: {subject_id}_image.nii.gz. Only image files are
        loaded (no labels) since this is for prediction only.
    """
    subjects = []
    for fname in os.listdir(input_dir):
        if fname.endswith('_image.nii.gz'):
            subject_id = fname.replace('_image.nii.gz', '')
            image_path = os.path.join(input_dir, f"{subject_id}_image.nii.gz")
            subjects.append({
                'subject_id': subject_id,
                'image': image_path
            })
    tio_subjects = [
        tio.Subject(
            image=tio.ScalarImage(s['image']),
            subject_id=s['subject_id']
        ) for s in subjects
    ]
    dataset = tio.SubjectsDataset(tio_subjects)
    loader = tio.SubjectsLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loader, subjects 