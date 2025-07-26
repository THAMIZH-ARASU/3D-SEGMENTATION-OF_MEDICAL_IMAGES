"""
predictor.py

Model prediction/inference module for medical image segmentation. Provides a high-level interface for running inference with trained segmentation models and saving predictions in NIfTI format.

This module handles model loading, batch prediction, post-processing, and
saving of segmentation results. It supports both multi-class and binary
segmentation tasks with automatic label mapping.
"""

import os
import torch
import torchio as tio
import pytorch_lightning as pl
import numpy as np
from typing import List
import nibabel as nib

class Predictor:
    """
    High-level predictor for medical image segmentation models.

    This class provides a complete prediction pipeline for segmentation models,
    including model loading, batch processing, post-processing, and result
    saving. It supports both multi-class and binary segmentation tasks.

    Attributes:
        config: Prediction configuration object.
        model: The segmentation model instance.
        lightning_module: PyTorch Lightning module loaded from checkpoint.
    """

    def __init__(self, lightning_module_class, config, model_class):
        """
        Initialize the Predictor with model and configuration.

        Args:
            lightning_module_class: PyTorch Lightning module class to load.
            config: Prediction configuration object containing model parameters.
            model_class: Model class constructor function.
        """
        self.config = config
        self.model = model_class(num_classes=config.num_classes, in_chan=config.input_channels, **config.model_params)
        self.lightning_module = lightning_module_class.load_from_checkpoint(
            config.checkpoint_path, model=self.model, config=config, map_location=config.device)
        self.lightning_module.eval()
        self.lightning_module.to(config.device)

    def predict(self, dataloader: torch.utils.data.DataLoader, subject_list: List[dict]):
        """
        Run prediction on all subjects in the dataloader.

        This method processes each batch from the dataloader, runs inference
        through the model, applies post-processing (including binary mapping
        if configured), and saves predictions as NIfTI files.

        Args:
            dataloader: DataLoader providing batches of images for prediction.
            subject_list: List of subject metadata dictionaries corresponding
                to the dataloader batches.

        Note:
            Predictions are saved in NIfTI format with the naming convention:
            {subject_id}_pred.nii.gz in the configured output directory.
            
            For binary segmentation (when target_label is set), predictions
            are mapped from binary output (0, 1) to the target label values.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        with torch.no_grad():
            for i, (batch, subject) in enumerate(zip(dataloader, subject_list)):
                x = batch['image'][tio.DATA].to(self.config.device)
                logits = self.lightning_module(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # If binary (tumor only), select only background and target_label channels
                if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                    logits = logits[:, [0, self.config.target_label], ...]
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                # If binary, map 1 -> target_label, 0 -> 0
                if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                    pred = np.where(pred == 1, self.config.target_label, 0)
                # Save prediction as NIfTI
                affine = batch['image'][tio.AFFINE][0]
                subject_id = subject['subject_id']
                out_path = os.path.join(self.config.output_dir, f"{subject_id}_pred.nii.gz")
                nib.save(nib.Nifti1Image(pred[0].astype(np.uint8), affine), out_path) 