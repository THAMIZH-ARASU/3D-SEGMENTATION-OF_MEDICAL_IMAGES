"""
PyTorch Lightning trainer wrapper for medical image segmentation training. Provides a high-level interface for training, validation, and testing of segmentation models with comprehensive logging and checkpointing.

This module encapsulates the PyTorch Lightning Trainer with custom callbacks,
logging configuration, and training/testing workflows specifically designed
for medical image segmentation tasks.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class SegmentationTrainer:
    """
    PyTorch Lightning trainer wrapper for medical image segmentation.

    This class provides a high-level interface for training segmentation models
    using PyTorch Lightning. It handles trainer configuration, callbacks setup,
    logging, and training/testing workflows.

    Attributes:
        config: Training configuration object.
        model: PyTorch Lightning module to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader (optional).
        logger: TensorBoard logger for training metrics.
        checkpoint_callback: Model checkpointing callback.
        early_stopping: Early stopping callback.
        trainer: PyTorch Lightning Trainer instance.
    """

    def __init__(self, config, model, train_loader, val_loader, test_loader=None):
        """
        Initialize the SegmentationTrainer.

        Args:
            config: Training configuration object containing trainer parameters.
            model: PyTorch Lightning module to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader (optional).
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=config.project_name,
            version=config.run_name
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            save_top_k=1,
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            filename="{epoch}-{val_loss:.4f}",
            save_last=True
        )

        self.early_stopping = EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            mode=config.monitor_mode,
            verbose=True
        )

        self.trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator="gpu" if config.gpus > 0 else "cpu",
            devices=config.gpus if config.gpus > 0 else 1,
            precision=config.precision,
            logger=self.logger,
            callbacks=[self.checkpoint_callback, self.early_stopping],
            log_every_n_steps=config.log_every_n_steps,
            deterministic=False,
            enable_progress_bar=True
        )

    def fit(self):
        """
        Train the model using the configured trainer.

        This method starts the training process using PyTorch Lightning.
        It handles both fresh training and resuming from checkpoints
        based on the configuration.

        Note:
            Training will automatically use early stopping and model
            checkpointing as configured during initialization.
        """
        if self.config.resume_from_checkpoint:
            self.trainer.fit(self.model, self.train_loader, self.val_loader, ckpt_path=self.config.resume_from_checkpoint)
        else:
            self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def test(self):
        """
        Test the trained model on the test dataset.

        This method evaluates the trained model on the test dataset
        if a test loader is provided. It can resume from a checkpoint
        if specified in the configuration.

        Note:
            This method requires a test_loader to be provided during
            initialization. If no test_loader is available, this
            method will do nothing.
        """
        if self.test_loader is not None:
            if self.config.resume_from_checkpoint:
                self.trainer.test(self.model, self.test_loader, ckpt_path=self.config.resume_from_checkpoint)
            else:
                self.trainer.test(self.model, self.test_loader) 