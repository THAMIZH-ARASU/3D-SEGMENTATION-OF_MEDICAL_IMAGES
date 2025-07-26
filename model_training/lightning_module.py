"""
lightning_module.py

PyTorch Lightning module for medical image segmentation training. Implements the training, validation, and testing logic for segmentation models with comprehensive metric logging and loss computation.

This module handles:
- Forward pass through the segmentation model
- Loss computation using cross-entropy
- Metric calculation (Dice, IoU, Boundary F1, Accuracy)
- Binary segmentation support for specific target labels
- Comprehensive logging for all metrics during training/validation/testing
"""

import pytorch_lightning as pl
import torch
import torchio as tio
from model_training.losses import SegmentationCrossEntropyLoss
from model_training.optimizers import get_optimizer
from model_training.metrics import dice_coefficient, accuracy, jaccard_index, boundary_f1_score

class SegmentationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for medical image segmentation training.

    This module encapsulates the training logic for segmentation models, including
    forward pass, loss computation, metric calculation, and optimization setup.
    It supports both multi-class and binary segmentation tasks.

    Attributes:
        model: The segmentation model (e.g., DFormer3D, SegFormer).
        config: Training configuration object.
        loss_fn: Cross-entropy loss function for segmentation.
    """

    def __init__(self, model, config):
        """
        Initialize the SegmentationLightningModule.

        Args:
            model: The segmentation model to train.
            config: Training configuration object containing hyperparameters.
        """
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = SegmentationCrossEntropyLoss()
        self.save_hyperparameters(config.__dict__)

    def forward(self, x):
        """
        Forward pass through the segmentation model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W) or (B, C, H, W).

        Returns:
            torch.Tensor: Model logits of shape (B, num_classes, D, H, W) or (B, num_classes, H, W).
        """
        return self.model(x)

    def _prepare_binary(self, logits, y):
        """
        Prepare binary segmentation from multi-class logits and targets.

        This method converts multi-class segmentation to binary by selecting
        only the background and target label channels.

        Args:
            logits (torch.Tensor): Model logits of shape (B, C, ...).
            y (torch.Tensor): Target labels of shape (B, ...).

        Returns:
            tuple: (binary_logits, binary_targets) for binary segmentation.
        """
        # logits: (B, C, ...), y: (B, ...)
        # Convert to binary: 1 for target_label, 0 for background
        # Use only the background and target_label channels
        # If model outputs 14 channels, select [0, target_label]
        if logits.shape[1] > 2:
            idxs = [0, self.config.target_label]
            logits = logits[:, idxs, ...]
        y_bin = (y == self.config.target_label).long()
        return logits, y_bin

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch of data.

        Computes loss and metrics for the training batch and logs them.

        Args:
            batch (dict): Batch dictionary containing 'image' and 'label' keys.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss for the batch.
        """
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA].squeeze(1).long()
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.config.target_label is not None:
            logits, y = self._prepare_binary(logits, y)
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=logits.shape[1])
        iou = jaccard_index(logits, y, num_classes=logits.shape[1])
        bf1 = boundary_f1_score(logits, y, num_classes=logits.shape[1])
        acc = accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_bf1', bf1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch of data.

        Computes loss and metrics for the validation batch and logs them.

        Args:
            batch (dict): Batch dictionary containing 'image' and 'label' keys.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Validation loss for the batch.
        """
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA].squeeze(1).long()
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.config.target_label is not None:
            logits, y = self._prepare_binary(logits, y)
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=logits.shape[1])
        iou = jaccard_index(logits, y, num_classes=logits.shape[1])
        bf1 = boundary_f1_score(logits, y, num_classes=logits.shape[1])
        acc = accuracy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_bf1', bf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for one batch of data.

        Computes loss and metrics for the test batch and logs them.

        Args:
            batch (dict): Batch dictionary containing 'image' and 'label' keys.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Test loss for the batch.
        """
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA].squeeze(1).long()
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.config.target_label is not None:
            logits, y = self._prepare_binary(logits, y)
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=logits.shape[1])
        iou = jaccard_index(logits, y, num_classes=logits.shape[1])
        bf1 = boundary_f1_score(logits, y, num_classes=logits.shape[1])
        acc = accuracy(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_bf1', bf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Sets up the optimizer and scheduler based on the training configuration.

        Returns:
            dict or torch.optim.Optimizer: Optimizer configuration or optimizer object.
        """
        optimizer, scheduler = get_optimizer(self.config, self)
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.config.monitor_metric}
        return optimizer 