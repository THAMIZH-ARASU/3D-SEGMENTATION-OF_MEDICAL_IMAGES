"""
Loss functions for medical image segmentation tasks. Implements cross-entropy loss with optional class weighting for handling class imbalance in segmentation datasets.

This module provides loss functions specifically designed for segmentation tasks,
supporting both standard cross-entropy and weighted variants for imbalanced datasets.
"""

import torch
import torch.nn as nn

class SegmentationCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss function for medical image segmentation.

    This loss function is specifically designed for segmentation tasks and wraps
    PyTorch's CrossEntropyLoss with optional class weighting to handle class
    imbalance in medical imaging datasets.

    Attributes:
        ce (nn.CrossEntropyLoss): The underlying cross-entropy loss function.
    """

    def __init__(self, weight=None):
        """
        Initialize the SegmentationCrossEntropyLoss.

        Args:
            weight (torch.Tensor, optional): Class weights to handle class imbalance.
                Should be a tensor of size (num_classes,). Default: None.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        """
        Compute the cross-entropy loss between predictions and targets.

        Args:
            logits (torch.Tensor): Model predictions of shape (B, C, D, H, W) or (B, C, H, W).
                Raw logits before softmax.
            targets (torch.Tensor): Ground truth labels of shape (B, D, H, W) or (B, H, W).
                Should contain class indices (not one-hot encoded).

        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        return self.ce(logits, targets) 