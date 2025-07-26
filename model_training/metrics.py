"""
metrics.py

Evaluation metrics for medical image segmentation tasks. Implements Dice coefficient, Jaccard index (IoU), accuracy, and boundary F1 score for comprehensive segmentation evaluation.

This module provides tensor-based implementations of segmentation metrics that work
with PyTorch tensors and support both 3D and 2D segmentation tasks. All metrics
expect tensors as produced by batch['image'][tio.DATA] and batch['label'][tio.DATA],
i.e., (B, C, ...) for images and (B, ...) for labels.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion

# All metric functions expect tensors as produced by batch['image'][tio.DATA] and batch['label'][tio.DATA]
# i.e., (B, C, ...) for images and (B, ...) for labels

def dice_coefficient(preds, targets, num_classes=3, smooth=1e-5):
    """
    Compute the Dice coefficient (F1 score) for segmentation evaluation.

    The Dice coefficient measures the overlap between predicted and ground truth
    segmentations. It is defined as: 2 * |A ∩ B| / (|A| + |B|), where A and B
    are the predicted and ground truth masks respectively.

    Args:
        preds (torch.Tensor): Model predictions of shape (B, C, ...).
            Raw logits before softmax.
        targets (torch.Tensor): Ground truth labels of shape (B, ...).
            Should contain class indices.
        num_classes (int): Number of classes in the segmentation task.
            Default: 3.
        smooth (float): Smoothing factor to prevent division by zero.
            Default: 1e-5.

    Returns:
        float: Mean Dice coefficient across all classes.

    Note:
        This function computes Dice coefficient for all classes and returns
        the mean value. For binary segmentation, use num_classes=2.
    """
    # preds: (B, C, ...), targets: (B, ...)
    preds = torch.argmax(preds, dim=1)
    preds_one_hot = F.one_hot(preds, num_classes).permute(0, 4, 1, 2, 3)
    targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3)
    dims = (0, 2, 3, 4)
    intersection = torch.sum(preds_one_hot * targets_one_hot, dim=dims)
    union = torch.sum(preds_one_hot, dim=dims) + torch.sum(targets_one_hot, dim=dims)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def jaccard_index(preds, targets, num_classes=3, smooth=1e-5):
    """
    Compute the Jaccard index (Intersection over Union, IoU) for segmentation evaluation.

    The Jaccard index measures the overlap between predicted and ground truth
    segmentations. It is defined as: |A ∩ B| / |A ∪ B|, where A and B are the
    predicted and ground truth masks respectively.

    Args:
        preds (torch.Tensor): Model predictions of shape (B, C, ...).
            Raw logits before softmax.
        targets (torch.Tensor): Ground truth labels of shape (B, ...).
            Should contain class indices.
        num_classes (int): Number of classes in the segmentation task.
            Default: 3.
        smooth (float): Smoothing factor to prevent division by zero.
            Default: 1e-5.

    Returns:
        float: Mean Jaccard index across all classes.

    Note:
        This function computes IoU for all classes and returns the mean value.
        For binary segmentation, use num_classes=2.
    """
    # preds: (B, C, ...), targets: (B, ...)
    preds = torch.argmax(preds, dim=1)
    preds_one_hot = F.one_hot(preds, num_classes).permute(0, 4, 1, 2, 3)
    targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3)
    dims = (0, 2, 3, 4)
    intersection = torch.sum(preds_one_hot * targets_one_hot, dim=dims)
    union = torch.sum(preds_one_hot, dim=dims) + torch.sum(targets_one_hot, dim=dims) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def accuracy(preds, targets):
    """
    Compute pixel-wise accuracy for segmentation evaluation.

    Accuracy measures the proportion of correctly classified pixels across
    all classes in the segmentation task.

    Args:
        preds (torch.Tensor): Model predictions of shape (B, C, ...).
            Raw logits before softmax.
        targets (torch.Tensor): Ground truth labels of shape (B, ...).
            Should contain class indices.

    Returns:
        float: Pixel-wise accuracy (proportion of correct predictions).

    Note:
        This metric gives equal weight to all pixels regardless of class
        frequency, which may not be ideal for imbalanced datasets.
    """
    # preds: (B, C, ...), targets: (B, ...)
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float()
    return correct.mean().item()

def boundary_f1_score(preds, targets, num_classes=3, tolerance=1):
    """
    Compute boundary F1 score for segmentation evaluation.

    The boundary F1 score measures the accuracy of segmentation boundaries
    by computing the F1 score between predicted and ground truth boundary
    pixels within a specified tolerance distance.

    Args:
        preds (torch.Tensor): Model predictions of shape (B, C, ...).
            Raw logits before softmax.
        targets (torch.Tensor): Ground truth labels of shape (B, ...).
            Should contain class indices.
        num_classes (int): Number of classes in the segmentation task.
            Default: 3.
        tolerance (int): Tolerance distance in pixels for boundary matching.
            Default: 1.

    Returns:
        float: Mean boundary F1 score across all foreground classes.

    Note:
        This metric is particularly useful for evaluating the quality of
        segmentation boundaries, which is important in medical imaging.
        Background class (class 0) is excluded from computation.
    """
    # preds: (B, C, ...), targets: (B, ...)
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    bf1s = []
    for c in range(1, num_classes):  # skip background
        bf1s_c = []
        for p, t in zip(preds, targets):
            p_bin = (p == c).astype(np.uint8)
            t_bin = (t == c).astype(np.uint8)
            if np.sum(p_bin) == 0 and np.sum(t_bin) == 0:
                bf1s_c.append(1.0)
                continue
            p_b = p_bin - binary_erosion(p_bin)
            t_b = t_bin - binary_erosion(t_bin)
            # Distance transform
            p_d = np.minimum(1, binary_erosion(p_b, iterations=tolerance))
            t_d = np.minimum(1, binary_erosion(t_b, iterations=tolerance))
            # Precision: fraction of predicted boundary pixels within tolerance of GT boundary
            precision = (p_b * t_d).sum() / (p_b.sum() + 1e-8)
            # Recall: fraction of GT boundary pixels within tolerance of predicted boundary
            recall = (t_b * p_d).sum() / (t_b.sum() + 1e-8)
            if precision + recall == 0:
                bf1 = 0.0
            else:
                bf1 = 2 * precision * recall / (precision + recall)
            bf1s_c.append(bf1)
        bf1s.append(np.mean(bf1s_c))
    return float(np.mean(bf1s)) 