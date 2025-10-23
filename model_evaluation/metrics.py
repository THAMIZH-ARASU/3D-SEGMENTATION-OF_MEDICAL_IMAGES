"""
Evaluation metrics for medical image segmentation using NumPy arrays. Implements Dice coefficient, Jaccard index (IoU), accuracy, and boundary F1 score for comprehensive segmentation evaluation.

This module provides NumPy-based implementations of segmentation metrics that work
with prediction and ground truth arrays loaded from NIfTI files. These metrics
are used in the evaluation pipeline for assessing model performance.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice_coefficient(pred, gt, label, smooth=1e-5):
    """
    Compute the Dice coefficient (F1 score) for a specific label.

    The Dice coefficient measures the overlap between predicted and ground truth
    segmentations for a specific class. It is defined as: 2 * |A ∩ B| / (|A| + |B|),
    where A and B are the predicted and ground truth masks for the given label.

    Args:
        pred (np.ndarray): Prediction array with class labels.
        gt (np.ndarray): Ground truth array with class labels.
        label (int): Label value to compute Dice coefficient for.
        smooth (float): Smoothing factor to prevent division by zero.
            Default: 1e-5.

    Returns:
        float: Dice coefficient for the specified label.

    Note:
        This function computes Dice coefficient for a single label only.
        For multi-class evaluation, call this function for each label separately.
    """
    pred_bin = (pred == label)
    gt_bin = (gt == label)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    return (2. * intersection + smooth) / (union + smooth)

def jaccard_index(pred, gt, label, smooth=1e-5):
    """
    Compute the Jaccard index (Intersection over Union, IoU) for a specific label.

    The Jaccard index measures the overlap between predicted and ground truth
    segmentations for a specific class. It is defined as: |A ∩ B| / |A ∪ B|,
    where A and B are the predicted and ground truth masks for the given label.

    Args:
        pred (np.ndarray): Prediction array with class labels.
        gt (np.ndarray): Ground truth array with class labels.
        label (int): Label value to compute IoU for.
        smooth (float): Smoothing factor to prevent division by zero.
            Default: 1e-5.

    Returns:
        float: Jaccard index for the specified label.

    Note:
        This function computes IoU for a single label only.
        For multi-class evaluation, call this function for each label separately.
    """
    pred_bin = (pred == label)
    gt_bin = (gt == label)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    return (intersection + smooth) / (union + smooth)

def accuracy(pred, gt, label):
    """
    Compute pixel-wise accuracy for a specific label.

    Accuracy measures the proportion of correctly classified pixels for a
    specific class across the entire image volume.

    Args:
        pred (np.ndarray): Prediction array with class labels.
        gt (np.ndarray): Ground truth array with class labels.
        label (int): Label value to compute accuracy for.

    Returns:
        float: Pixel-wise accuracy for the specified label.

    Note:
        This metric gives equal weight to all pixels regardless of class
        frequency, which may not be ideal for imbalanced datasets.
    """
    pred_bin = (pred == label)
    gt_bin = (gt == label)
    correct = np.sum(pred_bin == gt_bin)
    total = np.prod(pred.shape)
    return correct / total

def extract_boundary(mask):
    """
    Extract boundary pixels from a binary mask using morphological operations.

    This function uses binary erosion to identify boundary pixels by subtracting
    the eroded mask from the original mask.

    Args:
        mask (np.ndarray): Binary mask array.

    Returns:
        np.ndarray: Binary array indicating boundary pixels.

    Note:
        This is a helper function used by boundary_f1_score to extract
        boundary information from segmentation masks.
    """

    # mask: binary np array
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    return boundary

def boundary_f1_score(pred, gt, label, tolerance=1):
    """
    Compute boundary F1 score for a specific label.

    The boundary F1 score measures the accuracy of segmentation boundaries
    by computing the F1 score between predicted and ground truth boundary
    pixels within a specified tolerance distance using distance transforms.

    Args:
        pred (np.ndarray): Prediction array with class labels.
        gt (np.ndarray): Ground truth array with class labels.
        label (int): Label value to compute boundary F1 score for.
        tolerance (int): Tolerance distance in pixels for boundary matching.
            Default: 1.

    Returns:
        float: Boundary F1 score for the specified label.

    Note:
        This metric is particularly useful for evaluating the quality of
        segmentation boundaries, which is important in medical imaging.
        The function handles edge cases where no boundaries are present.
    """
    
    pred_bin = (pred == label).astype(np.uint8)
    gt_bin = (gt == label).astype(np.uint8)
    if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
        return 1.0
    pred_b = extract_boundary(pred_bin)
    gt_b = extract_boundary(gt_bin)
    if np.sum(pred_b) == 0 and np.sum(gt_b) == 0:
        return 1.0
    if np.sum(pred_b) == 0 or np.sum(gt_b) == 0:
        return 0.0
    # Distance transforms
    dt_pred = distance_transform_edt(1 - pred_b)
    dt_gt = distance_transform_edt(1 - gt_b)
    # Precision: fraction of pred boundary within tol of GT boundary
    pred_match = dt_gt[pred_b > 0] <= tolerance
    precision = np.sum(pred_match) / (np.sum(pred_b) + 1e-8)
    # Recall: fraction of GT boundary within tol of pred boundary
    gt_match = dt_pred[gt_b > 0] <= tolerance
    recall = np.sum(gt_match) / (np.sum(gt_b) + 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall) 