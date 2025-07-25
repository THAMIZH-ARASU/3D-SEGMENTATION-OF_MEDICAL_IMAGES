import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice_coefficient(pred, gt, label, smooth=1e-5):
    pred_bin = (pred == label)
    gt_bin = (gt == label)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    return (2. * intersection + smooth) / (union + smooth)

def jaccard_index(pred, gt, label, smooth=1e-5):
    pred_bin = (pred == label)
    gt_bin = (gt == label)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    return (intersection + smooth) / (union + smooth)

def accuracy(pred, gt, label):
    pred_bin = (pred == label)
    gt_bin = (gt == label)
    correct = np.sum(pred_bin == gt_bin)
    total = np.prod(pred.shape)
    return correct / total

def extract_boundary(mask):
    # mask: binary np array
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    return boundary

def boundary_f1_score(pred, gt, label, tolerance=1):
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