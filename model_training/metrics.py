import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion

# All metric functions expect tensors as produced by batch['image'][tio.DATA] and batch['label'][tio.DATA]
# i.e., (B, C, ...) for images and (B, ...) for labels

def dice_coefficient(preds, targets, num_classes=3, smooth=1e-5):
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
    # preds: (B, C, ...), targets: (B, ...)
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float()
    return correct.mean().item()

def boundary_f1_score(preds, targets, num_classes=3, tolerance=1):
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