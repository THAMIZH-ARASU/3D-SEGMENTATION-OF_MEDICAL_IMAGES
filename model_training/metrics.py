import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion

# All metric functions expect tensors as produced by batch['image'][tio.DATA] and batch['label'][tio.DATA]
# i.e., (B, C, ...) for images and (B, ...) for labels

def dice_coefficient(preds, targets, num_classes=3, smooth=1e-5):
    preds = torch.argmax(preds, dim=1)
    if preds.dim() == 3:  # (B, H, W)
        preds_one_hot = F.one_hot(preds, num_classes).permute(0, 3, 1, 2)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2)
        dims = (0, 2, 3)
    elif preds.dim() == 4:  # (B, D, H, W)
        preds_one_hot = F.one_hot(preds, num_classes).permute(0, 4, 1, 2, 3)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3)
        dims = (0, 2, 3, 4)
    else:
        raise ValueError(f'Unsupported prediction shape: {preds.shape}')
    intersection = torch.sum(preds_one_hot * targets_one_hot, dim=dims)
    union = torch.sum(preds_one_hot, dim=dims) + torch.sum(targets_one_hot, dim=dims)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def jaccard_index(preds, targets, num_classes=3, smooth=1e-5):
    preds = torch.argmax(preds, dim=1)
    if preds.dim() == 3:
        preds_one_hot = F.one_hot(preds, num_classes).permute(0, 3, 1, 2)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2)
        dims = (0, 2, 3)
    elif preds.dim() == 4:
        preds_one_hot = F.one_hot(preds, num_classes).permute(0, 4, 1, 2, 3)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3)
        dims = (0, 2, 3, 4)
    else:
        raise ValueError(f'Unsupported prediction shape: {preds.shape}')
    intersection = torch.sum(preds_one_hot * targets_one_hot, dim=dims)
    union = torch.sum(preds_one_hot, dim=dims) + torch.sum(targets_one_hot, dim=dims) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def accuracy(preds, targets):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float()
    return correct.mean().item()

def boundary_f1_score(preds, targets, num_classes=3, tolerance=1):
    preds = torch.argmax(preds, dim=1)
    if preds.dim() == 3:
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        bf1s = []
        for c in range(1, num_classes):
            bf1s_c = []
            for p, t in zip(preds, targets):
                p_bin = (p == c).astype(np.uint8)
                t_bin = (t == c).astype(np.uint8)
                if np.sum(p_bin) == 0 and np.sum(t_bin) == 0:
                    bf1s_c.append(1.0)
                    continue
                pred_b = p_bin ^ binary_erosion(p_bin)
                gt_b = t_bin ^ binary_erosion(t_bin)
                if np.sum(pred_b) == 0 and np.sum(gt_b) == 0:
                    bf1s_c.append(1.0)
                    continue
                if np.sum(pred_b) == 0 or np.sum(gt_b) == 0:
                    bf1s_c.append(0.0)
                    continue
                from scipy.ndimage import distance_transform_edt
                dt_pred = distance_transform_edt(1 - pred_b)
                dt_gt = distance_transform_edt(1 - gt_b)
                pred_match = dt_gt[pred_b > 0] <= tolerance
                precision = np.sum(pred_match) / (np.sum(pred_b) + 1e-8)
                gt_match = dt_pred[gt_b > 0] <= tolerance
                recall = np.sum(gt_match) / (np.sum(gt_b) + 1e-8)
                if precision + recall == 0:
                    bf1 = 0.0
                else:
                    bf1 = 2 * precision * recall / (precision + recall)
                bf1s_c.append(bf1)
            bf1s.append(np.mean(bf1s_c))
        return float(np.mean(bf1s))
    elif preds.dim() == 4:
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        bf1s = []
        for c in range(1, num_classes):
            bf1s_c = []
            for p, t in zip(preds, targets):
                for d in range(p.shape[0]):
                    p_bin = (p[d] == c).astype(np.uint8)
                    t_bin = (t[d] == c).astype(np.uint8)
                    if np.sum(p_bin) == 0 and np.sum(t_bin) == 0:
                        bf1s_c.append(1.0)
                        continue
                    pred_b = p_bin ^ binary_erosion(p_bin)
                    gt_b = t_bin ^ binary_erosion(t_bin)
                    if np.sum(pred_b) == 0 and np.sum(gt_b) == 0:
                        bf1s_c.append(1.0)
                        continue
                    if np.sum(pred_b) == 0 or np.sum(gt_b) == 0:
                        bf1s_c.append(0.0)
                        continue
                    from scipy.ndimage import distance_transform_edt
                    dt_pred = distance_transform_edt(1 - pred_b)
                    dt_gt = distance_transform_edt(1 - gt_b)
                    pred_match = dt_gt[pred_b > 0] <= tolerance
                    precision = np.sum(pred_match) / (np.sum(pred_b) + 1e-8)
                    gt_match = dt_pred[gt_b > 0] <= tolerance
                    recall = np.sum(gt_match) / (np.sum(gt_b) + 1e-8)
                    if precision + recall == 0:
                        bf1 = 0.0
                    else:
                        bf1 = 2 * precision * recall / (precision + recall)
                    bf1s_c.append(bf1)
            bf1s.append(np.mean(bf1s_c))
        return float(np.mean(bf1s))
    else:
        raise ValueError(f'Unsupported prediction shape: {preds.shape}') 