import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, ...), targets: (B, ...)
        num_classes = logits.shape[1]
        if logits.shape == targets.shape:
            targets_one_hot = targets
        else:
            targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, -1, *range(1, targets.dim()))
        probs = torch.softmax(logits, dim=1)
        dims = tuple(range(2, logits.dim()))
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(targets_one_hot, dim=dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets) 