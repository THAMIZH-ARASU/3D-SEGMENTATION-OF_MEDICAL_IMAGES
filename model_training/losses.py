import torch
import torch.nn as nn

class SegmentationCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        return self.ce(logits, targets) 