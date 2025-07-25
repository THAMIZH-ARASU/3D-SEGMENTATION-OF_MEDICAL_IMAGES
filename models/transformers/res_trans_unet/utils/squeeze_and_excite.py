import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (3D)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        out = self.global_avgpool(x).view(B, C)
        out = self.fc(out).view(B, C, 1, 1, 1)
        return x * out.expand_as(x)