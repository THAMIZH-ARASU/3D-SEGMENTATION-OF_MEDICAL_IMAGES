from typing import Tuple
import torch.nn as nn
import torch

class PatchMerging(nn.Module):
    """Patch Merging Layer (3D)."""
    
    def __init__(self, input_resolution: Tuple[int, int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."

        x = x.view(B, D, H, W, C)

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = x.view(B, -1, 8 * C)  # B D/2*H/2*W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)
        return x