from einops import rearrange
from torch import nn
import torch
import torch.nn.functional
import torch.nn.functional as F

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, 0::2, 0::2, 1::2, :]  # B D H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D H/2 W/2 C
        x3 = x[:, 0::2, 1::2, 1::2, :]  # B D H/2 W/2 C
        x4 = x[:, 1::2, 0::2, 0::2, :]  # B D H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D H/2 W/2 C
        x6 = x[:, 1::2, 1::2, 0::2, :]  # B D H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D H/2 W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand3D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        x = self.expand(x)
        B, D, H, W, C = x.shape

        x = rearrange(x, 'b d h w (p0 p1 p2 c)-> b (d p0) (h p1) (w p2) c', p0=2, p1=2, p2=2, c=C//8)
        x = self.norm(x)
        return x