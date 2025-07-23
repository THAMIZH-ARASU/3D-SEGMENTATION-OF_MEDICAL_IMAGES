from einops import rearrange
from torch import nn

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, patch_size=(2, 4, 4), norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.up_scale = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.expand = nn.Linear(dim, self.up_scale*dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        x = self.expand(x)
        B, D, H, W, C = x.shape

        x = x.view(B, D, H, W, C)

        x = rearrange(x, 'b d h w (p0 p1 p2 c)-> b (d p0) (h p1) (w p2) c', p0=self.patch_size[0], p1=self.patch_size[1], p2=self.patch_size[2], c=C//self.up_scale)
        x = self.norm(x)
        x = x.view(B, self.patch_size[0] * D, self.patch_size[1] * H, self.patch_size[2] * W, -1)
        x = x.permute(0, 4, 1, 2, 3)  # B,C,D,H,W

        return x