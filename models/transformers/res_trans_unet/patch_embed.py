from typing import Optional
import torch.nn as nn
from models.transformers.res_trans_unet.utils.drop_path import to_2tuple

def to_3tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x, x)

class PatchEmbed(nn.Module):
    """3D Image to Patch Embedding."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, 
                 embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert (D, H, W) == self.img_size, \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Pd*Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x