from typing import Optional, Tuple
from models.transformers.res_trans_unet.utils.swin_transformer_block import SwinTransformerBlock
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int, 
                 num_heads: int, window_size: int, mlp_ratio: float = 4., 
                 qkv_bias: bool = True, qk_scale: Optional[float] = None, 
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0., 
                 norm_layer=nn.LayerNorm, downsample: Optional[nn.Module] = None, 
                 use_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x