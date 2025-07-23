from torch import nn
import torch.utils.checkpoint as checkpoint

from models.transformers.d_former.utils.dformer_block_3d import DFormerBlock3D
from models.transformers.d_former.utils.patch_merge_and_expand import PatchExpand3D
from models.transformers.d_former.utils.post_cnn import PosCNN

class BasicLayer_up(nn.Module):
    def __init__(self, dim, depth, num_heads, group_size, interval,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.group_size = group_size
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DFormerBlock3D(dim=dim,
                                   num_heads=num_heads,
                                   group_size=group_size,
                                   interval=interval,
                                   gsm=0 if (i % 2 == 0) else 1,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer)
            for i in range(depth)])

        self.pos_block = PosCNN(in_chans=dim, embed_dim=dim)
        # patch merging layer
        self.upsample = upsample
        if upsample is not None:
            self.upsample = PatchExpand3D(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = self.pos_block(x)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x