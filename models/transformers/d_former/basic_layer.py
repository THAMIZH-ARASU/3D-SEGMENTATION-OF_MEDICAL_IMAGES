from einops import rearrange
from torch import nn

from models.transformers.d_former.utils.dformer_block_3d import DFormerBlock3D
from models.transformers.d_former.utils.post_cnn import PosCNN

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 interval,
                 depth,
                 num_heads,
                 group_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 i_layer=None):
        super().__init__()
        self.group_size = group_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DFormerBlock3D(
                dim=dim,
                num_heads=num_heads,
                group_size=group_size,
                interval=interval,
                gsm=0 if (i % 2 == 0) else 1,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

        self.pos_block = PosCNN(in_chans=dim, embed_dim=dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.pos_block(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x