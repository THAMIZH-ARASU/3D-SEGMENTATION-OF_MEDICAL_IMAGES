from torch import nn
import torch
import torch.nn.functional
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import reduce
from operator import mul
from timm.layers import DropPath, to_3tuple

from models.transformers.d_former.utils.attention import Attention
from models.transformers.d_former.utils.config import NEG_INF
from models.transformers.d_former.utils.mlp import Mlp

class DFormerBlock3D(nn.Module):

    def __init__(self, dim, num_heads, group_size=(2, 7, 7), interval=8, gsm=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.gsm = gsm
        self.interval = interval


        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, group_size=self.group_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        B, D, H, W, C = x.shape

        x = self.norm1(x)

        if H < self.group_size[1]:
            # if group size is larger than input resolution, we don't partition group
            self.gsm = 0
            self.group_size = (D, H, W)
        # pad feature maps to multiples of group size
        size_div = self.interval if self.gsm == 1 else self.group_size
        if isinstance(size_div, int): size_div = to_3tuple(size_div)
        pad_l = pad_t = pad_d0 = 0
        pad_d = (size_div[0] - D % size_div[0]) % size_div[0]
        pad_b = (size_div[1] - H % size_div[1]) % size_div[1]
        pad_r = (size_div[2] - W % size_div[2]) % size_div[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape

        mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
        if pad_d > 0:
            mask[:, -pad_d:, :, :, :] = -1
        if pad_b > 0:
            mask[:, :, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, :, -pad_r:, :] = -1

        # group embeddings and generate attn_mask
        if self.gsm == 0: # LS-MSA
            Gd = size_div[0]
            Gh = size_div[1]
            Gw = size_div[2]
            B, D2, H2, W2, C = x.shape

            x = x.view(B, D2 // Gd, Gd, H2 // Gh, Gh, W2 // Gw, Gw, C).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            x = x.reshape(-1, reduce(mul, size_div), C)

            nG = (Dp * Hp * Wp) // (Gd * Gh * Gw)  # group_num

            if pad_r > 0 or pad_b > 0 or pad_d > 0:
                mask = mask.reshape(1, Dp // Gd, Gd, Hp // Gh, Gh, Wp // Gw, Gw, 1).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()

                mask = mask.reshape(nG, 1, Gd * Gh * Gw)
                attn_mask = torch.zeros((nG, Gd * Gh * Gw, Gd * Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        else: # GS-MSA
            B, D2, H2, W2, C = x.shape
            interval_d = Dp // self.group_size[0]
            interval_h = Hp // self.group_size[1]
            interval_w = Wp // self.group_size[2]

            Id, Ih, Iw = interval_d, interval_h, interval_w
            Gd, Gh, Gw = Dp // interval_d, Hp // interval_h, Wp // interval_w
            x = x.reshape(B, Gd, Id, Gh, Ih, Gw, Iw, C).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            x = x.reshape(B * Id * Ih * Iw, Gd * Gh * Gw, C)

            nG = interval_d * interval_h * interval_w  # group_num

            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gd, Id, Gh, Ih, Gw, Iw, 1).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
                mask = mask.reshape(nG, 1, Gd * Gh * Gw)
                attn_mask = torch.zeros((nG, Gd * Gh * Gw, Gd * Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # multi-head self-attention
        x = self.attn(x, mask=attn_mask)

        # ungroup embeddings
        if self.gsm == 0:
            x = x.reshape(B, D2 // size_div[0], H2 // size_div[1], W2 // size_div[2], size_div[0], size_div[1],
                       size_div[2], C).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous() # B, Hp//G, G, Wp//G, G, C
            x = x.view(B, D2, H2, W2, -1)
        else:
            x = x.reshape(B, interval_d, interval_h, interval_w,
                          D2 // interval_d, H2 // interval_h, W2 // interval_w, C)\
                .permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous() # B, Gh, I, Gw, I, C

            x = x.view(B, D2, H2, W2, -1)

        # remove padding
        if pad_d > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x