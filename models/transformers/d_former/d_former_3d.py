from einops import rearrange
from torch import nn
import torch
import torch.nn.functional
import collections
from timm.models.layers import trunc_normal_

from models.transformers.d_former.basic_layer import BasicLayer
from models.transformers.d_former.basic_layer_up import BasicLayer_up
from models.transformers.d_former.patch_embed_3d import PatchEmbed3D
from models.transformers.d_former.utils.patch_merge_and_expand import PatchExpand3D, PatchMerging

class DFormer3D(nn.Module):
    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 interval_list=[8, 4, 2, 1],
                 patch_size=(2, 4, 4),
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 group_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.group_size = group_size
        self.patch_size = patch_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                group_size=group_size,
                interval=interval_list[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                i_layer=i_layer
            )
            self.layers.append(layer)


        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                layer_up = PatchExpand3D(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         interval=interval_list[(self.num_layers - 1 - i_layer)],
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         group_size=group_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand3D if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self._freeze_stages()
        self.init_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        pos_keys = [k for k in state_dict.keys() if "pos" in k]
        for k in pos_keys:
            del state_dict[k]

        # delete relative position biases
        biases_keys = [k for k in state_dict.keys() if "biases" in k]
        for k in biases_keys:
            del state_dict[k]
        attn_index_keys = [k for k in state_dict.keys() if "attn" in k]
        weight = collections.OrderedDict()
        for k in attn_index_keys:
            weight[k] = state_dict[k]

        mlp_keys = [k for k in state_dict.keys() if "mlp" in k]
        for k in mlp_keys:
            weight[k] = state_dict[k]

        up_attn_keys = [k for k in self.state_dict().keys() if ("attn" or "layers_up.") in k]
        up_attn_keys = up_attn_keys[len(attn_index_keys):]
        match_attn_keys = attn_index_keys[:len(up_attn_keys)]
        s1_up = up_attn_keys[0:24]
        s1_match = match_attn_keys[16:]
        s2_up = up_attn_keys[24:32]
        s2_match = match_attn_keys[8:16]
        s3_up = up_attn_keys[32:]
        s3_match = match_attn_keys[:8]

        for i in range(len(s1_up)):
            weight[s1_up[i]] = state_dict[s1_match[i]]

        for i in range(len(s2_up)):
            weight[s2_up[i]] = state_dict[s2_match[i]]

        for i in range(len(s3_up)):
            weight[s3_up[i]] = state_dict[s3_match[i]]


        self.load_state_dict(weight, strict=False)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.pretrained != None:
            if isinstance(self.pretrained, str):
                self.apply(_init_weights)
                if self.pretrained2d:
                    # Inflate 2D model into 3D model.
                    self.inflate_weights()
            else:
                raise TypeError('pretrained must be a str or None')


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(rearrange(x, 'n c d h w -> n d h w c'))
            x = layer(x.contiguous())
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        x_upsample = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[(self.num_layers-1) - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            if inx < 2:
                x_upsample.append(x)
        x = self.norm_up(x)  # B L C
        x_upsample.append(x)
        return x_upsample


    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x_upsample = self.forward_up_features(x, x_downsample)
        return x_upsample