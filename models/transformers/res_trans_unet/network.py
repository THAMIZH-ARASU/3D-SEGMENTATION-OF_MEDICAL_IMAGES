from typing import List, Optional
import torch.nn as nn
import torch

from models.transformers.res_trans_unet.basic_layer import BasicLayer
from models.transformers.res_trans_unet.patch_embed import PatchEmbed
from models.transformers.res_trans_unet.utils.atrous_spatial_pyramid_pooling import ASPP
from models.transformers.res_trans_unet.utils.drop_path import trunc_normal_
from models.transformers.res_trans_unet.utils.patch_merge import PatchMerging
from models.transformers.res_trans_unet.utils.residual_se_block import ResidualSEBlock
from models.transformers.res_trans_unet.utils.upsample import UpConv
from models.transformers.res_trans_unet.cross_modal_interaction import CrossModalInteraction

class ResTransUNet(nn.Module):
    """ResTransUNet: Hybrid architecture combining Swin Transformer with U-Net."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 1, 
                 num_classes: int = 4, embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2], 
                 num_heads: List[int] = [3, 6, 12, 24], window_size: int = 7, 
                 mlp_ratio: float = 4., qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm, ape: bool = False, patch_norm: bool = True,
                 use_checkpoint: bool = False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # CNN encoder filters
        n1 = 32
        self.filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # [32, 64, 128, 256, 512]

        # =====================================================================
        # Patch Embedding for Swin Transformer
        # =====================================================================
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # =====================================================================
        # Swin Transformer Layers
        # =====================================================================
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # =====================================================================
        # CNN Encoder Path
        # =====================================================================
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ResidualSEBlock(in_chans, self.filters[0])
        self.conv2 = ResidualSEBlock(self.filters[0], self.filters[1])
        self.conv3 = ResidualSEBlock(self.filters[1], self.filters[2])
        self.conv4 = ResidualSEBlock(self.filters[2], self.filters[3])
        self.conv5 = ResidualSEBlock(self.filters[3], self.filters[4])

        # =====================================================================
        # Cross-Modal Interaction Modules
        # =====================================================================
        self.interaction2 = CrossModalInteraction(embed_dim, self.filters[1])
        self.interaction3 = CrossModalInteraction(embed_dim * 2, self.filters[2])
        self.interaction4 = CrossModalInteraction(embed_dim * 4, self.filters[3])
        self.interaction5 = CrossModalInteraction(embed_dim * 8, self.filters[4])

        # =====================================================================
        # Bridge and Decoder
        # =====================================================================
        # Swin feature processing
        self.swin_upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.num_features, self.filters[4], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.filters[4]),
            nn.LeakyReLU(inplace=True)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(self.filters[4] * 2, self.filters[4], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.filters[4]),
            nn.LeakyReLU(inplace=True)
        )

        # ASPP modules
        self.aspp_bridge = ASPP(self.filters[4], self.filters[4])
        self.aspp_out = ASPP(self.filters[0], self.filters[0])

        # Decoder path
        self.up5 = UpConv(self.filters[4], self.filters[3])
        self.up_conv5 = ResidualSEBlock(self.filters[4], self.filters[3])

        self.up4 = UpConv(self.filters[3], self.filters[2])
        self.up_conv4 = ResidualSEBlock(self.filters[3], self.filters[2])

        self.up3 = UpConv(self.filters[2], self.filters[1])
        self.up_conv3 = ResidualSEBlock(self.filters[2], self.filters[1])

        self.up2 = UpConv(self.filters[1], self.filters[0])
        self.up_conv2 = ResidualSEBlock(self.filters[1], self.filters[0])

        # Final output layer
        self.final_conv = nn.Conv2d(self.filters[0], num_classes, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_swin_features(self, x):
        """Forward pass through Swin Transformer."""
        x = self.patch_embed(x)
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        swin_features = []
        for layer in self.layers:
            x = layer(x)
            swin_features.append(x)

        x = self.norm(x)
        swin_features[-1] = x
        
        return swin_features

    def forward(self, x):
        """Forward pass through the complete ResTransUNet architecture."""
        # =====================================================================
        # CNN Encoder Path
        # =====================================================================
        e1 = self.conv1(x)  # 224x224 -> 224x224, channels: 1 -> 32

        e2 = self.maxpool1(e1)  # 224x224 -> 112x112
        e2 = self.conv2(e2)  # channels: 32 -> 64

        e3 = self.maxpool2(e2)  # 112x112 -> 56x56
        e3 = self.conv3(e3)  # channels: 64 -> 128

        e4 = self.maxpool3(e3)  # 56x56 -> 28x28
        e4 = self.conv4(e4)  # channels: 128 -> 256

        e5 = self.maxpool4(e4)  # 28x28 -> 14x14
        e5 = self.conv5(e5)  # channels: 256 -> 512

        # =====================================================================
        # Swin Transformer Path with Cross-Modal Interactions
        # =====================================================================
        swin_features = self.forward_swin_features(x)

        # Interact with CNN features at different scales
        e2 = self.interaction2(swin_features[0], e2)  # 112x112
        e3 = self.interaction3(swin_features[1], e3)  # 56x56
        e4 = self.interaction4(swin_features[2], e4)  # 28x28
        e5 = self.interaction5(swin_features[3], e5)  # 14x14

        # =====================================================================
        # Process Swin Features for Fusion
        # =====================================================================
        # Final Swin features: B, L, C -> B, C, H, W
        B, L, C = swin_features[-1].shape
        swin_spatial = swin_features[-1].view(B, 7, 7, C).permute(0, 3, 1, 2)  # 7x7x768
        
        # Upsample and process Swin features
        swin_processed = self.swin_upconv(swin_spatial)  # 7x7 -> 14x14, 768 -> 512

        # =====================================================================
        # Feature Fusion and Bridge
        # =====================================================================
        # Fuse CNN and Swin features
        fused_features = torch.cat([e5, swin_processed], dim=1)  # Concatenate along channel dimension
        fused_features = self.feature_fusion(fused_features)  # 1024 -> 512

        # Apply ASPP for multi-scale context
        bridge = self.aspp_bridge(fused_features)

        # =====================================================================
        # Decoder Path
        # =====================================================================
        # Decoder level 5->4
        d5 = self.up5(bridge)  # 14x14 -> 28x28, 512 -> 256
        d5 = torch.cat([e4, d5], dim=1)  # Skip connection
        d5 = self.up_conv5(d5)  # 512 -> 256

        # Decoder level 4->3
        d4 = self.up4(d5)  # 28x28 -> 56x56, 256 -> 128
        d4 = torch.cat([e3, d4], dim=1)  # Skip connection
        d4 = self.up_conv4(d4)  # 256 -> 128

        # Decoder level 3->2
        d3 = self.up3(d4)  # 56x56 -> 112x112, 128 -> 64
        d3 = torch.cat([e2, d3], dim=1)  # Skip connection
        d3 = self.up_conv3(d3)  # 128 -> 64

        # Decoder level 2->1
        d2 = self.up2(d3)  # 112x112 -> 224x224, 64 -> 32
        d2 = torch.cat([e1, d2], dim=1)  # Skip connection
        d2 = self.up_conv2(d2)  # 64 -> 32

        # Apply output ASPP
        d2 = self.aspp_out(d2)

        # Final output
        output = self.final_conv(d2)  # 32 -> num_classes

        return output