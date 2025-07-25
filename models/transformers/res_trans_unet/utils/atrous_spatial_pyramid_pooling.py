from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module (3D)."""
    
    def __init__(self, in_channels: int, out_channels: int, rates: List[int] = [1, 6, 12, 18]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # 1x1 convolution
        self.convs.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in rates[1:]:
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv3d(len(rates) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )
    
    def forward(self, x):
        features = []
        
        # Apply all convolutions
        for conv in self.convs:
            features.append(conv(x))
        
        # Global average pooling
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=x.shape[-3:], mode='trilinear', align_corners=False)
        features.append(gap)
        
        # Concatenate and project
        out = torch.cat(features, dim=1)
        return self.project(out)