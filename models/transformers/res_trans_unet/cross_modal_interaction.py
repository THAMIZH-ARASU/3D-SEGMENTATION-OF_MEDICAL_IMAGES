import torch.nn as nn
import torch.nn.functional as F

class CrossModalInteraction(nn.Module):
    """Cross-modal interaction between Swin features and CNN features (3D)."""
    
    def __init__(self, swin_dim: int, cnn_channels: int):
        super().__init__()
        self.swin_dim = swin_dim
        self.cnn_channels = cnn_channels
        
        # Projection layer to match dimensions
        self.swin_proj = nn.Conv3d(swin_dim, cnn_channels, kernel_size=3, padding=1)
        
        # Attention mechanism
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels, cnn_channels),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, swin_features, cnn_features):
        B, L, C = swin_features.shape
        b, c, d, h, w = cnn_features.shape
        # Always infer the spatial shape from L
        D_s = H_s = W_s = int(round(L ** (1/3)))
        assert D_s * H_s * W_s == L, f"Cannot infer 3D shape from L={L}, got D_s={D_s}"
        swin_spatial = swin_features.view(B, D_s, H_s, W_s, C).permute(0, 4, 1, 2, 3)
        # Upsample Swin features to match CNN feature size
        swin_upsampled = F.interpolate(swin_spatial, size=(d, h, w), mode='nearest')
        # Project Swin features to match CNN channels
        swin_projected = self.swin_proj(swin_upsampled)
        # Gate mechanism
        gate = self.sigmoid(swin_projected)
        gated_features = gate * cnn_features + cnn_features
        # Channel attention
        attention_weights = self.global_pool(gated_features).view(b, c)
        attention_weights = self.fc(attention_weights).view(b, c, 1, 1, 1)
        output = gated_features * attention_weights
        return output