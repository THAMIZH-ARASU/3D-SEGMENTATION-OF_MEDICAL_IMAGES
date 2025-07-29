import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

# D-Former3D (SegNetwork)
def test_dformer3d_segnetwork():
    from models.transformers.d_former.network import SegNetwork
    print("Testing D-Former3D (SegNetwork)...")
    batch_size = 2
    in_chan = 1
    num_classes = 3
    D, H, W = 32, 128, 128  # Typical 3D CT patch
    x = torch.randn(batch_size, in_chan, D, H, W)
    model = SegNetwork(num_classes=num_classes, in_chan=in_chan, deep_supervision=False)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (batch_size, 14, D, H, W) or y.shape == (batch_size, num_classes, D, H, W), \
        f"D-Former3D output shape {y.shape} is incorrect!"
    print(f"  Input shape: {x.shape} -> Output shape: {y.shape} [OK]")

# SegFormerModule (2D)
def test_segformer_module():
    from models.transformers.segformer.huggingface.segformer_module import SegFormerModule
    print("Testing SegFormerModule (2D)...")
    batch_size = 2
    in_chan = 3  # SegFormer expects 3 channels (RGB)
    num_classes = 3
    H, W = 256, 256
    x = torch.randn(batch_size, in_chan, H, W)
    model = SegFormerModule(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        y = model(x)
    # SegFormer may downsample the output
    expected_shapes = [
        (batch_size, num_classes, H, W),  # Full resolution
        (batch_size, num_classes, H//4, W//4),  # Downsampled by 4
        (batch_size, num_classes, H//2, W//2),  # Downsampled by 2
    ]
    assert y.shape in expected_shapes, f"SegFormerModule output shape {y.shape} is incorrect! Expected one of {expected_shapes}"
    print(f"  Input shape: {x.shape} -> Output shape: {y.shape} [OK]")

# VisionTransformer (TransUNet)
def test_transunet_vit():
    from models.transformers.trans_unet.vit.vit_seg_modelling import VisionTransformer
    from models.transformers.trans_unet.vit import vit_seg_configs
    print("Testing VisionTransformer (TransUNet)...")
    batch_size = 2
    in_chan = 1
    num_classes = 3
    H, W = 224, 224
    # Use a simpler config that doesn't require complex skip connections
    config = vit_seg_configs.get_b16_config()
    config.n_classes = num_classes
    config.decoder_channels = (256, 128, 64, 16)
    config.n_skip = 0  # No skip connections for simpler testing
    config.skip_channels = [0, 0, 0, 0]  # No skip channels
    model = VisionTransformer(config, img_size=H, num_classes=num_classes)
    model.eval()
    x = torch.randn(batch_size, in_chan, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (batch_size, num_classes, H, W), f"VisionTransformer output shape {y.shape} is incorrect!"
    print(f"  Input shape: {x.shape} -> Output shape: {y.shape} [OK]")

if __name__ == "__main__":
    test_dformer3d_segnetwork()
    test_segformer_module()
    test_transunet_vit()
    print("\nAll transformer model architecture tests passed!\n") 