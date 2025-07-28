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
    from models.transformers.segformer.segformer_module import SegFormerModule
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
    assert y.shape == (batch_size, num_classes, H, W), f"SegFormerModule output shape {y.shape} is incorrect!"
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
    config = vit_seg_configs.get_testing()
    config.n_classes = num_classes
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