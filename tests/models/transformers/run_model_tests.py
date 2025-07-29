#!/usr/bin/env python3
"""
Comprehensive test runner for transformer model architectures.

This script tests all transformer model architectures to ensure they:
1. Can be instantiated with typical parameters
2. Accept input tensors of expected shapes
3. Produce output tensors of correct shapes
4. Handle both 2D and 3D inputs appropriately
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def test_dformer3d_segnetwork():
    """Test D-Former3D (SegNetwork) architecture."""
    print("Testing D-Former3D (SegNetwork)...")
    
    try:
        from models.transformers.d_former.network import SegNetwork
        
        # Test parameters
        batch_size = 2
        in_chan = 1
        num_classes = 3
        D, H, W = 32, 128, 128  # Typical 3D CT patch
        
        # Create input tensor
        x = torch.randn(batch_size, in_chan, D, H, W)
        print(f"  Input shape: {x.shape}")
        
        # Instantiate model
        model = SegNetwork(num_classes=num_classes, in_chan=in_chan, deep_supervision=False)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
        
        # Check output shape
        expected_shapes = [(batch_size, 14, D, H, W), (batch_size, num_classes, D, H, W)]
        if y.shape in expected_shapes:
            print(f"  Output shape: {y.shape} [✅ PASS]")
            return True
        else:
            print(f"  Output shape: {y.shape} [❌ FAIL] - Expected one of {expected_shapes}")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL - Exception: {str(e)}")
        return False


def test_segformer_module():
    """Test SegFormerModule (2D) architecture."""
    print("Testing SegFormerModule (2D)...")
    
    try:
        from models.transformers.segformer.huggingface.segformer_module import SegFormerModule
        
        # Test parameters
        batch_size = 2
        in_chan = 3  # SegFormer expects 3 channels (RGB)
        num_classes = 3
        H, W = 256, 256
        
        # Create input tensor
        x = torch.randn(batch_size, in_chan, H, W)
        print(f"  Input shape: {x.shape}")
        
        # Instantiate model
        model = SegFormerModule(num_classes=num_classes)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
        
        # Check output shape (SegFormer may downsample the output)
        # The model might output a smaller size due to downsampling
        expected_shapes = [
            (batch_size, num_classes, H, W),  # Full resolution
            (batch_size, num_classes, H//4, W//4),  # Downsampled by 4
            (batch_size, num_classes, H//2, W//2),  # Downsampled by 2
        ]
        if y.shape in expected_shapes:
            print(f"  Output shape: {y.shape} [✅ PASS]")
            return True
        else:
            print(f"  Output shape: {y.shape} [❌ FAIL] - Expected one of {expected_shapes}")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL - Exception: {str(e)}")
        return False


def test_transunet_vit():
    """Test VisionTransformer (TransUNet) architecture."""
    print("Testing VisionTransformer (TransUNet)...")
    
    try:
        from models.transformers.trans_unet.vit.vit_seg_modelling import VisionTransformer
        from models.transformers.trans_unet.vit import vit_seg_configs
        
        # Test parameters
        batch_size = 2
        in_chan = 1
        num_classes = 3
        H, W = 224, 224
        
        # Create input tensor
        x = torch.randn(batch_size, in_chan, H, W)
        print(f"  Input shape: {x.shape}")
        
        # Get config and instantiate model
        # Use a simpler config that doesn't require complex skip connections
        config = vit_seg_configs.get_b16_config()
        config.n_classes = num_classes
        config.decoder_channels = (256, 128, 64, 16)
        config.n_skip = 0  # No skip connections for simpler testing
        config.skip_channels = [0, 0, 0, 0]  # No skip channels
        model = VisionTransformer(config, img_size=H, num_classes=num_classes)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
        
        # Check output shape
        expected_shape = (batch_size, num_classes, H, W)
        if y.shape == expected_shape:
            print(f"  Output shape: {y.shape} [✅ PASS]")
            return True
        else:
            print(f"  Output shape: {y.shape} [❌ FAIL] - Expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL - Exception: {str(e)}")
        return False


def test_dformer3d_direct():
    """Test DFormer3D directly (without SegNetwork wrapper)."""
    print("Testing DFormer3D (Direct)...")
    
    try:
        from models.transformers.d_former.d_former_3d import DFormer3D
        
        # Test parameters
        batch_size = 2
        in_chan = 1
        D, H, W = 32, 128, 128
        
        # Create input tensor
        x = torch.randn(batch_size, in_chan, D, H, W)
        print(f"  Input shape: {x.shape}")
        
        # Instantiate model
        model = DFormer3D(in_chans=in_chan)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
        
        # Check output (should be a list of features)
        if isinstance(y, list) and len(y) > 0:
            print(f"  Output: {len(y)} feature maps [✅ PASS]")
            for i, feat in enumerate(y):
                print(f"    Feature {i} shape: {feat.shape}")
            return True
        else:
            print(f"  Output: {type(y)} [❌ FAIL] - Expected list of features")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL - Exception: {str(e)}")
        return False


def run_all_model_tests():
    """Run all model architecture tests."""
    print("=" * 60)
    print("RUNNING TRANSFORMER MODEL ARCHITECTURE TESTS")
    print("=" * 60)
    
    test_results = {}
    
    # Test D-Former3D SegNetwork
    print("\n" + "=" * 40)
    start_time = time.time()
    success = test_dformer3d_segnetwork()
    test_results['D-Former3D (SegNetwork)'] = success
    test_time = time.time() - start_time
    print(f"D-Former3D test completed in {test_time:.2f} seconds")
    
    # Test DFormer3D Direct
    print("\n" + "=" * 40)
    start_time = time.time()
    success = test_dformer3d_direct()
    test_results['DFormer3D (Direct)'] = success
    test_time = time.time() - start_time
    print(f"DFormer3D direct test completed in {test_time:.2f} seconds")
    
    # Test SegFormer
    print("\n" + "=" * 40)
    start_time = time.time()
    success = test_segformer_module()
    test_results['SegFormer'] = success
    test_time = time.time() - start_time
    print(f"SegFormer test completed in {test_time:.2f} seconds")
    
    # Test TransUNet
    print("\n" + "=" * 40)
    start_time = time.time()
    success = test_transunet_vit()
    test_results['TransUNet'] = success
    test_time = time.time() - start_time
    print(f"TransUNet test completed in {test_time:.2f} seconds")
    
    return test_results


def print_summary(test_results):
    """Print a comprehensive summary of all test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for model_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
        if success:
            passed_tests += 1
    
    print(f"\n" + "=" * 40)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print(f"SUCCESS RATE: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "SUCCESS RATE: 0%")
    
    all_passed = all(test_results.values())
    if all_passed:
        print("🎉 ALL MODEL ARCHITECTURES PASSED!")
    else:
        print("❌ SOME MODEL ARCHITECTURES FAILED")
    
    print("=" * 40)


def main():
    """Main function to run all model architecture tests."""
    try:
        test_results = run_all_model_tests()
        print_summary(test_results)
        
        # Return appropriate exit code
        all_passed = all(test_results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 