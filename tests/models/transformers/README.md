# Transformer Model Architecture Tests

This directory contains comprehensive tests for all transformer model architectures in the CT scan segmentation project.

## Test Files

### Architecture Tests

- **`test_architectures.py`**: Basic model architecture tests for individual models
- **`run_model_tests.py`**: Comprehensive test runner for all transformer models with detailed output

## Running Tests

### Run All Model Tests

To run all transformer model architecture tests with comprehensive output:

```bash
python tests/models/transformers/run_model_tests.py
```

### Run Individual Model Tests

To run tests for a specific model:

```bash
# Basic architecture tests
python tests/models/transformers/test_architectures.py
```

## Test Coverage

### Model Architectures Tested

#### D-Former3D (SegNetwork)
- **Input**: 5D tensor (B, C, D, H, W) - 3D CT scan patches
- **Output**: 5D tensor (B, num_classes, D, H, W) - 3D segmentation maps
- **Features**: 
  - 3D transformer architecture
  - Deep supervision support
  - Patch embedding and decoding
  - Multi-scale feature extraction

#### DFormer3D (Direct)
- **Input**: 5D tensor (B, C, D, H, W) - 3D CT scan patches
- **Output**: List of feature maps at different scales
- **Features**:
  - Core 3D transformer backbone
  - Multi-scale feature extraction
  - Encoder-decoder structure

#### SegFormerModule (2D)
- **Input**: 4D tensor (B, C, H, W) - 2D images (RGB)
- **Output**: 4D tensor (B, num_classes, H, W) - 2D segmentation maps
- **Features**:
  - 2D transformer architecture
  - HuggingFace SegFormer integration
  - Pre-trained model support

#### VisionTransformer (TransUNet)
- **Input**: 4D tensor (B, C, H, W) - 2D images
- **Output**: 4D tensor (B, num_classes, H, W) - 2D segmentation maps
- **Features**:
  - Vision Transformer architecture
  - Hybrid CNN-Transformer design
  - Skip connections and decoder

## Test Features

### Common Test Features

All model tests include:

1. **Model Instantiation**: Creates models with typical parameters
2. **Input Tensor Creation**: Generates dummy tensors with realistic shapes
3. **Forward Pass Testing**: Runs input through the model
4. **Output Shape Validation**: Verifies output tensor dimensions
5. **Error Handling**: Catches and reports exceptions gracefully

### Model-Specific Features

#### D-Former3D Tests
- Tests both SegNetwork wrapper and direct DFormer3D
- Validates 3D input/output shapes
- Checks deep supervision functionality
- Verifies multi-scale feature extraction

#### SegFormer Tests
- Tests 2D input/output shapes
- Validates RGB channel handling
- Checks pre-trained model loading

#### TransUNet Tests
- Tests 2D input/output shapes
- Validates configuration handling
- Checks hybrid architecture

## Test Output

Tests provide detailed output including:

- ✅ **PASS**: Test completed successfully
- ❌ **FAIL**: Test failed with reason
- 📊 **Summary**: Overall test results and statistics
- ⏱️ **Timing**: Test execution time

## Example Output

```
============================================================
RUNNING TRANSFORMER MODEL ARCHITECTURE TESTS
============================================================

========================================
Testing D-Former3D (SegNetwork)...
  Input shape: torch.Size([2, 1, 32, 128, 128])
  Output shape: torch.Size([2, 3, 32, 128, 128]) [✅ PASS]
D-Former3D test completed in 3.45 seconds

========================================
Testing DFormer3D (Direct)...
  Input shape: torch.Size([2, 1, 32, 128, 128])
  Output: 3 feature maps [✅ PASS]
    Feature 0 shape: torch.Size([2, 96, 16, 64, 64])
    Feature 1 shape: torch.Size([2, 192, 8, 32, 32])
    Feature 2 shape: torch.Size([2, 384, 4, 16, 16])
DFormer3D direct test completed in 2.78 seconds

========================================
Testing SegFormerModule (2D)...
  Input shape: torch.Size([2, 3, 256, 256])
  Output shape: torch.Size([2, 3, 256, 256]) [✅ PASS]
SegFormer test completed in 1.23 seconds

========================================
Testing VisionTransformer (TransUNet)...
  Input shape: torch.Size([2, 1, 224, 224])
  Output shape: torch.Size([2, 3, 224, 224]) [✅ PASS]
TransUNet test completed in 2.34 seconds

============================================================
TEST SUMMARY
============================================================

D-Former3D (SegNetwork): ✅ PASSED
DFormer3D (Direct): ✅ PASSED
SegFormer: ✅ PASSED
TransUNet: ✅ PASSED

========================================
TOTAL: 4/4 tests passed
SUCCESS RATE: 100.0%
🎉 ALL MODEL ARCHITECTURES PASSED!
========================================
```

## Input/Output Specifications

### D-Former3D
- **Input**: `(batch_size, channels, depth, height, width)`
- **Output**: `(batch_size, num_classes, depth, height, width)`
- **Typical shapes**: `(2, 1, 32, 128, 128)` → `(2, 3, 32, 128, 128)`

### SegFormer
- **Input**: `(batch_size, channels, height, width)`
- **Output**: `(batch_size, num_classes, height, width)`
- **Typical shapes**: `(2, 3, 256, 256)` → `(2, 3, 256, 256)`

### TransUNet
- **Input**: `(batch_size, channels, height, width)`
- **Output**: `(batch_size, num_classes, height, width)`
- **Typical shapes**: `(2, 1, 224, 224)` → `(2, 3, 224, 224)`

## Dependencies

Tests require the following Python packages:
- `torch`: PyTorch for tensor operations and model instantiation
- `numpy`: For numerical operations
- `transformers`: For SegFormer model (HuggingFace)
- `einops`: For tensor reshaping operations
- `timm`: For D-Former utilities

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in the Python path
2. **CUDA Memory Errors**: Tests use CPU by default, but large models may need GPU
3. **Model Loading Errors**: Some models require pre-trained weights or specific configurations

### Debug Mode

To run tests with more verbose output, modify the test functions to include additional print statements or use Python's logging module.

## Adding New Model Tests

To add tests for a new transformer model:

1. Create a test function following the naming convention `test_<model_name>()`
2. Implement model instantiation, input creation, forward pass, and output validation
3. Add the test function to `run_model_tests.py`
4. Update this README with information about the new model 