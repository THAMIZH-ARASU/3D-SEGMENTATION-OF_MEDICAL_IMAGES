# Model Prediction Module

This module provides a complete prediction/inference pipeline for medical image segmentation models. It handles model loading, batch prediction, post-processing, and saving of segmentation results in NIfTI format.

## Overview

The prediction module is designed for efficient inference on preprocessed medical images. It supports both multi-class and binary segmentation tasks with automatic label mapping and result saving.

## Components

### 1. `Predictor` (`predictor.py`)

The main prediction class that orchestrates the entire inference pipeline.

**Key Features:**
- Model loading from PyTorch Lightning checkpoints
- Batch processing of preprocessed images
- Automatic post-processing and label mapping
- NIfTI format output saving
- Support for both multi-class and binary segmentation

**Usage:**
```python
from model_prediction.predictor import Predictor
from model_training.lightning_module import SegmentationLightningModule
from models.transformers.d_former.network import SegNetwork

# Create predictor
predictor = Predictor(
    lightning_module_class=SegmentationLightningModule,
    config=config,
    model_class=SegNetwork
)

# Run prediction
predictor.predict(dataloader, subject_list)
```

**Methods:**
- `__init__(lightning_module_class, config, model_class)`: Initialize predictor
- `predict(dataloader, subject_list)`: Run prediction on all subjects

### 2. `get_prediction_loader` (`data.py`)

Utility function for creating data loaders specifically for prediction tasks.

**Key Features:**
- Automatic detection of preprocessed image files
- TorchIO-based data loading
- Support for batch processing
- Subject metadata extraction

**Usage:**
```python
from model_prediction.data import get_prediction_loader

# Create prediction data loader
loader, subjects = get_prediction_loader(
    input_dir="data_preprocessed/test",
    batch_size=1,
    num_workers=2
)
```

**Returns:**
- `loader`: TorchIO SubjectsLoader for batch processing
- `subjects`: List of subject metadata dictionaries

## Prediction Pipeline

### 1. Setup Configuration

```python
from configs.model_prediction_config import ModelPredictionConfig

config = ModelPredictionConfig(
    model_name="dformer3d",
    checkpoint_path="checkpoints/best.ckpt",
    input_dir="data_preprocessed/test",
    output_dir="predictions",
    batch_size=1,
    device="cuda"
)
```

### 2. Create Data Loader

```python
from model_prediction.data import get_prediction_loader

loader, subjects = get_prediction_loader(
    input_dir=config.input_dir,
    batch_size=config.batch_size,
    num_workers=config.num_workers
)
```

### 3. Run Prediction

```python
from model_prediction.predictor import Predictor
from model_training.lightning_module import SegmentationLightningModule
from models.transformers.d_former.network import SegNetwork

# Create predictor
predictor = Predictor(
    lightning_module_class=SegmentationLightningModule,
    config=config,
    model_class=SegNetwork
)

# Run prediction
predictor.predict(loader, subjects)
```

## Advanced Features

### Binary Segmentation Support

For binary segmentation tasks (e.g., tumor-only segmentation):

```python
config.target_label = 2  # Tumor label
```

The predictor automatically:
- Selects only background and target label channels from model output
- Maps binary predictions (0, 1) back to original label values
- Saves results with proper label mapping

### Multi-Class Segmentation

For multi-class segmentation (e.g., background, liver, tumor):

```python
config.num_classes = 3
config.target_label = None  # Disable binary mode
```

The predictor:
- Uses all output channels from the model
- Saves predictions with original class labels
- Maintains full multi-class information

### Device Management

Configure device usage for prediction:

```python
# GPU prediction
config.device = "cuda"
config.gpus = 1

# CPU prediction
config.device = "cpu"
config.gpus = 0
```

### Batch Processing

Configure batch size for memory-efficient prediction:

```python
# Single sample processing (recommended for large volumes)
config.batch_size = 1

# Batch processing (if memory allows)
config.batch_size = 4
```

## Output Format

### File Naming Convention

Predictions are saved with the naming convention:
```
{subject_id}_pred.nii.gz
```

Example:
```
1A_pred.nii.gz
2B_pred.nii.gz
liver_001_pred.nii.gz
```

### NIfTI Format

All predictions are saved in NIfTI format with:
- Proper affine matrices from input images
- Correct data types (uint8 for labels)
- Preserved spatial information

### Label Values

- **Multi-class**: Original class labels (0, 1, 2, ...)
- **Binary**: Mapped to target label values (0, target_label)

## Integration with Training

### Loading Trained Models

The predictor seamlessly loads models trained with the training module:

```python
# Load from checkpoint
config.checkpoint_path = "checkpoints/best.ckpt"

# Use same model configuration as training
config.model_name = "dformer3d"
config.num_classes = 3
config.input_channels = 1
```

### Consistent Configuration

Ensure prediction configuration matches training configuration:
- Same model architecture
- Same number of classes
- Same input channels
- Compatible model parameters

## Performance Optimization

### Memory Management

1. **Use Appropriate Batch Size**: Start with batch size 1 for 3D volumes
2. **Monitor GPU Memory**: Use `nvidia-smi` to monitor memory usage
3. **Gradient Disabled**: Prediction runs with `torch.no_grad()` for efficiency

### Processing Speed

1. **GPU Acceleration**: Use CUDA for faster inference
2. **Data Loading**: Adjust `num_workers` based on system capabilities
3. **Batch Processing**: Increase batch size if memory allows

### Parallel Processing

For large datasets, consider parallel processing:

```python
# Process multiple subjects in parallel
import multiprocessing as mp

def process_subject(subject_id):
    # Individual subject processing
    pass

with mp.Pool(processes=4) as pool:
    results = pool.map(process_subject, subject_ids)
```

## Error Handling

### Common Issues

1. **Checkpoint Not Found**: Ensure checkpoint path is correct
2. **Model Architecture Mismatch**: Verify model configuration matches training
3. **Memory Issues**: Reduce batch size or use CPU
4. **Data Format Issues**: Ensure input data is properly preprocessed

### Debug Mode

Enable debug mode for additional information:

```python
config.debug = True
```

This provides detailed logging about:
- Model loading process
- Data shapes and types
- Prediction steps
- File saving operations

## Best Practices

1. **Validate Input Data**: Ensure preprocessed images are in correct format
2. **Check Model Compatibility**: Verify model architecture matches checkpoint
3. **Monitor Output Quality**: Validate prediction results visually
4. **Backup Results**: Keep copies of important predictions
5. **Document Configuration**: Save prediction configuration for reproducibility

## Example Workflow

### Complete Prediction Script

```python
import os
from configs.model_prediction_config import ModelPredictionConfig
from model_prediction.data import get_prediction_loader
from model_prediction.predictor import Predictor
from model_training.lightning_module import SegmentationLightningModule
from models.transformers.d_former.network import SegNetwork

# Configuration
config = ModelPredictionConfig(
    model_name="dformer3d",
    checkpoint_path="checkpoints/best.ckpt",
    input_dir="data_preprocessed/test",
    output_dir="predictions",
    batch_size=1,
    device="cuda"
)

# Create data loader
loader, subjects = get_prediction_loader(
    input_dir=config.input_dir,
    batch_size=config.batch_size,
    num_workers=config.num_workers
)

# Create predictor
predictor = Predictor(
    lightning_module_class=SegmentationLightningModule,
    config=config,
    model_class=SegNetwork
)

# Run prediction
print(f"Processing {len(subjects)} subjects...")
predictor.predict(loader, subjects)
print("Prediction complete!")
```

### Command Line Usage

```bash
# Run prediction using the main script
python run_prediction.py \
    --model_name dformer3d \
    --checkpoint_path checkpoints/best.ckpt \
    --input_dir data_preprocessed/test \
    --output_dir predictions \
    --batch_size 1
```

This module provides a robust and efficient way to run inference on trained segmentation models with comprehensive support for different segmentation tasks and output formats. 