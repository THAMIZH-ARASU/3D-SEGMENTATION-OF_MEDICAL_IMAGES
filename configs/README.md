# Configuration Module

This module contains configuration classes for all major components of the CT scan segmentation pipeline. Each configuration class is implemented using Python's `dataclass` decorator for type safety and ease of use.

## Overview

The configuration system provides a centralized way to manage parameters for:
- Data preprocessing and augmentation
- Model training and optimization
- Model prediction and inference
- Model evaluation and metrics

## Configuration Classes

### 1. `PreprocessingConfig` (`data_preprocessing_config.py`)

Configuration for data preprocessing pipeline parameters.

**Key Parameters:**
- **Spatial Parameters:**
  - `target_spacing`: Target voxel spacing in mm (default: `[1.0, 1.0, 1.0]`)
  - `target_size`: Target image size in voxels (default: `[256, 256, 256]`)

- **Intensity Parameters:**
  - `intensity_range`: Intensity clipping range in HU units (default: `[-100, 400]`)
  - `normalize_method`: Normalization method (`"zscore"`, `"minmax"`, `"robust"`)

- **Augmentation Parameters:**
  - `apply_augmentation`: Enable/disable augmentation (default: `True`)
  - `rotation_degrees`: Maximum rotation angle (default: `10.0`)
  - `translation_range`: Maximum translation range (default: `10.0`)
  - `elastic_deformation`: Enable elastic deformation (default: `False`)

- **Dataset Split:**
  - `train_ratio`: Training set proportion (default: `0.7`)
  - `val_ratio`: Validation set proportion (default: `0.2`)
  - `test_ratio`: Test set proportion (default: `0.1`)

**Usage:**
```python
from configs.data_preprocessing_config import PreprocessingConfig

config = PreprocessingConfig(
    target_spacing=(1.0, 1.0, 1.0),
    target_size=(256, 256, 256),
    normalize_method="zscore"
)
```

### 2. `ModelTrainingConfig` (`model_training_config.py`)

Configuration for model training parameters.

**Key Parameters:**
- **Model Configuration:**
  - `model_name`: Model architecture (`"dformer3d"`, `"segformer"`, etc.)
  - `model_params`: Additional model-specific parameters
  - `input_channels`: Number of input channels (default: `1`)
  - `num_classes`: Number of output classes (default: `14`)

- **Data Configuration:**
  - `data_dir`: Directory containing preprocessed data
  - `train_batch_size`: Training batch size (default: `1`)
  - `val_batch_size`: Validation batch size (default: `1`)
  - `num_workers`: Number of data loading workers (default: `4`)

- **Training Parameters:**
  - `max_epochs`: Maximum training epochs (default: `100`)
  - `gpus`: Number of GPUs to use (default: `1`)
  - `precision`: Training precision (`"16-mixed"`, `"32"`) (default: `"16-mixed"`)
  - `seed`: Random seed for reproducibility (default: `42`)

- **Optimizer Configuration:**
  - `optimizer`: Optimizer type (`"adam"`, `"adamw"`, `"sgd"`) (default: `"adam"`)
  - `learning_rate`: Learning rate (default: `1e-4`)
  - `weight_decay`: Weight decay (default: `1e-5`)
  - `scheduler`: Learning rate scheduler (`"step"`, `"plateau"`) (default: `None`)

- **Monitoring:**
  - `monitor_metric`: Metric to monitor for checkpointing (default: `"val_loss"`)
  - `monitor_mode`: Monitoring mode (`"min"` or `"max"`) (default: `"min"`)
  - `early_stopping_patience`: Early stopping patience (default: `30`)

**Usage:**
```python
from configs.model_training_config import ModelTrainingConfig

config = ModelTrainingConfig(
    model_name="dformer3d",
    max_epochs=200,
    learning_rate=1e-4,
    optimizer="adamw"
)
```

### 3. `ModelPredictionConfig` (`model_prediction_config.py`)

Configuration for model prediction and inference.

**Key Parameters:**
- **Model Configuration:**
  - `model_name`: Model architecture to use
  - `checkpoint_path`: Path to trained model checkpoint
  - `input_channels`: Number of input channels (default: `1`)
  - `num_classes`: Number of output classes (default: `14`)

- **Data Configuration:**
  - `input_dir`: Directory containing input images
  - `output_dir`: Directory to save predictions
  - `batch_size`: Prediction batch size (default: `1`)
  - `num_workers`: Number of data loading workers (default: `2`)

- **Binary Segmentation:**
  - `target_label`: Target label for binary segmentation (default: `None`)

- **Device Configuration:**
  - `gpus`: Number of GPUs to use (default: `1`)
  - `device`: Device to use (`"cpu"` or `"cuda"`) (default: `"cuda"`)

**Usage:**
```python
from configs.model_prediction_config import ModelPredictionConfig

config = ModelPredictionConfig(
    model_name="dformer3d",
    checkpoint_path="checkpoints/best.ckpt",
    input_dir="data_preprocessed/test",
    output_dir="predictions"
)
```

### 4. `ModelEvaluationConfig` (`model_evaluation_config.py`)

Configuration for model evaluation and metrics computation.

**Key Parameters:**
- **Data Paths:**
  - `pred_dir`: Directory containing prediction files
  - `gt_dir`: Directory containing ground truth files
  - `pred_suffix`: Suffix for prediction files (default: `"_pred.nii.gz"`)
  - `gt_suffix`: Suffix for ground truth files (default: `"_label.nii.gz"`)

- **Label Configuration:**
  - `liver_label`: Label value for liver tissue (default: `1`)
  - `tumor_label`: Label value for tumor tissue (default: `2`)
  - `background_label`: Label value for background (default: `0`)
  - `num_classes`: Total number of classes (default: `3`)

- **Evaluation Metrics:**
  - `metrics`: List of metrics to compute (default: `["dice", "iou", "bf1", "accuracy"]`)

- **Output Configuration:**
  - `save_csv`: Save results to CSV (default: `True`)
  - `csv_path`: Path for CSV output (default: `"evaluation_results.csv"`)
  - `print_summary`: Print summary to console (default: `True`)

**Usage:**
```python
from configs.model_evaluation_config import ModelEvaluationConfig

config = ModelEvaluationConfig(
    pred_dir="predictions",
    gt_dir="data_preprocessed/test",
    metrics=["dice", "iou", "accuracy"]
)
```

## Configuration Management

### Loading and Saving Configurations

Configurations can be easily serialized and deserialized:

```python
import json
from dataclasses import asdict

# Save configuration
config = ModelTrainingConfig()
with open("config.json", "w") as f:
    json.dump(asdict(config), f, indent=2)

# Load configuration
with open("config.json", "r") as f:
    config_dict = json.load(f)
config = ModelTrainingConfig(**config_dict)
```

### Configuration Validation

All configuration classes use type hints and dataclass validation. Invalid parameter types will raise errors during instantiation.

### Default Values

Each configuration class provides sensible defaults for all parameters, making it easy to get started with minimal configuration.

## Best Practices

1. **Use Type Hints**: All configuration parameters are typed for better IDE support and error catching.

2. **Document Changes**: When modifying configurations, document the changes and their rationale.

3. **Version Control**: Keep configuration files in version control to track parameter changes.

4. **Environment-Specific Configs**: Consider creating separate configuration files for different environments (development, testing, production).

5. **Validation**: Validate configurations before using them in production pipelines.

## Integration with Pipelines

These configuration classes are used throughout the pipeline:

- **Data Preprocessing**: `PreprocessingConfig` → `DataPreprocessor`
- **Model Training**: `ModelTrainingConfig` → `SegmentationTrainer`
- **Model Prediction**: `ModelPredictionConfig` → `Predictor`
- **Model Evaluation**: `ModelEvaluationConfig` → `Evaluator`

Each pipeline component expects the appropriate configuration class and uses its parameters to configure behavior. 