# Pipeline Modules

This directory contains high-level pipeline orchestrators that coordinate the different stages of the medical image segmentation workflow. Each pipeline provides a complete end-to-end solution for specific tasks.

## Overview

The pipeline modules serve as the main entry points for different stages of the segmentation workflow:
- **Data Loading**: Loading preprocessed data for training/prediction
- **Data Preprocessing**: Complete preprocessing pipeline for raw datasets
- **Model Training**: End-to-end training pipeline
- **Model Prediction**: Complete prediction pipeline
- **Model Evaluation**: Comprehensive evaluation pipeline

## Pipeline Components

### 1. Data Loading Pipeline (`data_loading_pipeline.py`)

Handles loading of preprocessed data for training, validation, and prediction.

**Key Features:**
- Automatic subject discovery in data directories
- TorchIO-based data loading with proper transforms
- Support for train/val/test splits
- Configurable batch sizes and workers

**Usage:**
```python
from pipelines.data_loading_pipeline import get_dataloader

# Create data loaders for different splits
train_loader = get_dataloader(
    data_dir="data_preprocessed",
    split="train",
    batch_size=1,
    num_workers=4,
    shuffle=True
)

val_loader = get_dataloader(
    data_dir="data_preprocessed",
    split="val",
    batch_size=1,
    num_workers=4,
    shuffle=False
)
```

**Functions:**
- `get_subjects_list(split_dir)`: Extract subject list from directory
- `get_dataloader(data_dir, split, batch_size, num_workers, shuffle)`: Create data loader

### 2. Data Preprocessing Pipeline (`data_preprocessing_pipeline.py`)

Complete preprocessing pipeline for spatial and intensity transformations.

**Key Features:**
- Spatial transformations (resampling, resizing)
- Intensity normalization (z-score, min-max, robust)
- Data augmentation for training
- Metadata extraction and preservation
- Support for different dataset formats

**Usage:**
```python
from pipelines.data_preprocessing_pipeline import CTPreprocessingPipeline
from configs.data_preprocessing_config import PreprocessingConfig

# Create preprocessing pipeline
config = PreprocessingConfig()
pipeline = CTPreprocessingPipeline(config)

# Create transforms
transforms = pipeline.create_transforms(is_training=True)

# Process subject
processed_subject, metadata = pipeline.process_subject(subject_info, transforms)
```

**Methods:**
- `create_transforms(is_training)`: Create TorchIO transform pipeline
- `extract_metadata(subject)`: Extract subject metadata
- `process_subject(subject_info, transform)`: Process single subject
- `create_combined_mask(liver_mask, tumor_mask)`: Combine separate masks

### 3. Model Training Pipeline (`model_training_pipeline.py`)

Complete end-to-end training pipeline for segmentation models.

**Key Features:**
- Model registry for different architectures
- Automatic model instantiation
- Data loader setup
- Training and testing workflows
- Reproducible training with seed setting

**Usage:**
```python
from pipelines.model_training_pipeline import run_training_pipeline
from configs.model_training_config import ModelTrainingConfig

# Setup configuration
config = ModelTrainingConfig(
    model_name="dformer3d",
    data_dir="data_preprocessed",
    max_epochs=100
)

# Run training pipeline
run_training_pipeline(config)
```

**Functions:**
- `get_model_class(model_name)`: Get model class from registry
- `run_training_pipeline(config)`: Complete training workflow

**Supported Models:**
- `dformer3d`: DFormer3D architecture
- `segformer`: SegFormer architecture
- Extensible for additional models

### 4. Model Prediction Pipeline (`model_prediction_pipeline.py`)

Complete prediction pipeline for model inference.

**Key Features:**
- Model loading from checkpoints
- Data loader setup for prediction
- Batch prediction processing
- Result saving in NIfTI format

**Usage:**
```python
from pipelines.model_prediction_pipeline import run_prediction_pipeline
from configs.model_prediction_config import ModelPredictionConfig

# Setup configuration
config = ModelPredictionConfig(
    model_name="dformer3d",
    checkpoint_path="checkpoints/best.ckpt",
    input_dir="data_preprocessed/test",
    output_dir="predictions"
)

# Run prediction pipeline
run_prediction_pipeline(config)
```

**Functions:**
- `get_model_class(model_name)`: Get model class from registry
- `run_prediction_pipeline(config)`: Complete prediction workflow

### 5. Model Evaluation Pipeline (`model_evaluation_pipeline.py`)

Comprehensive evaluation pipeline for model performance assessment.

**Key Features:**
- Automatic evaluation of predictions
- Multiple metric computation
- Result saving and reporting
- Statistical analysis support

**Usage:**
```python
from pipelines.model_evaluation_pipeline import run_evaluation_pipeline
from configs.model_evaluation_config import ModelEvaluationConfig

# Setup configuration
config = ModelEvaluationConfig(
    pred_dir="predictions",
    gt_dir="data_preprocessed/test",
    save_csv=True,
    print_summary=True
)

# Run evaluation pipeline
run_evaluation_pipeline(config)
```

**Functions:**
- `run_evaluation_pipeline(config)`: Complete evaluation workflow

## Pipeline Workflows

### Complete Training Workflow

```python
from configs.model_training_config import ModelTrainingConfig
from pipelines.model_training_pipeline import run_training_pipeline

# 1. Setup training configuration
config = ModelTrainingConfig(
    model_name="dformer3d",
    data_dir="data_preprocessed",
    max_epochs=100,
    learning_rate=1e-4,
    optimizer="adamw",
    gpus=1
)

# 2. Run training pipeline
run_training_pipeline(config)
```

### Complete Prediction Workflow

```python
from configs.model_prediction_config import ModelPredictionConfig
from pipelines.model_prediction_pipeline import run_prediction_pipeline

# 1. Setup prediction configuration
config = ModelPredictionConfig(
    model_name="dformer3d",
    checkpoint_path="checkpoints/best.ckpt",
    input_dir="data_preprocessed/test",
    output_dir="predictions",
    batch_size=1
)

# 2. Run prediction pipeline
run_prediction_pipeline(config)
```

### Complete Evaluation Workflow

```python
from configs.model_evaluation_config import ModelEvaluationConfig
from pipelines.model_evaluation_pipeline import run_evaluation_pipeline

# 1. Setup evaluation configuration
config = ModelEvaluationConfig(
    pred_dir="predictions",
    gt_dir="data_preprocessed/test",
    metrics=["dice", "iou", "bf1", "accuracy"],
    save_csv=True,
    csv_path="evaluation_results.csv"
)

# 2. Run evaluation pipeline
run_evaluation_pipeline(config)
```

## Advanced Features

### Model Registry

The training and prediction pipelines use a model registry for extensibility:

```python
# Model registry in training pipeline
registry = {
    'dformer3d': 'models.transformers.d_former.network.SegNetwork',
    'segformer': SegFormerModule,
}

# Adding new models
registry['unet3d'] = 'models.cnns.unet_3d.UNet3D'
```

### Configuration Management

All pipelines use configuration classes for parameter management:

```python
# Training configuration
config = ModelTrainingConfig(
    model_name="dformer3d",
    model_params={"hidden_size": 768, "num_heads": 12},
    data_dir="data_preprocessed",
    max_epochs=100
)

# Prediction configuration
config = ModelPredictionConfig(
    model_name="dformer3d",
    checkpoint_path="checkpoints/best.ckpt",
    target_label=2  # Binary segmentation
)
```

### Error Handling

Pipelines include comprehensive error handling:

```python
# Training pipeline error handling
try:
    run_training_pipeline(config)
except Exception as e:
    logger.error(f"Training failed: {e}")
    # Handle error appropriately
```

## Integration with Other Modules

### Data Preprocessing Integration

```python
# Preprocessing pipeline integration
from data_preprocessing.data_preprocessor import DataPreprocessor
from pipelines.data_preprocessing_pipeline import CTPreprocessingPipeline

# Use preprocessing pipeline in data preprocessor
pipeline = CTPreprocessingPipeline(config)
preprocessor = DataPreprocessor(config, pipeline=pipeline)
```

### Model Integration

```python
# Model integration in training pipeline
from models.transformers.d_former.network import SegNetwork
from model_training.lightning_module import SegmentationLightningModule

# Model instantiation
model = SegNetwork(num_classes=config.num_classes, in_chan=config.input_channels)
lightning_module = SegmentationLightningModule(model, config)
```

### Evaluation Integration

```python
# Evaluation integration
from model_evaluation.evaluator import Evaluator

# Evaluator usage in evaluation pipeline
evaluator = Evaluator(config)
evaluator.evaluate()
evaluator.save_results()
```

## Best Practices

### Pipeline Design

1. **Modularity**: Each pipeline focuses on a specific task
2. **Configuration**: Use configuration classes for parameter management
3. **Error Handling**: Include comprehensive error handling
4. **Logging**: Provide detailed logging for debugging
5. **Extensibility**: Design for easy addition of new models/features

### Usage Guidelines

1. **Configuration First**: Always start with proper configuration setup
2. **Data Validation**: Ensure data is in correct format before processing
3. **Resource Management**: Monitor memory and GPU usage
4. **Result Validation**: Verify pipeline outputs are as expected
5. **Documentation**: Document any custom modifications

### Performance Optimization

1. **Batch Processing**: Use appropriate batch sizes for your hardware
2. **Parallel Processing**: Utilize multiple workers for data loading
3. **Memory Management**: Monitor and optimize memory usage
4. **GPU Utilization**: Ensure efficient GPU usage during training

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model architecture matches checkpoint
2. **Data Loading Issues**: Verify data directory structure and file formats
3. **Memory Problems**: Reduce batch size or use gradient checkpointing
4. **Configuration Errors**: Validate configuration parameters

### Debug Mode

Enable debug mode for detailed logging:

```python
config.debug = True
```

This provides additional information about:
- Pipeline execution steps
- Data shapes and types
- Model loading process
- Error details

## Example Scripts

### Complete Workflow Script

```python
#!/usr/bin/env python3
"""
Complete segmentation workflow script.
"""

import os
from configs.model_training_config import ModelTrainingConfig
from configs.model_prediction_config import ModelPredictionConfig
from configs.model_evaluation_config import ModelEvaluationConfig
from pipelines.model_training_pipeline import run_training_pipeline
from pipelines.model_prediction_pipeline import run_prediction_pipeline
from pipelines.model_evaluation_pipeline import run_evaluation_pipeline

def main():
    # 1. Training
    print("Starting training...")
    train_config = ModelTrainingConfig(
        model_name="dformer3d",
        data_dir="data_preprocessed",
        max_epochs=100
    )
    run_training_pipeline(train_config)
    
    # 2. Prediction
    print("Starting prediction...")
    pred_config = ModelPredictionConfig(
        model_name="dformer3d",
        checkpoint_path="checkpoints/best.ckpt",
        input_dir="data_preprocessed/test",
        output_dir="predictions"
    )
    run_prediction_pipeline(pred_config)
    
    # 3. Evaluation
    print("Starting evaluation...")
    eval_config = ModelEvaluationConfig(
        pred_dir="predictions",
        gt_dir="data_preprocessed/test",
        save_csv=True,
        print_summary=True
    )
    run_evaluation_pipeline(eval_config)
    
    print("Workflow complete!")

if __name__ == "__main__":
    main()
```

The pipeline modules provide a complete, integrated solution for medical image segmentation workflows with comprehensive error handling, configuration management, and extensibility. 