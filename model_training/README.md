# Model Training Module

This module provides a complete training pipeline for medical image segmentation models using PyTorch Lightning. It includes training, validation, testing, loss functions, metrics, optimizers, and callbacks specifically designed for segmentation tasks.

## Overview

The training module is built around PyTorch Lightning for scalable, reproducible training with automatic logging, checkpointing, and distributed training support. It supports both multi-class and binary segmentation tasks.

## Components

### 1. `SegmentationLightningModule` (`lightning_module.py`)

The core PyTorch Lightning module that encapsulates the training logic.

**Key Features:**
- Forward pass through segmentation models
- Loss computation using cross-entropy
- Comprehensive metric calculation (Dice, IoU, Boundary F1, Accuracy)
- Binary segmentation support for specific target labels
- Automatic logging of all metrics during training/validation/testing

**Usage:**
```python
from model_training.lightning_module import SegmentationLightningModule
from configs.model_training_config import ModelTrainingConfig

config = ModelTrainingConfig()
model = YourSegmentationModel()
lightning_module = SegmentationLightningModule(model, config)
```

**Methods:**
- `forward(x)`: Forward pass through the model
- `training_step(batch, batch_idx)`: Training step with loss and metrics
- `validation_step(batch, batch_idx)`: Validation step with metrics
- `test_step(batch, batch_idx)`: Test step with metrics
- `configure_optimizers()`: Setup optimizer and scheduler

### 2. `SegmentationTrainer` (`trainer.py`)

High-level trainer wrapper that configures and manages the PyTorch Lightning Trainer.

**Key Features:**
- Automatic callback setup (checkpointing, early stopping)
- TensorBoard logging configuration
- GPU/CPU device management
- Training and testing workflows

**Usage:**
```python
from model_training.trainer import SegmentationTrainer
from model_training.lightning_module import SegmentationLightningModule

# Create trainer
trainer = SegmentationTrainer(
    config=config,
    model=lightning_module,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# Train the model
trainer.fit()

# Test the model
trainer.test()
```

### 3. `SegmentationCrossEntropyLoss` (`losses.py`)

Cross-entropy loss function specifically designed for segmentation tasks.

**Features:**
- Wraps PyTorch's CrossEntropyLoss
- Optional class weighting for imbalanced datasets
- Supports both 2D and 3D segmentation

**Usage:**
```python
from model_training.losses import SegmentationCrossEntropyLoss

# Standard loss
loss_fn = SegmentationCrossEntropyLoss()

# With class weights for imbalanced data
class_weights = torch.tensor([1.0, 2.0, 3.0])  # Background, Liver, Tumor
loss_fn = SegmentationCrossEntropyLoss(weight=class_weights)
```

### 4. Evaluation Metrics (`metrics.py`)

Comprehensive set of segmentation evaluation metrics implemented in PyTorch.

**Available Metrics:**
- **Dice Coefficient**: Measures overlap between predictions and ground truth
- **Jaccard Index (IoU)**: Intersection over Union metric
- **Accuracy**: Pixel-wise accuracy
- **Boundary F1 Score**: Boundary accuracy with tolerance

**Usage:**
```python
from model_training.metrics import dice_coefficient, jaccard_index, accuracy

# Compute metrics
dice = dice_coefficient(logits, targets, num_classes=3)
iou = jaccard_index(logits, targets, num_classes=3)
acc = accuracy(logits, targets)
```

**Mathematical Formulations:**
- **Dice**: `2 * |A ∩ B| / (|A| + |B|)`
- **IoU**: `|A ∩ B| / |A ∪ B|`
- **Boundary F1**: F1 score on boundary pixels with tolerance

### 5. Optimizer Configuration (`optimizers.py`)

Centralized optimizer and scheduler configuration.

**Supported Optimizers:**
- Adam (adaptive learning rates)
- AdamW (decoupled weight decay)
- SGD (with momentum)

**Supported Schedulers:**
- StepLR (step-based learning rate reduction)
- ReduceLROnPlateau (metric-based reduction)

**Usage:**
```python
from model_training.optimizers import get_optimizer

optimizer, scheduler = get_optimizer(config, model)

# Use in Lightning module
def configure_optimizers(self):
    optimizer, scheduler = get_optimizer(self.config, self)
    if scheduler:
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    return optimizer
```

### 6. Callbacks (`callbacks.py`)

Utility for creating PyTorch Lightning callbacks.

**Available Callbacks:**
- **ModelCheckpoint**: Saves best model based on monitored metric
- **EarlyStopping**: Stops training when metric doesn't improve
- **LearningRateMonitor**: Logs learning rate changes

**Usage:**
```python
from model_training.callbacks import get_callbacks

callbacks = get_callbacks(config)
trainer = pl.Trainer(callbacks=callbacks)
```

## Training Pipeline

### 1. Setup Configuration

```python
from configs.model_training_config import ModelTrainingConfig

config = ModelTrainingConfig(
    model_name="dformer3d",
    data_dir="data_preprocessed",
    max_epochs=100,
    learning_rate=1e-4,
    optimizer="adamw",
    gpus=1
)
```

### 2. Create Model and Lightning Module

```python
from models.transformers.d_former.network import SegNetwork
from model_training.lightning_module import SegmentationLightningModule

# Create model
model = SegNetwork(
    num_classes=config.num_classes,
    in_chan=config.input_channels,
    **config.model_params
)

# Create Lightning module
lightning_module = SegmentationLightningModule(model, config)
```

### 3. Setup Data Loaders

```python
from pipelines.data_loading_pipeline import get_dataloader

train_loader = get_dataloader(
    config.data_dir, 'train', 
    config.train_batch_size, 
    config.num_workers, 
    shuffle=True
)
val_loader = get_dataloader(
    config.data_dir, 'val', 
    config.val_batch_size, 
    config.num_workers, 
    shuffle=False
)
```

### 4. Train the Model

```python
from model_training.trainer import SegmentationTrainer

# Create trainer
trainer = SegmentationTrainer(
    config=config,
    model=lightning_module,
    train_loader=train_loader,
    val_loader=val_loader
)

# Start training
trainer.fit()
```

## Advanced Features

### Binary Segmentation Support

For binary segmentation tasks (e.g., tumor-only segmentation):

```python
config.target_label = 2  # Tumor label
```

The module automatically:
- Selects only background and target label channels
- Maps binary predictions back to original label values
- Computes metrics for binary classification

### Mixed Precision Training

Enable mixed precision for faster training and reduced memory usage:

```python
config.precision = "16-mixed"
```

### Multi-GPU Training

Scale training across multiple GPUs:

```python
config.gpus = 4  # Use 4 GPUs
```

### Custom Metrics

Add custom metrics by extending the Lightning module:

```python
def validation_step(self, batch, batch_idx):
    # ... existing code ...
    
    # Add custom metric
    custom_metric = self.compute_custom_metric(logits, y)
    self.log('val_custom_metric', custom_metric, on_epoch=True)
```

## Monitoring and Logging

### TensorBoard Integration

Training metrics are automatically logged to TensorBoard:

```bash
tensorboard --logdir logs/
```

### Checkpointing

Models are automatically saved based on monitored metrics:
- Best model based on validation loss
- Last checkpoint for resuming training

### Early Stopping

Training automatically stops when validation metric doesn't improve:
- Configurable patience
- Configurable monitoring metric and mode

## Best Practices

1. **Use Appropriate Batch Sizes**: Start with batch size 1 for 3D volumes and increase if memory allows.

2. **Monitor Memory Usage**: Use gradient checkpointing for large models if needed.

3. **Validate Data**: Ensure data loaders provide correct tensor shapes and types.

4. **Use Mixed Precision**: Enable mixed precision for faster training on modern GPUs.

5. **Regular Checkpointing**: Save checkpoints regularly to resume training if interrupted.

6. **Monitor Metrics**: Use TensorBoard to monitor training progress and detect overfitting.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Enable mixed precision or use more GPUs
3. **Poor Convergence**: Adjust learning rate or try different optimizers
4. **Overfitting**: Increase regularization or reduce model complexity

### Debug Mode

Enable debug mode for additional logging:

```python
config.debug = True
```

This will provide more detailed information about data shapes, model outputs, and training progress. 