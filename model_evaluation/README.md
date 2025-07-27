# Model Evaluation Module

This module provides comprehensive evaluation capabilities for medical image segmentation models. It implements multiple evaluation metrics, supports different tissue types, and provides detailed analysis of model performance.

## Overview

The evaluation module is designed to assess segmentation model performance using standard medical imaging metrics. It supports evaluation of different tissue types (liver, tumor, whole region) and provides both individual subject and aggregate performance analysis.

## Components

### 1. `Evaluator` (`evaluator.py`)

The main evaluation class that orchestrates the entire evaluation pipeline.

**Key Features:**
- Automatic loading of predictions and ground truth
- Multi-tissue evaluation (liver, tumor, whole region)
- Comprehensive metric computation
- CSV result export
- Console summary reporting

**Usage:**
```python
from model_evaluation.evaluator import Evaluator
from configs.model_evaluation_config import ModelEvaluationConfig

# Create evaluator
config = ModelEvaluationConfig(
    pred_dir="predictions",
    gt_dir="data_preprocessed/test"
)
evaluator = Evaluator(config)

# Run evaluation
evaluator.evaluate()
evaluator.save_results()
evaluator.print_summary()
```

**Methods:**
- `__init__(config)`: Initialize evaluator with configuration
- `evaluate_subject(pred_path, gt_path, subject_id)`: Evaluate single subject
- `evaluate()`: Evaluate all subjects in prediction directory
- `save_results()`: Save results to CSV file
- `print_summary()`: Print formatted summary to console

### 2. Evaluation Metrics (`metrics.py`)

NumPy-based implementations of segmentation evaluation metrics.

**Available Metrics:**
- **Dice Coefficient**: Measures overlap between predictions and ground truth
- **Jaccard Index (IoU)**: Intersection over Union metric
- **Accuracy**: Pixel-wise accuracy
- **Boundary F1 Score**: Boundary accuracy with tolerance

**Usage:**
```python
from model_evaluation.metrics import dice_coefficient, jaccard_index, accuracy

# Compute metrics for specific label
dice = dice_coefficient(pred, gt, label=1)  # Liver
iou = jaccard_index(pred, gt, label=2)      # Tumor
acc = accuracy(pred, gt, label=1)           # Liver accuracy
```

**Mathematical Formulations:**
- **Dice**: `2 * |A ∩ B| / (|A| + |B|)`
- **IoU**: `|A ∩ B| / |A ∪ B|`
- **Boundary F1**: F1 score on boundary pixels with tolerance

## Evaluation Pipeline

### 1. Setup Configuration

```python
from configs.model_evaluation_config import ModelEvaluationConfig

config = ModelEvaluationConfig(
    pred_dir="predictions",
    gt_dir="data_preprocessed/test",
    liver_label=1,
    tumor_label=2,
    metrics=["dice", "iou", "bf1", "accuracy"],
    save_csv=True,
    csv_path="evaluation_results.csv"
)
```

### 2. Create Evaluator

```python
from model_evaluation.evaluator import Evaluator

evaluator = Evaluator(config)
```

### 3. Run Evaluation

```python
# Evaluate all subjects
evaluator.evaluate()

# Save results
evaluator.save_results()

# Print summary
evaluator.print_summary()
```

## Evaluation Metrics

### Dice Coefficient

Measures the overlap between predicted and ground truth segmentations.

**Formula:** `2 * |A ∩ B| / (|A| + |B|)`

**Range:** 0 to 1 (higher is better)

**Usage:**
```python
dice = dice_coefficient(pred, gt, label=1, smooth=1e-5)
```

### Jaccard Index (IoU)

Intersection over Union metric for segmentation evaluation.

**Formula:** `|A ∩ B| / |A ∪ B|`

**Range:** 0 to 1 (higher is better)

**Usage:**
```python
iou = jaccard_index(pred, gt, label=2, smooth=1e-5)
```

### Accuracy

Pixel-wise accuracy across the entire volume.

**Formula:** `(Correct Pixels) / (Total Pixels)`

**Range:** 0 to 1 (higher is better)

**Usage:**
```python
acc = accuracy(pred, gt, label=1)
```

### Boundary F1 Score

Measures accuracy of segmentation boundaries using distance transforms.

**Features:**
- Tolerance-based boundary matching
- Distance transform computation
- F1 score on boundary pixels

**Usage:**
```python
bf1 = boundary_f1_score(pred, gt, label=2, tolerance=1)
```

## Tissue-Specific Evaluation

### Liver Evaluation

Evaluate liver segmentation performance:

```python
# Liver metrics
liver_dice = dice_coefficient(pred, gt, label=config.liver_label)
liver_iou = jaccard_index(pred, gt, label=config.liver_label)
liver_accuracy = accuracy(pred, gt, label=config.liver_label)
liver_bf1 = boundary_f1_score(pred, gt, label=config.liver_label)
```

### Tumor Evaluation

Evaluate tumor segmentation performance:

```python
# Tumor metrics
tumor_dice = dice_coefficient(pred, gt, label=config.tumor_label)
tumor_iou = jaccard_index(pred, gt, label=config.tumor_label)
tumor_accuracy = accuracy(pred, gt, label=config.tumor_label)
tumor_bf1 = boundary_f1_score(pred, gt, label=config.tumor_label)
```

### Whole Region Evaluation

Evaluate combined liver and tumor segmentation:

```python
# Create combined mask
mask_pred = (pred == config.liver_label) | (pred == config.tumor_label)
mask_gt = (gt == config.liver_label) | (gt == config.tumor_label)

# Whole region metrics
whole_dice = dice_coefficient(mask_pred, mask_gt, label=1)
whole_iou = jaccard_index(mask_pred, mask_gt, label=1)
whole_accuracy = accuracy(mask_pred, mask_gt, label=1)
whole_bf1 = boundary_f1_score(mask_pred, mask_gt, label=1)
```

## Output Formats

### CSV Results

Evaluation results are saved in CSV format with columns:

Example:
```csv
subject_id,dice_liver,iou_liver,bf1_liver,accuracy_liver,dice_tumor,iou_tumor,bf1_tumor,accuracy_tumor,dice_whole,iou_whole,bf1_whole,accuracy_whole
1A,0.9234,0.8567,0.7891,0.9876,0.8123,0.6789,0.6543,0.9234,0.9456,0.8978,0.8234,0.9876
2A,0.9345,0.8789,0.8123,0.9890,0.8234,0.6890,0.6654,0.9345,0.9567,0.9089,0.8345,0.9890
```

### Console Summary

Formatted summary printed to console:

```
===== Evaluation Summary =====

--- Liver ---
Dice: 0.9234
IoU: 0.8567
BF1: 0.7891
Accuracy: 0.9876

--- Tumor ---
Dice: 0.8123
IoU: 0.6789
BF1: 0.6543
Accuracy: 0.9234

--- Whole ---
Dice: 0.9456
IoU: 0.8978
BF1: 0.8234
Accuracy: 0.9876

=============================
```

## Advanced Features

### Selective Evaluation

Evaluate specific subjects only:

```python
config.subject_ids = ["1A", "2B", "3C"]
evaluator.evaluate()  # Only evaluates specified subjects
```

### Custom Metrics

Add custom metrics by extending the evaluator:

```python
def evaluate_subject(self, pred_path, gt_path, subject_id):
    # ... existing code ...
    
    # Add custom metric
    custom_metric = self.compute_custom_metric(pred, gt)
    metrics["custom_metric"] = custom_metric
    
    return metrics
```

### Batch Processing

Process multiple subjects efficiently:

```python
# The evaluator automatically processes all subjects in the prediction directory
evaluator.evaluate()  # Processes all subjects
```

## Performance Analysis

### Statistical Analysis

Analyze evaluation results:

```python
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("evaluation_results.csv")

# Statistical summary
print("Liver Dice Statistics:")
print(f"Mean: {df['dice_liver'].mean():.4f}")
print(f"Std: {df['dice_liver'].std():.4f}")
print(f"Min: {df['dice_liver'].min():.4f}")
print(f"Max: {df['dice_liver'].max():.4f}")

# Correlation analysis
correlation = df['dice_liver'].corr(df['dice_tumor'])
print(f"Liver-Tumor Dice Correlation: {correlation:.4f}")
```

### Visualization

Create visualizations of results:

```python
import matplotlib.pyplot as plt

# Box plot of metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics = ['dice_liver', 'dice_tumor', 'iou_liver', 'iou_tumor']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    df[metric].plot(kind='box', ax=ax)
    ax.set_title(metric.replace('_', ' ').title())
    ax.set_ylabel('Score')

plt.tight_layout()
plt.show()
```

## Error Handling

### Common Issues

1. **Missing Files**: Evaluator handles missing ground truth files gracefully
2. **Label Mismatch**: Ensure label values match between predictions and ground truth
3. **File Format**: Ensure predictions and ground truth are in NIfTI format

### Debug Mode

Enable debug mode for additional information:

```python
config.debug = True
```

This provides detailed logging about:
- File loading process
- Metric computation
- Error handling
- Result processing

## Best Practices

1. **Validate Predictions**: Ensure predictions are in correct format and range
2. **Check Ground Truth**: Verify ground truth labels match expected values
3. **Use Appropriate Metrics**: Choose metrics based on clinical requirements
4. **Document Results**: Save evaluation results and configurations
5. **Statistical Analysis**: Perform statistical analysis for robust conclusions

## Example Workflow

### Complete Evaluation Script

```python
import os
from configs.model_evaluation_config import ModelEvaluationConfig
from model_evaluation.evaluator import Evaluator

# Configuration
config = ModelEvaluationConfig(
    pred_dir="predictions",
    gt_dir="data_preprocessed/test",
    liver_label=1,
    tumor_label=2,
    metrics=["dice", "iou", "bf1", "accuracy"],
    save_csv=True,
    csv_path="evaluation_results.csv",
    print_summary=True
)

# Create evaluator
evaluator = Evaluator(config)

# Run evaluation
print("Starting evaluation...")
evaluator.evaluate()

# Save and display results
evaluator.save_results()
evaluator.print_summary()

print("Evaluation complete!")
```

### Command Line Usage

```bash
# Run evaluation using the main script
python run_evaluation.py \
    --pred_dir predictions \
    --gt_dir data_preprocessed/test \
    --save_csv \
    --print_summary
```

This module provides comprehensive evaluation capabilities for segmentation models with support for multiple metrics, tissue types, and output formats. 