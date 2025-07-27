# Utility Modules

This directory contains utility functions and tools that support the medical image segmentation pipeline. These utilities provide common functionality for logging, file analysis, and data inspection.

## Overview

The utility modules provide essential support functions for:
- **Logging**: Centralized logging configuration
- **NIfTI Analysis**: Comprehensive NIfTI file information extraction
- **Data Inspection**: Tools for examining data properties and unique values

## Components

### 1. Logging Utilities (`logging_utils.py`)

Centralized logging configuration for consistent logging across the entire pipeline.

**Key Features:**
- Standardized logging format with timestamps
- Configurable log levels
- Stream handler for console output
- Consistent formatting across all modules

**Usage:**
```python
from utils.logging_utils import setup_logging
import logging

# Setup logging with default INFO level
setup_logging()

# Setup logging with custom level
setup_logging(log_level=logging.DEBUG)

# Use logger in your code
logger = logging.getLogger(__name__)
logger.info("Processing started")
logger.debug("Detailed debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

**Function:**
- `setup_logging(log_level=logging.INFO)`: Configure logging with specified level

**Log Format:**
```
2024-01-15 10:30:45,123 [INFO] module_name: Processing started
2024-01-15 10:30:45,124 [DEBUG] module_name: Detailed debug information
2024-01-15 10:30:45,125 [WARNING] module_name: Warning message
2024-01-15 10:30:45,126 [ERROR] module_name: Error occurred
```

### 2. NIfTI File Information (`nifti_file_information.py`)

Comprehensive tool for analyzing NIfTI medical image files.

**Key Features:**
- Complete file information extraction
- Header analysis and metadata display
- Statistical analysis of image data
- Visualization of middle slices
- Error handling for file operations

**Usage:**
```python
from utils.nifti_file_information import analyze_nifti

# Analyze a NIfTI file
analyze_nifti("data_preprocessed/test/1A_image.nii.gz")

# Analyze prediction file
analyze_nifti("predictions/1A_pred.nii.gz")
```

**Function:**
- `analyze_nifti(file_path)`: Comprehensive analysis of NIfTI file

**Output Information:**
- **Basic Information:**
  - File path and existence
  - Image shape and dimensions
  - Data type and voxel sizes
  - Total number of voxels

- **Data Statistics:**
  - Unique values and count
  - Minimum and maximum values
  - Mean and standard deviation
  - Data range analysis

- **Header Information:**
  - Header keys and structure
  - Affine transformation matrix
  - Qform and Sform codes
  - Data scaling parameters

- **Visualization:**
  - Middle slice display (for 3D/4D data)
  - Grayscale visualization with colorbar
  - Automatic slice selection

**Example Output:**
```
=== NIfTI File Information ===
File Path: data_preprocessed/test/1A_image.nii.gz
Shape: (256, 256, 256)
Number of Dimensions: 3
Data Type: float32
Voxel Sizes (mm): (1.0, 1.0, 1.0)
Total Number of Voxels: 16777216
Number of Unique Values: 1024
Unique Values (first 10): [0.0, 0.1, 0.2, ...]
Minimum Value: 0.0
Maximum Value: 1.0
Mean Value: 0.2345
Standard Deviation: 0.3456

=== Header Information ===
Header Keys: ['sizeof_hdr', 'data_type', 'db_name', ...]
Affine Matrix:
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
Qform Code: 1
Sform Code: 1
Data Scaling Slope: 1.0
Data Scaling Intercept: 0.0
```

### 3. Find Unique Values (`find_unique.py`)

Simple utility for extracting unique values from NIfTI files.

**Key Features:**
- Quick unique value extraction
- Simple command-line style usage
- Support for both .nii and .nii.gz formats
- Direct value display

**Usage:**
```python
# Direct usage in script
import nibabel as nib
import numpy as np

# Load NIfTI file
nii_path = "predictions/1A_pred.nii.gz"
nii_img = nib.load(nii_path)
nii_data = nii_img.get_fdata()

# Get unique values
unique_vals = np.unique(nii_data)
print(f"Unique values in {nii_path}:")
print(unique_vals)
```

**Typical Output:**
```
Unique values in predictions/1A_pred.nii.gz:
[0. 1. 2.]
```

## Integration with Pipeline

### Logging Integration

Integrate logging utilities throughout the pipeline:

```python
# In any module
from utils.logging_utils import setup_logging
import logging

# Setup logging at the start of your script
setup_logging(log_level=logging.INFO)

# Use throughout the pipeline
logger = logging.getLogger(__name__)

def process_data():
    logger.info("Starting data processing")
    try:
        # Processing code
        logger.debug("Processing step 1 completed")
        # More processing
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
```

### NIfTI Analysis Integration

Use NIfTI analysis for data validation and debugging:

```python
from utils.nifti_file_information import analyze_nifti

# Validate preprocessed data
def validate_preprocessed_data(data_dir):
    import os
    for file in os.listdir(data_dir):
        if file.endswith('.nii.gz'):
            file_path = os.path.join(data_dir, file)
            print(f"\nAnalyzing {file}:")
            analyze_nifti(file_path)

# Debug prediction results
def debug_predictions(pred_dir):
    import os
    for file in os.listdir(pred_dir):
        if file.endswith('_pred.nii.gz'):
            file_path = os.path.join(pred_dir, file)
            print(f"\nAnalyzing prediction {file}:")
            analyze_nifti(file_path)
```

### Data Inspection Integration

Use unique value analysis for quick data checks:

```python
# Quick check for label values
def check_label_values(label_file):
    import nibabel as nib
    import numpy as np
    
    nii_img = nib.load(label_file)
    nii_data = nii_img.get_fdata()
    unique_vals = np.unique(nii_data)
    
    print(f"Label values in {label_file}: {unique_vals}")
    return unique_vals

# Validate prediction labels
def validate_predictions(pred_dir):
    import os
    for file in os.listdir(pred_dir):
        if file.endswith('_pred.nii.gz'):
            file_path = os.path.join(pred_dir, file)
            check_label_values(file_path)
```

## Advanced Usage

### Custom Logging Configuration

Extend logging utilities for custom needs:

```python
import logging
from utils.logging_utils import setup_logging

# Custom logging setup
def setup_custom_logging(log_file=None, log_level=logging.INFO):
    # Setup basic logging
    setup_logging(log_level=log_level)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)

# Usage
setup_custom_logging(log_file="pipeline.log", log_level=logging.DEBUG)
```

### Batch NIfTI Analysis

Analyze multiple files efficiently:

```python
import os
from utils.nifti_file_information import analyze_nifti

def batch_analyze_nifti(directory, file_pattern="*.nii.gz"):
    """Analyze all NIfTI files in a directory."""
    import glob
    
    files = glob.glob(os.path.join(directory, file_pattern))
    
    for file_path in files:
        print(f"\n{'='*50}")
        print(f"Analyzing: {file_path}")
        print(f"{'='*50}")
        try:
            analyze_nifti(file_path)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

# Usage
batch_analyze_nifti("data_preprocessed/test")
```

### Data Quality Checks

Create comprehensive data quality validation:

```python
from utils.nifti_file_information import analyze_nifti
import nibabel as nib
import numpy as np

def validate_data_quality(data_dir):
    """Comprehensive data quality validation."""
    import os
    
    issues = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.nii.gz'):
            file_path = os.path.join(data_dir, file)
            
            try:
                # Load and analyze
                nii_img = nib.load(file_path)
                data = nii_img.get_fdata()
                
                # Check for common issues
                if np.isnan(data).any():
                    issues.append(f"{file}: Contains NaN values")
                
                if np.isinf(data).any():
                    issues.append(f"{file}: Contains infinite values")
                
                if data.min() < 0 and "image" in file:
                    issues.append(f"{file}: Negative values in image")
                
                # Check label values for label files
                if "label" in file:
                    unique_vals = np.unique(data)
                    if not all(val in [0, 1, 2] for val in unique_vals):
                        issues.append(f"{file}: Unexpected label values {unique_vals}")
                
            except Exception as e:
                issues.append(f"{file}: Error loading file - {e}")
    
    return issues

# Usage
issues = validate_data_quality("data_preprocessed/test")
if issues:
    print("Data quality issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("All files passed quality checks!")
```

## Best Practices

### Logging Best Practices

1. **Consistent Setup**: Use `setup_logging()` at the start of all scripts
2. **Appropriate Levels**: Use DEBUG for detailed info, INFO for general progress, WARNING for issues, ERROR for failures
3. **Contextual Messages**: Include relevant context in log messages
4. **Error Handling**: Always log exceptions with full context

### NIfTI Analysis Best Practices

1. **Regular Validation**: Use NIfTI analysis to validate data at each pipeline stage
2. **Documentation**: Keep analysis results for reproducibility
3. **Error Handling**: Always handle file loading errors gracefully
4. **Performance**: For large datasets, consider batch processing

### Data Inspection Best Practices

1. **Quick Checks**: Use unique value analysis for rapid data validation
2. **Consistent Format**: Ensure consistent file naming and organization
3. **Validation**: Cross-reference results with expected data formats
4. **Documentation**: Document any unexpected findings

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure file paths are correct and files exist
2. **Memory Issues**: For large files, consider loading in chunks
3. **Format Issues**: Ensure files are valid NIfTI format
4. **Permission Issues**: Check file read permissions

### Debug Mode

Enable debug logging for detailed information:

```python
from utils.logging_utils import setup_logging
import logging

setup_logging(log_level=logging.DEBUG)
```

This provides detailed information about:
- File loading operations
- Data processing steps
- Error details and stack traces
- Performance metrics

## Example Scripts

### Complete Data Validation Script

```python
#!/usr/bin/env python3
"""
Complete data validation script using utility modules.
"""

import os
from utils.logging_utils import setup_logging
from utils.nifti_file_information import analyze_nifti
import logging

def main():
    # Setup logging
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate preprocessed data
    logger.info("Validating preprocessed data...")
    data_dir = "data_preprocessed/test"
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.nii.gz'):
                file_path = os.path.join(data_dir, file)
                logger.info(f"Analyzing {file}")
                try:
                    analyze_nifti(file_path)
                except Exception as e:
                    logger.error(f"Error analyzing {file}: {e}")
    else:
        logger.error(f"Data directory {data_dir} not found")
    
    # Validate predictions
    logger.info("Validating predictions...")
    pred_dir = "predictions"
    
    if os.path.exists(pred_dir):
        for file in os.listdir(pred_dir):
            if file.endswith('_pred.nii.gz'):
                file_path = os.path.join(pred_dir, file)
                logger.info(f"Analyzing prediction {file}")
                try:
                    analyze_nifti(file_path)
                except Exception as e:
                    logger.error(f"Error analyzing {file}: {e}")
    else:
        logger.warning(f"Predictions directory {pred_dir} not found")
    
    logger.info("Validation complete!")

if __name__ == "__main__":
    main()
```

The utility modules provide essential support functions for the medical image segmentation pipeline, ensuring robust logging, comprehensive data analysis, and efficient debugging capabilities. 