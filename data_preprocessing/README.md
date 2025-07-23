# Data Preprocessing Module

This module provides tools and pipelines for preprocessing CT scan datasets for segmentation tasks. It supports both the Medical Segmentation Decathlon (MSD) and JIPMER dataset formats, and outputs preprocessed images, labels, and metadata for downstream machine learning workflows.

## Main Components

- **data_preprocessor.py**: Orchestrates the preprocessing workflow, including dataset splitting, transformation, and saving outputs.
- **handler.py**: Base class for dataset handlers.
- **msd_handler.py**: Handler for Medical Decathlon datasets.
- **jipmer_handler.py**: Handler for JIPMER datasets.
- **intensity_normalizer.py**: Implements intensity normalization methods (z-score, min-max, robust).
- **subject_metadata.py**: Dataclass for storing subject metadata.
- **pipelines/data_preprocessing_pipeline.py**: Main pipeline for spatial and intensity preprocessing, augmentation, and metadata extraction.

## Features

- **Supports multiple dataset formats** (MSD, JIPMER)
- **Spatial preprocessing**: Resampling, resizing
- **Intensity preprocessing**: Clipping, normalization
- **Data augmentation** (optional, for training)
- **Metadata extraction**: Spacing, shape, origin, direction, affine, intensity stats
- **Train/val/test split**
- **Saves preprocessed images, labels, and metadata**

## Usage

### Command Line

Run the preprocessing pipeline with:

```bash
python3 run_data_preprocessing.py \
  --dataset_path /path/to/dataset \
  --dataset_type [medical_decathlon|jipmer] \
  --output_dir data_preprocessed \
  [--config /path/to/config.json]
```

#### Example

```bash
python3 run_data_preprocessing.py \
  --dataset_path /home/user/Data/Task03_Liver_rs \
  --dataset_type medical_decathlon \
  --output_dir data_preprocessed
```

### Configuration

You can specify preprocessing parameters via command-line arguments or a JSON config file. Key parameters include:
- `target_spacing`: Target voxel spacing (default: [1.0, 1.0, 1.0])
- `target_size`: Target image size (default: [256, 256, 128])
- `intensity_range`: Intensity clipping range (default: [-100, 400])
- `normalize_method`: Normalization method (`zscore`, `minmax`, `robust`)
- `apply_augmentation`: Enable/disable augmentation (default: True)
- `train_ratio`, `val_ratio`, `test_ratio`: Dataset split ratios

## Input Structure

- **MSD Format**:
  - `imagesTr/`: Input images (`.nii`)
  - `labelsTr/`: Corresponding labels (`.nii`)
- **JIPMER Format**:
  - `images/`: Input images (`.nii.gz`)
  - `liver_masks/`: Liver masks (`.nii.gz`)
  - `tumor_masks/`: Tumor masks (`.nii.gz`)

## Output Structure

- `data_preprocessed/`
  - `train/`, `val/`, `test/`: Preprocessed splits
    - `{subject_id}_image.nii.gz`: Preprocessed image
    - `{subject_id}_label.nii.gz`: Preprocessed label
  - `preprocessing_config.json`: Used configuration
  - `preprocessing_metadata.pkl`: Metadata for all subjects
  - `preprocessing_summary.txt`: Human-readable summary

## Extending

- Add new dataset handlers by subclassing `DatasetHandler`.
- Add new normalization methods in `intensity_normalizer.py`.
- Modify or extend the pipeline in `pipelines/data_preprocessing_pipeline.py`.

