"""
data_preprocessor.py

Main orchestrator for CT scan data preprocessing. Handles dataset validation, splitting, transformation, normalization, and saving of preprocessed images, labels, and metadata for segmentation tasks. Supports both Medical Segmentation Decathlon (MSD) and JIPMER datasets.

Key Features:
- Validates dataset structure and format.
- Splits data into train/val/test sets.
- Applies spatial and intensity preprocessing, normalization, and augmentation.
- Saves preprocessed data and metadata for downstream ML workflows.
"""
from dataclasses import asdict
import json
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from configs.data_preprocessing_config import PreprocessingConfig
from data_preprocessing.handler import DatasetHandler
from data_preprocessing.jipmer_handler import JIPMERHandler
from data_preprocessing.lits_handler import LiTSHandler
from data_preprocessing.msd_handler import MedicalDecathlonHandler
from pipelines.data_preprocessing_pipeline import CTPreprocessingPipeline
import numpy as np


class DataPreprocessor:
    """
    Main data preprocessing orchestrator for CT scan segmentation datasets.

    Attributes:
        config (PreprocessingConfig): Configuration for preprocessing.
        phase (str): Imaging phase (e.g., 'Arterial') for JIPMER datasets.
        pipeline (CTPreprocessingPipeline): Pipeline for spatial/intensity transforms.
    """
    
    def __init__(self, config: PreprocessingConfig, phase: str = "Arterial"):
        """
        Initialize the DataPreprocessor.

        Args:
            config (PreprocessingConfig): Preprocessing configuration.
            phase (str): Imaging phase for JIPMER datasets.
        """
        self.config = config
        self.phase = phase
        self.pipeline = CTPreprocessingPipeline(config)
        
    def get_dataset_handler(self, dataset_path: str, dataset_type: str) -> DatasetHandler:
        """
        Factory method to get the appropriate dataset handler for the given dataset type.

        Args:
            dataset_path (str): Path to the dataset.
            dataset_type (str): Type of dataset ('medical_decathlon', 'jipmer', or 'lits').

        Returns:
            DatasetHandler: Handler for the specified dataset type.
        """
        if dataset_type == "medical_decathlon":
            return MedicalDecathlonHandler(dataset_path)
        elif dataset_type == "jipmer":
            return JIPMERHandler(dataset_path, phase=self.phase)
        elif dataset_type == "lits":
            return LiTSHandler(dataset_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def split_dataset(self, subjects: List[Dict[str, str]]) -> Tuple[List, List, List]:
        """
        Split the dataset into train, validation, and test sets.

        Args:
            subjects (List[Dict[str, str]]): List of subject metadata dicts.

        Returns:
            Tuple[List, List, List]: Train, validation, and test subject lists.
        """
        train_subjects, temp_subjects = train_test_split(
            subjects, test_size=(1 - self.config.train_ratio), random_state=42
        )
        
        val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        val_subjects, test_subjects = train_test_split(
            temp_subjects, test_size=(1 - val_ratio_adjusted), random_state=42
        )
        
        return train_subjects, val_subjects, test_subjects
    
    def process_dataset(self, dataset_path: str, dataset_type: str, output_dir: str):
        """
        Process the entire dataset: validate, split, preprocess, and save data and metadata.

        Args:
            dataset_path (str): Path to the dataset.
            dataset_type (str): Type of dataset ('medical_decathlon' or 'jipmer').
            output_dir (str): Directory to save preprocessed data.
        """
        print(f"Processing {dataset_type} dataset from {dataset_path}")
        
        # Initialize dataset handler
        handler = self.get_dataset_handler(dataset_path, dataset_type)
        
        # Validate dataset
        if not handler.validate_dataset():
            raise ValueError(f"Dataset validation failed for {dataset_path}")
        
        # Get subject list
        subjects = handler.get_subject_list()
        print(f"Found {len(subjects)} subjects")
        
        if not subjects:
            raise ValueError("No valid subjects found in dataset")
        
        # Split dataset
        train_subjects, val_subjects, test_subjects = self.split_dataset(subjects)
        
        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }
        
        # Process each split
        all_metadata = {}
        
        for split_name, split_subjects in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_subjects)} subjects)")
            
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create transforms
            transform = self.pipeline.create_transforms(is_training=(split_name == 'train'))
            
            split_metadata = []
            
            for subject_info in tqdm(split_subjects, desc=f"Processing {split_name}"):
                try:
                    # Add dataset type to subject info
                    subject_info['dataset_type'] = dataset_type
                    
                    # Process subject
                    processed_subject, metadata = self.pipeline.process_subject(
                        subject_info, transform
                    )
                    
                    # Save processed data
                    subject_id = subject_info['subject_id']
                    
                    # Save image and label
                    image_path = split_dir / f"{subject_id}_image.nii.gz"
                    label_path = split_dir / f"{subject_id}_label.nii.gz"
                    
                    processed_subject['image'].save(image_path)
                    processed_subject['label'].save(label_path)
                    
                    # Store metadata
                    split_metadata.append(metadata)
                    
                except Exception as e:
                    print(f"Error processing subject {subject_info['subject_id']}: {str(e)}")
                    continue
            
            all_metadata[split_name] = split_metadata
        
        # Save metadata and configuration
        self.save_preprocessing_info(output_path, all_metadata, self.config)
        
        print(f"\nPreprocessing completed. Output saved to {output_path}")
        print(f"Train: {len(all_metadata['train'])} subjects")
        print(f"Val: {len(all_metadata['val'])} subjects")
        print(f"Test: {len(all_metadata['test'])} subjects")
    
    def save_preprocessing_info(self, output_path: Path, metadata: Dict, config: PreprocessingConfig):
        """
        Save preprocessing configuration, metadata, and a human-readable summary.

        Args:
            output_path (Path): Output directory.
            metadata (Dict): Metadata for all splits.
            config (PreprocessingConfig): Preprocessing configuration.
        """
        # Save configuration
        config_path = output_path / "preprocessing_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Save metadata
        metadata_path = output_path / "preprocessing_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save human-readable summary
        summary_path = output_path / "preprocessing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CT Scan Preprocessing Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Target spacing: {config.target_spacing}\n")
            f.write(f"Target size: {config.target_size}\n")
            f.write(f"Intensity range: {config.intensity_range}\n")
            f.write(f"Normalization method: {config.normalize_method}\n\n")
            
            for split_name, split_metadata in metadata.items():
                f.write(f"{split_name.upper()} Split: {len(split_metadata)} subjects\n")
                if split_metadata:
                    dataset_types = set(m.dataset_type for m in split_metadata)
                    f.write(f"  Dataset types: {', '.join(dataset_types)}\n")
            f.write("\n")