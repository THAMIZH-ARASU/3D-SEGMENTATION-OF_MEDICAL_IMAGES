"""
handler.py

Base class for dataset handlers in CT scan segmentation preprocessing. Defines the interface for dataset validation and subject list extraction for different dataset formats (e.g., MSD, JIPMER).
"""
from pathlib import Path
from typing import Dict, List


class DatasetHandler:
    """
    Base class for dataset handling in preprocessing pipelines.

    Attributes:
        dataset_path (Path): Path to the dataset root.
        dataset_type (str): Type of dataset (e.g., 'medical_decathlon', 'jipmer').
    """
    
    def __init__(self, dataset_path: str, dataset_type: str):
        """
        Initialize the DatasetHandler.

        Args:
            dataset_path (str): Path to the dataset root.
            dataset_type (str): Type of dataset.
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """
        Get a list of subjects with their file paths for images and labels.

        Returns:
            List[Dict[str, str]]: List of subject metadata dicts.
        """
        raise NotImplementedError
        
    def validate_dataset(self) -> bool:
        """
        Validate the dataset structure (e.g., required directories/files exist).

        Returns:
            bool: True if dataset is valid, False otherwise.
        """
        raise NotImplementedError