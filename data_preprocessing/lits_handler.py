"""
lits_handler.py

Handler for the LiTS (Liver Tumor Segmentation) dataset format. Extracts subject lists and validates dataset structure for preprocessing pipelines.

The LiTS dataset structure:
- volumes/ (folder): Contains CT nifti files (volume-0.nii, volume-1.nii, etc.)
- segmentations/ (folder): Contains annotation nifti files (segmentation-0.nii, segmentation-1.nii, etc.)
"""
from typing import Dict, List
from pathlib import Path
import nibabel as nib
import numpy as np

from data_preprocessing.handler import DatasetHandler


class LiTSHandler(DatasetHandler):
    """
    Handler for LiTS (Liver Tumor Segmentation) dataset format.

    Attributes:
        volumes_dir (Path): Directory containing CT volume files.
        segmentations_dir (Path): Directory containing segmentation annotation files.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the LiTSHandler.

        Args:
            dataset_path (str): Path to the dataset root.
        """
        super().__init__(dataset_path, "lits")
        self.volumes_dir = self.dataset_path / "volumes"
        self.segmentations_dir = self.dataset_path / "segmentations"
        
    def validate_dataset(self) -> bool:
        """
        Validate the LiTS dataset structure (volumes and segmentations directories).

        Returns:
            bool: True if required directories exist, False otherwise.
        """
        if not self.volumes_dir.exists():
            print(f"Error: Expected directory 'volumes' not found in {self.dataset_path}")
            return False
            
        if not self.segmentations_dir.exists():
            print(f"Error: Expected directory 'segmentations' not found in {self.dataset_path}")
            return False
            
        # Check if there are any volume files
        volume_files = list(self.volumes_dir.glob("volume-*.nii*"))
        if not volume_files:
            print(f"Error: No volume files found in {self.volumes_dir}")
            return False
            
        # Check if there are any segmentation files
        segmentation_files = list(self.segmentations_dir.glob("segmentation-*.nii*"))
        if not segmentation_files:
            print(f"Error: No segmentation files found in {self.segmentations_dir}")
            return False
            
        return True
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """
        Get subject list for LiTS format.

        Returns:
            List[Dict[str, str]]: List of subject metadata dicts with image and label paths.
        """
        subjects = []
        
        # Find all volume files
        for volume_file in self.volumes_dir.glob("volume-*.nii*"):
            # Extract subject number from filename (e.g., volume-0.nii -> 0)
            subject_num = volume_file.stem.replace("volume-", "")
            segmentation_file = self.segmentations_dir / f"segmentation-{subject_num}.nii"
            
            # Check for different possible extensions
            if not segmentation_file.exists():
                segmentation_file = self.segmentations_dir / f"segmentation-{subject_num}.nii.gz"
            
            if segmentation_file.exists():
                subjects.append({
                    "subject_id": f"lits_{subject_num}",
                    "image": str(volume_file),
                    "combined_label": str(segmentation_file),
                    "liver_label": None,
                    "tumor_label": None
                })
            else:
                print(f"Warning: Segmentation file not found for volume {volume_file.name}")
                
        return subjects
    
    def extract_liver_and_tumor_labels(self, segmentation_path: str) -> tuple:
        """
        Extract liver and tumor labels from LiTS segmentation file.
        
        LiTS segmentation labels:
        - 0: Background
        - 1: Liver
        - 2: Tumor
        
        Args:
            segmentation_path (str): Path to the segmentation file.
            
        Returns:
            tuple: (liver_mask, tumor_mask) as numpy arrays.
        """
        try:
            # Load segmentation
            seg_img = nib.load(segmentation_path)
            seg_data = seg_img.get_fdata()
            
            # Extract liver mask (label 1)
            liver_mask = (seg_data == 1).astype(np.uint8)
            
            # Extract tumor mask (label 2)
            tumor_mask = (seg_data == 2).astype(np.uint8)
            
            return liver_mask, tumor_mask
            
        except Exception as e:
            print(f"Error extracting labels from {segmentation_path}: {str(e)}")
            return None, None
    
    def validate_segmentation_labels(self, segmentation_path: str) -> bool:
        """
        Validate that the segmentation file contains valid LiTS labels.
        
        Args:
            segmentation_path (str): Path to the segmentation file.
            
        Returns:
            bool: True if valid LiTS labels found, False otherwise.
        """
        try:
            seg_img = nib.load(segmentation_path)
            seg_data = seg_img.get_fdata()
            
            # Check for valid LiTS labels (0, 1, 2)
            unique_labels = np.unique(seg_data)
            valid_labels = {0, 1, 2}
            
            if not set(unique_labels).issubset(valid_labels):
                print(f"Warning: Invalid labels found in {segmentation_path}: {unique_labels}")
                return False
                
            # Check if liver (label 1) is present
            if 1 not in unique_labels:
                print(f"Warning: No liver labels found in {segmentation_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating segmentation {segmentation_path}: {str(e)}")
            return False
