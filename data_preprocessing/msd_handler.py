"""
Handler for the Medical Segmentation Decathlon (MSD) dataset format. Extracts subject lists and validates dataset structure for preprocessing pipelines.
"""
from typing import Dict, List
from data_preprocessing.handler import DatasetHandler


class MedicalDecathlonHandler(DatasetHandler):
    """
    Handler for Medical Segmentation Decathlon dataset format.

    Attributes:
        images_dir (Path): Directory for training images.
        labels_dir (Path): Directory for training labels.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the MedicalDecathlonHandler.

        Args:
            dataset_path (str): Path to the dataset root.
        """
        super().__init__(dataset_path, "medical_decathlon")
        self.images_dir = self.dataset_path / "imagesTr"
        self.labels_dir = self.dataset_path / "labelsTr"
        
    def validate_dataset(self) -> bool:
        """
        Validate the MSD dataset structure (imagesTr and labelsTr directories).

        Returns:
            bool: True if required directories exist, False otherwise.
        """
        if not self.images_dir.exists() or not self.labels_dir.exists():
            print(f"Error: Expected directories 'imagesTr' and 'labelsTr' not found in {self.dataset_path}")
            return False
        return True
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """
        Get subject list for Medical Decathlon format.

        Returns:
            List[Dict[str, str]]: List of subject metadata dicts with image and label paths.
        """
        subjects = []
        
        for img_file in self.images_dir.glob("*.nii"):
            subject_id = img_file.stem.replace(".nii", "")
            label_file = self.labels_dir / img_file.name
            
            if label_file.exists():
                subjects.append({
                    "subject_id": subject_id,
                    "image": str(img_file),
                    "combined_label": str(label_file),
                    "liver_label": None,
                    "tumor_label": None
                })
            else:
                print(f"Warning: Label file not found for {img_file.name}")
                
        return subjects