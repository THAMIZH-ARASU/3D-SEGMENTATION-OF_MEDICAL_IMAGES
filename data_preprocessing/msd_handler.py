from typing import Dict, List
from data_preprocessing.handler import DatasetHandler


class MedicalDecathlonHandler(DatasetHandler):
    """Handler for Medical Segmentation Decathlon dataset"""
    
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, "medical_decathlon")
        self.images_dir = self.dataset_path / "imagesTr"
        self.labels_dir = self.dataset_path / "labelsTr"
        
    def validate_dataset(self) -> bool:
        """Validate Medical Decathlon dataset structure"""
        if not self.images_dir.exists() or not self.labels_dir.exists():
            print(f"Error: Expected directories 'imagesTr' and 'labelsTr' not found in {self.dataset_path}")
            return False
        return True
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """Get subject list for Medical Decathlon format"""
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