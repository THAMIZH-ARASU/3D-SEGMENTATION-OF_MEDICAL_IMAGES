from typing import Dict, List

from data_preprocessing.handler import DatasetHandler


class JIPMERHandler(DatasetHandler):
    """Handler for JIPMER dataset"""
    
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, "jipmer")
        self.images_dir = self.dataset_path / "images"
        self.liver_dir = self.dataset_path / "liver_masks"
        self.tumor_dir = self.dataset_path / "tumor_masks"
        
    def validate_dataset(self) -> bool:
        """Validate JIPMER dataset structure"""
        required_dirs = [self.images_dir, self.liver_dir, self.tumor_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Error: Expected directory {dir_path.name} not found in {self.dataset_path}")
                return False
        return True
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """Get subject list for JIPMER format"""
        subjects = []
        
        for img_file in self.images_dir.glob("*.nii.gz"):
            subject_id = img_file.stem.replace(".nii", "")
            liver_file = self.liver_dir / img_file.name
            tumor_file = self.tumor_dir / img_file.name
            
            if liver_file.exists() and tumor_file.exists():
                subjects.append({
                    "subject_id": subject_id,
                    "image": str(img_file),
                    "combined_label": None,
                    "liver_label": str(liver_file),
                    "tumor_label": str(tumor_file)
                })
            else:
                missing = []
                if not liver_file.exists():
                    missing.append("liver")
                if not tumor_file.exists():
                    missing.append("tumor")
                print(f"Warning: Missing {', '.join(missing)} mask(s) for {img_file.name}")
                
        return subjects