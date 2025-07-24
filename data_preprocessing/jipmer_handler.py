from typing import Dict, List

from data_preprocessing.handler import DatasetHandler

PHASE_SUFFIXES = {
    "Arterial": "A",
    "Portal": "P",
    "Venous": "V"
}

class JIPMERHandler(DatasetHandler):
    """Handler for JIPMER dataset with phase support (Arterial, Portal, Venous)"""
    
    def __init__(self, dataset_path: str, phase: str = "Arterial"):
        super().__init__(dataset_path, "jipmer")
        if phase not in PHASE_SUFFIXES:
            raise ValueError(f"Unknown phase: {phase}. Must be one of {list(PHASE_SUFFIXES.keys())}")
        self.phase = phase
        self.phase_suffix = PHASE_SUFFIXES[phase]
        self.images_dir = self.dataset_path / f"niigz dicom/{phase} Phase"
        self.liver_dir = self.dataset_path / f"niigz liver/{phase} Phase"
        self.tumor_dir = self.dataset_path / f"niigz tumor/{phase} Phase"
        
    def validate_dataset(self) -> bool:
        """Validate JIPMER dataset structure"""
        required_dirs = [self.images_dir, self.liver_dir, self.tumor_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Error: Expected directory {dir_path.name} not found in {self.dataset_path}")
                return False
        return True
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """Get subject list for JIPMER format and selected phase"""
        subjects = []
        suffix = self.phase_suffix
        # Dicom files: Dliver1A.nii, Dliver2A.nii, ...
        for img_file in self.images_dir.glob(f"Dliver*{suffix}.nii"):
            # Extract subject number from filename (e.g., Dliver1A.nii -> 1)
            stem = img_file.stem  # e.g., Dliver1A
            subject_num = stem.replace("Dliver", "").replace(suffix, "")
            subject_id = f"{subject_num}{suffix}"  # e.g., 1A
            liver_file = self.liver_dir / f"LS{subject_num}{suffix}.nii"
            tumor_file = self.tumor_dir / f"TS{subject_num}{suffix}.nii"
            
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