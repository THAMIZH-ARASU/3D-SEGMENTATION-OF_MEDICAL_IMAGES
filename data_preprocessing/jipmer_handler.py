"""
Handler for the JIPMER CT scan dataset format, supporting multiple imaging phases (Arterial, Portal, Venous). Extracts subject lists and validates dataset structure for preprocessing pipelines.
"""
from typing import Dict, List

from data_preprocessing.handler import DatasetHandler

PHASE_SUFFIXES = {
    "Arterial": "A",
    "Portal": "P",
    "Venous": "V"
}

class JIPMERHandler(DatasetHandler):
    """
    Handler for JIPMER dataset with phase support (Arterial, Portal, Venous).

    Attributes:
        phase (str): Imaging phase (e.g., 'Arterial').
        phase_suffix (str): Suffix for phase-specific files.
        images_dir (Path): Directory for phase images.
        liver_dir (Path): Directory for liver masks.
        tumor_dir (Path): Directory for tumor masks.
    """
    
    def __init__(self, dataset_path: str, phase: str = "Arterial"):
        """
        Initialize the JIPMERHandler for a specific phase.

        Args:
            dataset_path (str): Path to the dataset root.
            phase (str): Imaging phase ('Arterial', 'Portal', 'Venous').
        """
        super().__init__(dataset_path, "jipmer")
        if phase not in PHASE_SUFFIXES:
            raise ValueError(f"Unknown phase: {phase}. Must be one of {list(PHASE_SUFFIXES.keys())}")
        self.phase = phase
        self.phase_suffix = PHASE_SUFFIXES[phase]
        self.images_dir = self.dataset_path / f"niigz dicom/{phase} Phase"
        self.liver_dir = self.dataset_path / f"niigz liver/{phase} Phase"
        self.tumor_dir = self.dataset_path / f"niigz tumor/{phase} Phase"
        
    def validate_dataset(self) -> bool:
        """
        Validate the JIPMER dataset structure for the selected phase.

        Returns:
            bool: True if all required directories exist, False otherwise.
        """
        required_dirs = [self.images_dir, self.liver_dir, self.tumor_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Error: Expected directory {dir_path.name} not found in {self.dataset_path}")
                return False
        return True
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """
        Get subject list for JIPMER format and selected phase.

        Returns:
            List[Dict[str, str]]: List of subject metadata dicts with image and mask paths.
        """
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