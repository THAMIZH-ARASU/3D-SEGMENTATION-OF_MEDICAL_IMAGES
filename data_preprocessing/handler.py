from pathlib import Path
from typing import Dict, List


class DatasetHandler:
    """Base class for dataset handling"""
    
    def __init__(self, dataset_path: str, dataset_type: str):
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """Get list of subjects with their file paths"""
        raise NotImplementedError
        
    def validate_dataset(self) -> bool:
        """Validate dataset structure"""
        raise NotImplementedError