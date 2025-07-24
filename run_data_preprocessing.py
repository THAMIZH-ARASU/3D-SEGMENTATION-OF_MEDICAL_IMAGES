import argparse
import json
from pathlib import Path
import torchio as tio

from cofigs.data_preprocessing_config import PreprocessingConfig
from data_preprocessing.data_preprocessor import DataPreprocessor

def create_pytorch_dataset(processed_data_dir: str, split: str = 'train'):
    """Create PyTorch dataset from processed data"""
    from torch.utils.data import Dataset
    
    class CTSegmentationDataset(Dataset):
        def __init__(self, data_dir: str, split: str):
            self.data_dir = Path(data_dir) / split
            self.subjects = []
            
            # Find all processed subjects
            for image_file in self.data_dir.glob("*_image.nii.gz"):
                subject_id = image_file.stem.replace("_image", "")
                label_file = self.data_dir / f"{subject_id}_label.nii.gz"
                
                if label_file.exists():
                    self.subjects.append({
                        'subject_id': subject_id,
                        'image_path': str(image_file),
                        'label_path': str(label_file)
                    })
        
        def __len__(self):
            return len(self.subjects)
        
        def __getitem__(self, idx):
            subject = self.subjects[idx]
            
            # Load image and label
            image = tio.ScalarImage(subject['image_path'])
            label = tio.LabelMap(subject['label_path'])
            
            return {
                'image': image.data,  # Shape: (1, H, W, D)
                'label': label.data.squeeze().long(),  # Shape: (H, W, D)
                'subject_id': subject['subject_id']
            }
    
    return CTSegmentationDataset(processed_data_dir, split)


def main():
    parser = argparse.ArgumentParser(description="CT Scan Preprocessing Pipeline")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--dataset_type", type=str, required=True,
                       choices=["medical_decathlon", "jipmer"],
                       help="Type of dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration JSON file")
    # Add phase argument for JIPMER
    parser.add_argument("--phase", type=str, default="Arterial", choices=["Arterial", "Portal", "Venous"],
                       help="Phase for JIPMER dataset (Arterial, Portal, Venous)")
    # Configuration parameters
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--target_size", nargs=3, type=int, default=[256, 256, 256])
    parser.add_argument("--intensity_range", nargs=2, type=float, default=[-100, 400])
    parser.add_argument("--normalize_method", type=str, default="zscore",
                       choices=["zscore", "minmax", "robust"])
    parser.add_argument("--no_augmentation", action="store_true",
                       help="Disable data augmentation")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PreprocessingConfig(**config_dict)
    else:
        config = PreprocessingConfig(
            target_spacing=tuple(args.target_spacing),
            target_size=tuple(args.target_size),
            intensity_range=tuple(args.intensity_range),
            normalize_method=args.normalize_method,
            apply_augmentation=not args.no_augmentation
        )
    
    # Initialize and run preprocessor, passing phase for JIPMER
    preprocessor = DataPreprocessor(config, phase=args.phase)
    preprocessor.process_dataset(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


# Example usage 

# MSD dataset
# python3 run_data_preprocessing.py --dataset_path /home/icmr/Documents/D-Former-Attempt/Data/Task03_Liver_rs --dataset_type medical_decathlon --output_dir data_preprocessed


# JIPMER dataset
# python3 run_data_preprocessing.py --dataset_path /home/icmr/Documents/MultiPhaseSegmentation/JIPMER_Data --dataset_type jipmer --output_dir data_preprocessed --phase Arterial
