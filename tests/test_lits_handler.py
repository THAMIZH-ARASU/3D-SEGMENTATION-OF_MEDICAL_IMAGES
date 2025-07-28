#!/usr/bin/env python3
"""
Test script for LiTS dataset handler.

This script tests the LiTSHandler class to ensure it correctly:
1. Validates the dataset structure
2. Extracts subject lists
3. Handles the LiTS dataset format properly
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib

from data_preprocessing.lits_handler import LiTSHandler


def create_test_lits_dataset(base_path: Path):
    """
    Create a test LiTS dataset structure for testing.
    
    Args:
        base_path (Path): Base path for the test dataset.
    """
    # Create directories
    volumes_dir = base_path / "volumes"
    segmentations_dir = base_path / "segmentations"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    segmentations_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data for 3 subjects
    for i in range(3):
        # Create volume data (CT scan)
        volume_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        volume_img = nib.Nifti1Image(volume_data, np.eye(4))
        volume_path = volumes_dir / f"volume-{i}.nii"
        nib.save(volume_img, volume_path)
        
        # Create segmentation data (labels: 0=background, 1=liver, 2=tumor)
        seg_data = np.zeros((256, 256, 128), dtype=np.uint8)
        # Add some liver regions
        seg_data[50:150, 50:150, 30:90] = 1
        # Add some tumor regions within liver
        seg_data[80:120, 80:120, 50:80] = 2
        
        seg_img = nib.Nifti1Image(seg_data, np.eye(4))
        seg_path = segmentations_dir / f"segmentation-{i}.nii"
        nib.save(seg_img, seg_path)
    
    print(f"Created test LiTS dataset at {base_path}")
    print(f"  Volumes: {list(volumes_dir.glob('*.nii'))}")
    print(f"  Segmentations: {list(segmentations_dir.glob('*.nii'))}")


def test_lits_handler():
    """Test the LiTSHandler functionality."""
    print("Testing LiTS Handler...")
    
    # Create temporary directory for test dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test dataset
        create_test_lits_dataset(temp_path)
        
        # Initialize handler
        handler = LiTSHandler(str(temp_path))
        
        # Test validation
        print("\n1. Testing dataset validation...")
        is_valid = handler.validate_dataset()
        print(f"   Dataset validation: {'PASS' if is_valid else 'FAIL'}")
        
        if not is_valid:
            print("   Validation failed!")
            return False
        
        # Test subject list extraction
        print("\n2. Testing subject list extraction...")
        subjects = handler.get_subject_list()
        print(f"   Found {len(subjects)} subjects")
        
        for subject in subjects:
            print(f"   Subject: {subject['subject_id']}")
            print(f"     Image: {subject['image']}")
            print(f"     Label: {subject['combined_label']}")
        
        # Test label extraction
        print("\n3. Testing label extraction...")
        if subjects:
            first_subject = subjects[0]
            seg_path = first_subject['combined_label']
            
            liver_mask, tumor_mask = handler.extract_liver_and_tumor_labels(seg_path)
            
            if liver_mask is not None and tumor_mask is not None:
                print(f"   Liver mask shape: {liver_mask.shape}")
                print(f"   Tumor mask shape: {tumor_mask.shape}")
                print(f"   Liver voxels: {np.sum(liver_mask)}")
                print(f"   Tumor voxels: {np.sum(tumor_mask)}")
                print("   Label extraction: PASS")
            else:
                print("   Label extraction: FAIL")
                return False
        
        # Test segmentation validation
        print("\n4. Testing segmentation validation...")
        if subjects:
            seg_path = subjects[0]['combined_label']
            is_valid_seg = handler.validate_segmentation_labels(seg_path)
            print(f"   Segmentation validation: {'PASS' if is_valid_seg else 'FAIL'}")
        
        print("\nAll tests completed successfully!")
        return True


if __name__ == "__main__":
    success = test_lits_handler()
    if success:
        print("\n✅ LiTS Handler test PASSED")
    else:
        print("\n❌ LiTS Handler test FAILED") 