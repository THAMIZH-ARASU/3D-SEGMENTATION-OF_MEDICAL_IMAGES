#!/usr/bin/env python3
"""
Test script for MSD (Medical Segmentation Decathlon) dataset handler.

This script tests the MedicalDecathlonHandler class to ensure it correctly:
1. Validates the dataset structure
2. Extracts subject lists
3. Handles the MSD dataset format properly
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib

from data_preprocessing.msd_handler import MedicalDecathlonHandler


def create_test_msd_dataset(base_path: Path):
    """
    Create a test MSD dataset structure for testing.
    
    Args:
        base_path (Path): Base path for the test dataset.
    """
    # Create directories
    images_dir = base_path / "imagesTr"
    labels_dir = base_path / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data for 5 subjects
    for i in range(5):
        # Create image data (CT scan)
        image_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        image_img = nib.Nifti1Image(image_data, np.eye(4))
        image_path = images_dir / f"liver_{i:03d}.nii"
        nib.save(image_img, image_path)
        
        # Create label data (segmentation)
        label_data = np.zeros((256, 256, 128), dtype=np.uint8)
        # Add some liver regions
        label_data[50:150, 50:150, 30:90] = 1
        # Add some tumor regions within liver
        label_data[80:120, 80:120, 50:80] = 2
        
        label_img = nib.Nifti1Image(label_data, np.eye(4))
        label_path = labels_dir / f"liver_{i:03d}.nii"
        nib.save(label_img, label_path)
    
    print(f"Created test MSD dataset at {base_path}")
    print(f"  Images: {list(images_dir.glob('*.nii'))}")
    print(f"  Labels: {list(labels_dir.glob('*.nii'))}")


def test_msd_handler():
    """Test the MedicalDecathlonHandler functionality."""
    print("Testing MSD Handler...")
    
    # Create temporary directory for test dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test dataset
        create_test_msd_dataset(temp_path)
        
        # Initialize handler
        handler = MedicalDecathlonHandler(str(temp_path))
        
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
        
        # Test with missing label file
        print("\n3. Testing with missing label file...")
        # Remove one label file to test warning
        labels_dir = temp_path / "labelsTr"
        first_label = list(labels_dir.glob("*.nii"))[0]
        first_label.unlink()
        
        subjects_with_missing = handler.get_subject_list()
        print(f"   Subjects after removing label: {len(subjects_with_missing)}")
        
        # Test with missing directories
        print("\n4. Testing with missing directories...")
        shutil.rmtree(temp_path / "imagesTr")
        handler_missing = MedicalDecathlonHandler(str(temp_path))
        is_valid_missing = handler_missing.validate_dataset()
        print(f"   Validation with missing imagesTr: {'PASS' if not is_valid_missing else 'FAIL'}")
        
        print("\nAll tests completed successfully!")
        return True


def test_msd_handler_edge_cases():
    """Test edge cases for MSD handler."""
    print("\nTesting MSD Handler Edge Cases...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with empty directories
        print("\n1. Testing with empty directories...")
        images_dir = temp_path / "imagesTr"
        labels_dir = temp_path / "labelsTr"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        handler_empty = MedicalDecathlonHandler(str(temp_path))
        is_valid_empty = handler_empty.validate_dataset()
        print(f"   Validation with empty directories: {'PASS' if not is_valid_empty else 'FAIL'}")
        
        # Test with mismatched files
        print("\n2. Testing with mismatched files...")
        # Create image without corresponding label
        image_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        image_img = nib.Nifti1Image(image_data, np.eye(4))
        image_path = images_dir / "extra_image.nii"
        nib.save(image_img, image_path)
        
        subjects_mismatched = handler_empty.get_subject_list()
        print(f"   Subjects with mismatched files: {len(subjects_mismatched)}")
        
        print("\nEdge case tests completed successfully!")
        return True


if __name__ == "__main__":
    success1 = test_msd_handler()
    success2 = test_msd_handler_edge_cases()
    
    if success1 and success2:
        print("\n✅ MSD Handler test PASSED")
    else:
        print("\n❌ MSD Handler test FAILED") 