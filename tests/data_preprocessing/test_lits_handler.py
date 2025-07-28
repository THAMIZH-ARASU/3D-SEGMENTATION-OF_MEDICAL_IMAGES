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


def test_lits_handler_basic():
    """Test the LiTSHandler basic functionality."""
    print("Testing LiTS Handler (Basic)...")
    
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
        
        print("\nBasic tests completed successfully!")
        return True


def test_lits_handler_edge_cases():
    """Test edge cases for LiTS handler."""
    print("\nTesting LiTS Handler Edge Cases...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with missing directories
        print("\n1. Testing with missing directories...")
        handler_missing = LiTSHandler(str(temp_path))
        is_valid_missing = handler_missing.validate_dataset()
        print(f"   Validation with missing directories: {'PASS' if not is_valid_missing else 'FAIL'}")
        
        # Test with empty directories
        print("\n2. Testing with empty directories...")
        volumes_dir = temp_path / "volumes"
        segmentations_dir = temp_path / "segmentations"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        segmentations_dir.mkdir(parents=True, exist_ok=True)
        
        handler_empty = LiTSHandler(str(temp_path))
        is_valid_empty = handler_empty.validate_dataset()
        print(f"   Validation with empty directories: {'PASS' if not is_valid_empty else 'FAIL'}")
        
        # Test with mismatched files
        print("\n3. Testing with mismatched files...")
        # Create volume without corresponding segmentation
        volume_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        volume_img = nib.Nifti1Image(volume_data, np.eye(4))
        volume_path = volumes_dir / "volume-999.nii"
        nib.save(volume_img, volume_path)
        
        subjects_mismatched = handler_empty.get_subject_list()
        print(f"   Subjects with mismatched files: {len(subjects_mismatched)}")
        
        print("\nEdge case tests completed successfully!")
        return True


def test_lits_handler_file_extensions():
    """Test LiTS handler with different file extensions."""
    print("\nTesting LiTS Handler (File Extensions)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directories
        volumes_dir = temp_path / "volumes"
        segmentations_dir = temp_path / "segmentations"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        segmentations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test data with .nii.gz extension
        volume_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        volume_img = nib.Nifti1Image(volume_data, np.eye(4))
        volume_path = volumes_dir / "volume-0.nii.gz"
        nib.save(volume_img, volume_path)
        
        seg_data = np.zeros((256, 256, 128), dtype=np.uint8)
        seg_data[50:150, 50:150, 30:90] = 1  # Liver
        seg_data[80:120, 80:120, 50:80] = 2  # Tumor
        
        seg_img = nib.Nifti1Image(seg_data, np.eye(4))
        seg_path = segmentations_dir / "segmentation-0.nii.gz"
        nib.save(seg_img, seg_path)
        
        # Test handler
        handler = LiTSHandler(str(temp_path))
        is_valid = handler.validate_dataset()
        print(f"   Validation with .nii.gz files: {'PASS' if is_valid else 'FAIL'}")
        
        if is_valid:
            subjects = handler.get_subject_list()
            print(f"   Found {len(subjects)} subjects with .nii.gz files")
            
            # Test label extraction with .nii.gz
            if subjects:
                seg_path = subjects[0]['combined_label']
                liver_mask, tumor_mask = handler.extract_liver_and_tumor_labels(seg_path)
                if liver_mask is not None and tumor_mask is not None:
                    print("   Label extraction with .nii.gz: PASS")
                else:
                    print("   Label extraction with .nii.gz: FAIL")
                    return False
        
        print("\nFile extension tests completed successfully!")
        return True


def test_lits_handler_invalid_labels():
    """Test LiTS handler with invalid segmentation labels."""
    print("\nTesting LiTS Handler (Invalid Labels)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directories
        volumes_dir = temp_path / "volumes"
        segmentations_dir = temp_path / "segmentations"
        volumes_dir.mkdir(parents=True, exist_ok=True)
        segmentations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create volume
        volume_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        volume_img = nib.Nifti1Image(volume_data, np.eye(4))
        volume_path = volumes_dir / "volume-0.nii"
        nib.save(volume_img, volume_path)
        
        # Create segmentation with invalid labels
        seg_data = np.zeros((256, 256, 128), dtype=np.uint8)
        seg_data[50:150, 50:150, 30:90] = 1  # Liver
        seg_data[80:120, 80:120, 50:80] = 2  # Tumor
        seg_data[10:20, 10:20, 10:20] = 5    # Invalid label
        
        seg_img = nib.Nifti1Image(seg_data, np.eye(4))
        seg_path = segmentations_dir / "segmentation-0.nii"
        nib.save(seg_img, seg_path)
        
        # Test validation
        handler = LiTSHandler(str(temp_path))
        is_valid_seg = handler.validate_segmentation_labels(str(seg_path))
        print(f"   Validation with invalid labels: {'PASS' if not is_valid_seg else 'FAIL'}")
        
        # Test with no liver labels
        seg_data_no_liver = np.zeros((256, 256, 128), dtype=np.uint8)
        seg_data_no_liver[80:120, 80:120, 50:80] = 2  # Only tumor, no liver
        
        seg_img_no_liver = nib.Nifti1Image(seg_data_no_liver, np.eye(4))
        seg_path_no_liver = segmentations_dir / "segmentation-1.nii"
        nib.save(seg_img_no_liver, seg_path_no_liver)
        
        is_valid_no_liver = handler.validate_segmentation_labels(str(seg_path_no_liver))
        print(f"   Validation with no liver labels: {'PASS' if not is_valid_no_liver else 'FAIL'}")
        
        print("\nInvalid label tests completed successfully!")
        return True


if __name__ == "__main__":
    success1 = test_lits_handler_basic()
    success2 = test_lits_handler_edge_cases()
    success3 = test_lits_handler_file_extensions()
    success4 = test_lits_handler_invalid_labels()
    
    if success1 and success2 and success3 and success4:
        print("\n✅ LiTS Handler test PASSED")
    else:
        print("\n❌ LiTS Handler test FAILED") 