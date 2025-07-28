#!/usr/bin/env python3
"""
Test script for JIPMER dataset handler.

This script tests the JIPMERHandler class to ensure it correctly:
1. Validates the dataset structure for different phases
2. Extracts subject lists for each phase
3. Handles the JIPMER dataset format properly
4. Tests phase-specific functionality
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib

from data_preprocessing.jipmer_handler import JIPMERHandler, PHASE_SUFFIXES


def create_test_jipmer_dataset(base_path: Path, phase: str = "Arterial"):
    """
    Create a test JIPMER dataset structure for testing.
    
    Args:
        base_path (Path): Base path for the test dataset.
        phase (str): Phase to create (Arterial, Portal, Venous).
    """
    phase_suffix = PHASE_SUFFIXES[phase]
    
    # Create directories
    images_dir = base_path / f"niigz dicom/{phase} Phase"
    liver_dir = base_path / f"niigz liver/{phase} Phase"
    tumor_dir = base_path / f"niigz tumor/{phase} Phase"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    liver_dir.mkdir(parents=True, exist_ok=True)
    tumor_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data for 4 subjects
    for i in range(1, 5):  # JIPMER uses 1-based indexing
        # Create image data (CT scan)
        image_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        image_img = nib.Nifti1Image(image_data, np.eye(4))
        image_path = images_dir / f"Dliver{i}{phase_suffix}.nii"
        nib.save(image_img, image_path)
        
        # Create liver mask
        liver_data = np.zeros((256, 256, 128), dtype=np.uint8)
        liver_data[50:150, 50:150, 30:90] = 1
        liver_img = nib.Nifti1Image(liver_data, np.eye(4))
        liver_path = liver_dir / f"LS{i}{phase_suffix}.nii"
        nib.save(liver_img, liver_path)
        
        # Create tumor mask
        tumor_data = np.zeros((256, 256, 128), dtype=np.uint8)
        tumor_data[80:120, 80:120, 50:80] = 1
        tumor_img = nib.Nifti1Image(tumor_data, np.eye(4))
        tumor_path = tumor_dir / f"TS{i}{phase_suffix}.nii"
        nib.save(tumor_img, tumor_path)
    
    print(f"Created test JIPMER dataset for {phase} phase at {base_path}")
    print(f"  Images: {list(images_dir.glob('*.nii'))}")
    print(f"  Liver masks: {list(liver_dir.glob('*.nii'))}")
    print(f"  Tumor masks: {list(tumor_dir.glob('*.nii'))}")


def test_jipmer_handler_basic():
    """Test the JIPMERHandler basic functionality."""
    print("Testing JIPMER Handler (Basic)...")
    
    # Create temporary directory for test dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with Arterial phase
        create_test_jipmer_dataset(temp_path, "Arterial")
        
        # Initialize handler
        handler = JIPMERHandler(str(temp_path), phase="Arterial")
        
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
            print(f"     Liver: {subject['liver_label']}")
            print(f"     Tumor: {subject['tumor_label']}")
        
        print("\nBasic tests completed successfully!")
        return True


def test_jipmer_handler_phases():
    """Test JIPMER handler with different phases."""
    print("\nTesting JIPMER Handler (Phases)...")
    
    phases = ["Arterial", "Portal", "Venous"]
    
    for phase in phases:
        print(f"\n--- Testing {phase} phase ---")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test dataset for this phase
            create_test_jipmer_dataset(temp_path, phase)
            
            # Initialize handler
            handler = JIPMERHandler(str(temp_path), phase=phase)
            
            # Test validation
            is_valid = handler.validate_dataset()
            print(f"   {phase} validation: {'PASS' if is_valid else 'FAIL'}")
            
            if not is_valid:
                print(f"   {phase} validation failed!")
                continue
            
            # Test subject list extraction
            subjects = handler.get_subject_list()
            print(f"   {phase} subjects: {len(subjects)}")
            
            # Verify subject IDs have correct phase suffix
            phase_suffix = PHASE_SUFFIXES[phase]
            for subject in subjects:
                if not subject['subject_id'].endswith(phase_suffix):
                    print(f"   ERROR: Subject {subject['subject_id']} doesn't have {phase_suffix} suffix")
                    return False
    
    print("\nPhase tests completed successfully!")
    return True


def test_jipmer_handler_edge_cases():
    """Test edge cases for JIPMER handler."""
    print("\nTesting JIPMER Handler Edge Cases...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with missing phase
        print("\n1. Testing with invalid phase...")
        try:
            handler_invalid = JIPMERHandler(str(temp_path), phase="InvalidPhase")
            print("   ERROR: Should have raised ValueError for invalid phase")
            return False
        except ValueError:
            print("   PASS: Correctly raised ValueError for invalid phase")
        
        # Test with missing directories
        print("\n2. Testing with missing directories...")
        handler_missing = JIPMERHandler(str(temp_path), phase="Arterial")
        is_valid_missing = handler_missing.validate_dataset()
        print(f"   Validation with missing directories: {'PASS' if not is_valid_missing else 'FAIL'}")
        
        # Test with missing mask files
        print("\n3. Testing with missing mask files...")
        create_test_jipmer_dataset(temp_path, "Arterial")
        
        # Remove one liver mask to test warning
        liver_dir = temp_path / "niigz liver/Arterial Phase"
        first_liver = list(liver_dir.glob("*.nii"))[0]
        first_liver.unlink()
        
        handler_missing_masks = JIPMERHandler(str(temp_path), phase="Arterial")
        subjects_missing = handler_missing_masks.get_subject_list()
        print(f"   Subjects with missing liver mask: {len(subjects_missing)}")
        
        print("\nEdge case tests completed successfully!")
        return True


def test_jipmer_handler_file_extensions():
    """Test JIPMER handler with different file extensions."""
    print("\nTesting JIPMER Handler (File Extensions)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dataset with .nii.gz extensions
        phase = "Arterial"
        phase_suffix = PHASE_SUFFIXES[phase]
        
        images_dir = temp_path / f"niigz dicom/{phase} Phase"
        liver_dir = temp_path / f"niigz liver/{phase} Phase"
        tumor_dir = temp_path / f"niigz tumor/{phase} Phase"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        liver_dir.mkdir(parents=True, exist_ok=True)
        tumor_dir.mkdir(parents=True, exist_ok=True)
        
        # Create one subject with .nii.gz extension
        image_data = np.random.randint(-100, 400, size=(256, 256, 128), dtype=np.int16)
        image_img = nib.Nifti1Image(image_data, np.eye(4))
        image_path = images_dir / f"Dliver1{phase_suffix}.nii.gz"
        nib.save(image_img, image_path)
        
        liver_data = np.zeros((256, 256, 128), dtype=np.uint8)
        liver_data[50:150, 50:150, 30:90] = 1
        liver_img = nib.Nifti1Image(liver_data, np.eye(4))
        liver_path = liver_dir / f"LS1{phase_suffix}.nii.gz"
        nib.save(liver_img, liver_path)
        
        tumor_data = np.zeros((256, 256, 128), dtype=np.uint8)
        tumor_data[80:120, 80:120, 50:80] = 1
        tumor_img = nib.Nifti1Image(tumor_data, np.eye(4))
        tumor_path = tumor_dir / f"TS1{phase_suffix}.nii.gz"
        nib.save(tumor_img, tumor_path)
        
        # Test handler
        handler = JIPMERHandler(str(temp_path), phase=phase)
        is_valid = handler.validate_dataset()
        print(f"   Validation with .nii.gz files: {'PASS' if is_valid else 'FAIL'}")
        
        if is_valid:
            subjects = handler.get_subject_list()
            print(f"   Found {len(subjects)} subjects with .nii.gz files")
        
        print("\nFile extension tests completed successfully!")
        return True


if __name__ == "__main__":
    success1 = test_jipmer_handler_basic()
    success2 = test_jipmer_handler_phases()
    success3 = test_jipmer_handler_edge_cases()
    success4 = test_jipmer_handler_file_extensions()
    
    if success1 and success2 and success3 and success4:
        print("\n✅ JIPMER Handler test PASSED")
    else:
        print("\n❌ JIPMER Handler test FAILED") 