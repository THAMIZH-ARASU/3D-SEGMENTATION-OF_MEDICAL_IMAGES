#!/usr/bin/env python3
"""
Test script for the base DatasetHandler class.

This script tests the base DatasetHandler class to ensure:
1. The interface is properly defined
2. Subclasses can be created correctly
3. Abstract methods are properly enforced
"""

import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib

from data_preprocessing.handler import DatasetHandler


class TestHandler(DatasetHandler):
    """
    A test implementation of DatasetHandler for testing purposes.
    """
    
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, "test")
        self.test_dir = self.dataset_path / "test_data"
        
    def validate_dataset(self) -> bool:
        """Test implementation of validate_dataset."""
        return self.test_dir.exists()
        
    def get_subject_list(self):
        """Test implementation of get_subject_list."""
        subjects = []
        if self.test_dir.exists():
            for file_path in self.test_dir.glob("*.nii"):
                subjects.append({
                    "subject_id": file_path.stem,
                    "image": str(file_path),
                    "combined_label": str(file_path),
                    "liver_label": None,
                    "tumor_label": None
                })
        return subjects


def test_base_handler_interface():
    """Test the base DatasetHandler interface."""
    print("Testing Base DatasetHandler Interface...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test handler initialization
        print("\n1. Testing handler initialization...")
        handler = TestHandler(str(temp_path))
        print(f"   Dataset path: {handler.dataset_path}")
        print(f"   Dataset type: {handler.dataset_type}")
        print("   Initialization: PASS")
        
        # Test validation with missing directory
        print("\n2. Testing validation with missing directory...")
        is_valid_missing = handler.validate_dataset()
        print(f"   Validation with missing directory: {'PASS' if not is_valid_missing else 'FAIL'}")
        
        # Test validation with existing directory
        print("\n3. Testing validation with existing directory...")
        test_dir = temp_path / "test_data"
        test_dir.mkdir(parents=True, exist_ok=True)
        is_valid_existing = handler.validate_dataset()
        print(f"   Validation with existing directory: {'PASS' if is_valid_existing else 'FAIL'}")
        
        # Test subject list extraction
        print("\n4. Testing subject list extraction...")
        subjects = handler.get_subject_list()
        print(f"   Subjects with empty directory: {len(subjects)}")
        
        # Test with actual files
        print("\n5. Testing with actual files...")
        # Create a test file
        test_data = np.random.randint(0, 255, size=(64, 64, 32), dtype=np.uint8)
        test_img = nib.Nifti1Image(test_data, np.eye(4))
        test_file = test_dir / "test_subject.nii"
        nib.save(test_img, test_file)
        
        subjects_with_files = handler.get_subject_list()
        print(f"   Subjects with files: {len(subjects_with_files)}")
        
        if subjects_with_files:
            subject = subjects_with_files[0]
            print(f"   Subject ID: {subject['subject_id']}")
            print(f"   Image path: {subject['image']}")
            print("   Subject extraction: PASS")
        
        print("\nBase handler interface tests completed successfully!")
        return True


def test_base_handler_inheritance():
    """Test that subclasses properly inherit from DatasetHandler."""
    print("\nTesting Base DatasetHandler Inheritance...")
    
    # Test that TestHandler is a subclass of DatasetHandler
    print("\n1. Testing inheritance...")
    if issubclass(TestHandler, DatasetHandler):
        print("   Inheritance: PASS")
    else:
        print("   Inheritance: FAIL")
        return False
    
    # Test that required methods are implemented
    print("\n2. Testing method implementation...")
    required_methods = ['validate_dataset', 'get_subject_list']
    for method_name in required_methods:
        if hasattr(TestHandler, method_name):
            print(f"   {method_name}: PASS")
        else:
            print(f"   {method_name}: FAIL")
            return False
    
    # Test instantiation
    print("\n3. Testing instantiation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            handler = TestHandler(temp_dir)
            print("   Instantiation: PASS")
        except Exception as e:
            print(f"   Instantiation: FAIL - {e}")
            return False
    
    print("\nInheritance tests completed successfully!")
    return True


def test_base_handler_attributes():
    """Test that the base handler has the correct attributes."""
    print("\nTesting Base DatasetHandler Attributes...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = TestHandler(temp_dir)
        
        # Test required attributes
        print("\n1. Testing required attributes...")
        required_attrs = ['dataset_path', 'dataset_type']
        for attr_name in required_attrs:
            if hasattr(handler, attr_name):
                print(f"   {attr_name}: PASS")
            else:
                print(f"   {attr_name}: FAIL")
                return False
        
        # Test attribute types
        print("\n2. Testing attribute types...")
        if isinstance(handler.dataset_path, Path):
            print("   dataset_path type: PASS")
        else:
            print("   dataset_path type: FAIL")
            return False
        
        if isinstance(handler.dataset_type, str):
            print("   dataset_type type: PASS")
        else:
            print("   dataset_type type: FAIL")
            return False
        
        print("\nAttribute tests completed successfully!")
        return True


if __name__ == "__main__":
    success1 = test_base_handler_interface()
    success2 = test_base_handler_inheritance()
    success3 = test_base_handler_attributes()
    
    if success1 and success2 and success3:
        print("\n✅ Base DatasetHandler test PASSED")
    else:
        print("\n❌ Base DatasetHandler test FAILED") 