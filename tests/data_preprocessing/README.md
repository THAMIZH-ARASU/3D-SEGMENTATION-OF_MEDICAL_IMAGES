# Data Preprocessing Tests

This directory contains comprehensive tests for all data preprocessing handlers in the CT scan segmentation project.

## Test Files

### Handler Tests

- **`test_base_handler.py`**: Tests for the base DatasetHandler class and interface
- **`test_msd_handler.py`**: Tests for Medical Segmentation Decathlon (MSD) dataset handler
- **`test_jipmer_handler.py`**: Tests for JIPMER dataset handler with phase support
- **`test_lits_handler.py`**: Tests for LiTS (Liver Tumor Segmentation) dataset handler
- **`run_all_handler_tests.py`**: Comprehensive test runner for all handlers

## Running Tests

### Run All Tests

To run all handler tests with a comprehensive summary:

```bash
python tests/data_preprocessing/run_all_handler_tests.py
```

### Run Individual Handler Tests

To run tests for a specific handler:

```bash
# MSD Handler
python tests/data_preprocessing/test_msd_handler.py

# JIPMER Handler
python tests/data_preprocessing/test_jipmer_handler.py

# LiTS Handler
python tests/data_preprocessing/test_lits_handler.py
```

## Test Coverage

### Base Handler Tests

- **Interface testing**: Tests the base DatasetHandler interface
- **Inheritance**: Verifies proper subclassing and method implementation
- **Attributes**: Tests required attributes and their types

# MSD Handler Tests

- **Basic functionality**: Dataset validation and subject list extraction
- **Edge cases**: Missing directories, empty directories, mismatched files
- **File handling**: Proper handling of .nii files

### JIPMER Handler Tests

- **Basic functionality**: Dataset validation and subject list extraction
- **Phase support**: Testing all three phases (Arterial, Portal, Venous)
- **Edge cases**: Invalid phases, missing directories, missing mask files
- **File extensions**: Support for both .nii and .nii.gz files

### LiTS Handler Tests

- **Basic functionality**: Dataset validation and subject list extraction
- **Label extraction**: Extracting liver and tumor masks from segmentation files
- **Segmentation validation**: Validating LiTS label format (0, 1, 2)
- **Edge cases**: Missing directories, empty directories, mismatched files
- **File extensions**: Support for both .nii and .nii.gz files
- **Invalid labels**: Testing with invalid segmentation labels

## Test Features

### Common Test Features

All handler tests include:

1. **Dataset Structure Validation**: Verifies that required directories and files exist
2. **Subject List Extraction**: Tests the extraction of subject metadata
3. **File Matching**: Ensures proper matching between images and labels
4. **Error Handling**: Tests behavior with missing or invalid files
5. **Edge Cases**: Tests with empty directories, mismatched files, etc.

### Handler-Specific Features

#### Base Handler
- Tests the abstract DatasetHandler interface
- Validates proper inheritance and method implementation
- Tests attribute initialization and types

#### MSD Handler
- Tests `imagesTr/` and `labelsTr/` directory structure
- Validates file naming convention (e.g., `liver_001.nii`)

#### JIPMER Handler
- Tests phase-specific directory structure
- Validates phase suffixes (A, P, V)
- Tests all three imaging phases
- Validates separate liver and tumor mask files

#### LiTS Handler
- Tests `volumes/` and `segmentations/` directory structure
- Validates LiTS label format (0=background, 1=liver, 2=tumor)
- Tests label extraction functionality
- Validates segmentation file integrity

## Test Output

Tests provide detailed output including:

- ✅ **PASS**: Test completed successfully
- ❌ **FAIL**: Test failed
- ⚠️ **WARNING**: Non-critical issues detected
- 📊 **Summary**: Overall test results and statistics

## Example Output

```
============================================================
RUNNING ALL DATA HANDLER TESTS
============================================================

========================================
TESTING MSD HANDLER
========================================
Testing MSD Handler...

1. Testing dataset validation...
   Dataset validation: PASS

2. Testing subject list extraction...
   Found 5 subjects
   Subject: liver_000
     Image: /tmp/.../imagesTr/liver_000.nii
     Label: /tmp/.../labelsTr/liver_000.nii

MSD Handler Tests: ✅ PASSED
MSD Tests completed in 2.34 seconds

========================================
TESTING JIPMER HANDLER
========================================
Testing JIPMER Handler (Basic)...

1. Testing dataset validation...
   Dataset validation: PASS

2. Testing subject list extraction...
   Found 4 subjects
   Subject: 1A
     Image: /tmp/.../niigz dicom/Arterial Phase/Dliver1A.nii
     Liver: /tmp/.../niigz liver/Arterial Phase/LS1A.nii
     Tumor: /tmp/.../niigz tumor/Arterial Phase/TS1A.nii

JIPMER Handler Tests: ✅ PASSED
JIPMER Tests completed in 3.45 seconds

========================================
TESTING LITS HANDLER
========================================
Testing LiTS Handler (Basic)...

1. Testing dataset validation...
   Dataset validation: PASS

2. Testing subject list extraction...
   Found 3 subjects
   Subject: lits_0
     Image: /tmp/.../volumes/volume-0.nii
     Label: /tmp/.../segmentations/segmentation-0.nii

3. Testing label extraction...
   Liver mask shape: (256, 256, 128)
   Tumor mask shape: (256, 256, 128)
   Liver voxels: 1000000
   Tumor voxels: 64000
   Label extraction: PASS

LiTS Handler Tests: ✅ PASSED
LiTS Tests completed in 2.78 seconds

============================================================
TEST SUMMARY
============================================================

MSD Handler:
  ✅ basic: PASSED
  ✅ edge_cases: PASSED
  🎉 Overall: PASSED

JIPMER Handler:
  ✅ basic: PASSED
  ✅ phases: PASSED
  ✅ edge_cases: PASSED
  ✅ file_extensions: PASSED
  🎉 Overall: PASSED

LiTS Handler:
  ✅ basic: PASSED
  ✅ edge_cases: PASSED
  ✅ file_extensions: PASSED
  ✅ invalid_labels: PASSED
  🎉 Overall: PASSED

========================================
TOTAL: 12/12 tests passed
SUCCESS RATE: 100.0%
🎉 ALL HANDLERS PASSED ALL TESTS!
========================================
```

## Adding New Tests

To add tests for a new handler:

1. Create a new test file following the naming convention `test_<handler_name>_handler.py`
2. Implement test functions that return `True` for success, `False` for failure
3. Add the test functions to the `run_all_handler_tests.py` file
4. Update this README with information about the new tests

## Dependencies

Tests require the following Python packages:
- `numpy`: For array operations
- `nibabel`: For NIfTI file handling
- `tempfile`: For temporary directory creation
- `pathlib`: For path operations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in the Python path
2. **File Permission Errors**: Tests create temporary files, ensure write permissions
3. **Memory Issues**: Large test datasets may require significant memory

### Debug Mode

To run tests with more verbose output, modify the test functions to include additional print statements or use Python's logging module. 