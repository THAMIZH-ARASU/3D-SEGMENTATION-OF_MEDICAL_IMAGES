#!/usr/bin/env python3
"""
Comprehensive test runner for all data preprocessing handlers.

This script runs tests for all dataset handlers:
- MSD (Medical Segmentation Decathlon)
- JIPMER
- LiTS

Provides a summary of all test results.
"""

import sys
import time
from pathlib import Path

# Add the project root to the path so we can import the handlers
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.data_preprocessing.test_msd_handler import test_msd_handler, test_msd_handler_edge_cases
from tests.data_preprocessing.test_jipmer_handler import (
    test_jipmer_handler_basic, 
    test_jipmer_handler_phases, 
    test_jipmer_handler_edge_cases, 
    test_jipmer_handler_file_extensions
)
from tests.data_preprocessing.test_lits_handler import (
    test_lits_handler_basic, 
    test_lits_handler_edge_cases, 
    test_lits_handler_file_extensions, 
    test_lits_handler_invalid_labels
)
from tests.data_preprocessing.test_base_handler import (
    test_base_handler_interface,
    test_base_handler_inheritance,
    test_base_handler_attributes
)


def run_handler_tests():
    """Run all handler tests and return results."""
    print("=" * 60)
    print("RUNNING ALL DATA HANDLER TESTS")
    print("=" * 60)
    
    test_results = {}
    
    # MSD Handler Tests
    print("\n" + "=" * 40)
    print("TESTING MSD HANDLER")
    print("=" * 40)
    
    start_time = time.time()
    try:
        success1 = test_msd_handler()
        success2 = test_msd_handler_edge_cases()
        test_results['MSD'] = {
            'basic': success1,
            'edge_cases': success2,
            'overall': success1 and success2
        }
        print(f"\nMSD Handler Tests: {'✅ PASSED' if test_results['MSD']['overall'] else '❌ FAILED'}")
    except Exception as e:
        print(f"MSD Handler Tests: ❌ FAILED with exception: {e}")
        test_results['MSD'] = {'overall': False, 'error': str(e)}
    
    msd_time = time.time() - start_time
    print(f"MSD Tests completed in {msd_time:.2f} seconds")
    
    # JIPMER Handler Tests
    print("\n" + "=" * 40)
    print("TESTING JIPMER HANDLER")
    print("=" * 40)
    
    start_time = time.time()
    try:
        success1 = test_jipmer_handler_basic()
        success2 = test_jipmer_handler_phases()
        success3 = test_jipmer_handler_edge_cases()
        success4 = test_jipmer_handler_file_extensions()
        test_results['JIPMER'] = {
            'basic': success1,
            'phases': success2,
            'edge_cases': success3,
            'file_extensions': success4,
            'overall': success1 and success2 and success3 and success4
        }
        print(f"\nJIPMER Handler Tests: {'✅ PASSED' if test_results['JIPMER']['overall'] else '❌ FAILED'}")
    except Exception as e:
        print(f"JIPMER Handler Tests: ❌ FAILED with exception: {e}")
        test_results['JIPMER'] = {'overall': False, 'error': str(e)}
    
    jipmer_time = time.time() - start_time
    print(f"JIPMER Tests completed in {jipmer_time:.2f} seconds")
    
    # LiTS Handler Tests
    print("\n" + "=" * 40)
    print("TESTING LITS HANDLER")
    print("=" * 40)
    
    start_time = time.time()
    try:
        success1 = test_lits_handler_basic()
        success2 = test_lits_handler_edge_cases()
        success3 = test_lits_handler_file_extensions()
        success4 = test_lits_handler_invalid_labels()
        test_results['LiTS'] = {
            'basic': success1,
            'edge_cases': success2,
            'file_extensions': success3,
            'invalid_labels': success4,
            'overall': success1 and success2 and success3 and success4
        }
        print(f"\nLiTS Handler Tests: {'✅ PASSED' if test_results['LiTS']['overall'] else '❌ FAILED'}")
    except Exception as e:
        print(f"LiTS Handler Tests: ❌ FAILED with exception: {e}")
        test_results['LiTS'] = {'overall': False, 'error': str(e)}
    
    lits_time = time.time() - start_time
    print(f"LiTS Tests completed in {lits_time:.2f} seconds")
    
    # Base Handler Tests
    print("\n" + "=" * 40)
    print("TESTING BASE HANDLER")
    print("=" * 40)
    
    start_time = time.time()
    try:
        success1 = test_base_handler_interface()
        success2 = test_base_handler_inheritance()
        success3 = test_base_handler_attributes()
        test_results['Base'] = {
            'interface': success1,
            'inheritance': success2,
            'attributes': success3,
            'overall': success1 and success2 and success3
        }
        print(f"\nBase Handler Tests: {'✅ PASSED' if test_results['Base']['overall'] else '❌ FAILED'}")
    except Exception as e:
        print(f"Base Handler Tests: ❌ FAILED with exception: {e}")
        test_results['Base'] = {'overall': False, 'error': str(e)}
    
    base_time = time.time() - start_time
    print(f"Base Tests completed in {base_time:.2f} seconds")
    
    return test_results


def print_summary(test_results):
    """Print a comprehensive summary of all test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for handler_name, results in test_results.items():
        print(f"\n{handler_name} Handler:")
        if 'error' in results:
            print(f"  ❌ FAILED - Exception: {results['error']}")
            total_tests += 1
        else:
            for test_name, passed in results.items():
                if test_name != 'overall':
                    total_tests += 1
                    if passed:
                        passed_tests += 1
                        print(f"  ✅ {test_name}: PASSED")
                    else:
                        print(f"  ❌ {test_name}: FAILED")
            
            if results['overall']:
                print(f"  🎉 Overall: PASSED")
            else:
                print(f"  💥 Overall: FAILED")
    
    print(f"\n" + "=" * 40)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print(f"SUCCESS RATE: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "SUCCESS RATE: 0%")
    
    all_passed = all(results.get('overall', False) for results in test_results.values())
    if all_passed:
        print("🎉 ALL HANDLERS PASSED ALL TESTS!")
    else:
        print("❌ SOME TESTS FAILED")
    
    print("=" * 40)


def main():
    """Main function to run all handler tests."""
    try:
        test_results = run_handler_tests()
        print_summary(test_results)
        
        # Return appropriate exit code
        all_passed = all(results.get('overall', False) for results in test_results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 