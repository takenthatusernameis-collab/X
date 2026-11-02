#!/usr/bin/env python3
"""Test script for enhanced novelty guard."""

import sys
from pathlib import Path

# Add research to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.checks.novelty_guard_enhanced import check_hypothesis_equivalence


def main():
    """Test the enhanced novelty guard."""
    print("Testing Enhanced Novelty Guard")
    print("============================")

    # Test 1: Check if momentum_lookback_sweep is flagged as duplicate
    test_artifact = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
    test_check = Path.cwd() / "research" / "checks" / "momentum_lookback_sweep.py"

    print(f"\nTest 1: Checking momentum_lookback_sweep")
    print(f"Artifact: {test_artifact.name}")
    print(f"Check file: {test_check.name}")

    result = check_hypothesis_equivalence(test_artifact, test_check)

    print(f"\nResult:")
    print(f"  Is duplicate: {result['is_duplicate']}")
    print(f"  Duplicate reason: {result['duplicate_reason']}")
    print(f"  Can proceed: {result['can_proceed']}")

    if result["is_duplicate"]:
        print(f"  ✓ PASS: Correctly identified as duplicate lookback 5 work")
    else:
        print(f"  ✗ FAIL: Should have been identified as duplicate")

    # Test 2: Check if momentum_cost_robustness would be allowed (it's lookback 3,5,10)
    print(f"\nTest 2: Checking momentum_cost_robustness")
    test_artifact2 = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

    result2 = check_hypothesis_equivalence(test_artifact2, test_check)

    print(f"\nResult:")
    print(f"  Is duplicate: {result2['is_duplicate']}")
    print(f"  Duplicate reason: {result2['duplicate_reason']}")
    print(f"  Can proceed: {result2['can_proceed']}")

    if not result2["is_duplicate"]:
        print(f"  ✓ PASS: Correctly identified as novel work (cost robustness test)")
    else:
        print(f"  ✗ FAIL: Should not have been identified as duplicate")

    # Summary
    print(f"\nSummary:")
    all_passed = True
    if result["is_duplicate"]:
        print(f"  ✓ Test 1 passed: Duplicate detection working")
    else:
        print(f"  ✗ Test 1 failed: Duplicate detection not working")
        all_passed = False

    if not result2["is_duplicate"]:
        print(f"  ✓ Test 2 passed: Novel work detection working")
    else:
        print(f"  ✗ Test 2 failed: Incorrectly flagging novel work")
        all_passed = False

    if all_passed:
        print(f"\n✓ ALL TESTS PASSED: Enhanced novelty guard is working correctly")
        return 0
    else:
        print(f"\n✗ SOME TESTS FAILED: Enhanced novelty guard needs fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
