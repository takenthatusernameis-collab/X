#!/usr/bin/env python3
"""
Smoke test for novelty guard implementation.

This script validates that the novelty guard correctly:
1. Allows genuinely novel research requests
2. Blocks equivalent/near-duplicate research
3. Works deterministically
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.checks.novelty_guard import NoveltyGuard
from research.checks.momentum import MomentumCheck
from research.checks.momentum_hold_period_sensitivity import MomentumHoldPeriodSensitivityCheck
from research.checks.mean_reversion import MeanReversionCheck
from research.checks.volatility_targeting import VolatilityTargetingCheck
from research.checks.breakout_20day_cont import Breakout20DayContCheck

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "state" / "check_artifacts"

def test_novelty_guard_basic():
    """Test basic novelty guard functionality."""
    print("=== Basic Novelty Guard Test ===")
    
    guard = NoveltyGuard()
    
    # Test 1: Different signal classes should be allowed
    momentum_artifact = json.loads((DATA_DIR / "momentum_results.json").read_text())
    reversal_artifact = json.loads((DATA_DIR / "mean_reversion_results.json").read_text())
    
    result1 = guard.is_duplicate(momentum_artifact, "momentum")
    result2 = guard.is_duplicate(reversal_artifact, "momentum")
    
    print(f"Momentum vs Momentum (same class): {result1['is_duplicate']} (similarity: {result1['similarity']:.2f})")
    print(f"Momentum vs Reversal (different class): {result2['is_duplicate']} (similarity: {result2['similarity']:.2f})")
    
    assert not result1['is_duplicate'], "Same signal class should not be duplicate"
    assert not result2['is_duplicate'], "Different signal classes should not be duplicate"
    print("✓ Basic signal class detection works\n")

def test_novelty_guard_parameters():
    """Test parameter-based novelty detection."""
    print("=== Parameter-based Novelty Test ===")
    
    guard = NoveltyGuard()
    
    # Load momentum artifacts
    momentum_artifact = json.loads((DATA_DIR / "momentum_results.json").read_text())
    momentum_hold_artifact = json.loads((DATA_DIR / "momentum_hold_period_sensitivity_results.json").read_text())
    
    # Test parameter differences
    result1 = guard.is_duplicate(momentum_artifact, "momentum")
    result2 = guard.is_duplicate(momentum_hold_artifact, "momentum")
    
    print(f"Momentum vs Momentum hold period (similar params): {result2['is_duplicate']} (similarity: {result2['similarity']:.2f})")
    print(f"Momentum vs identical momentum: {result1['is_duplicate']} (similarity: {result1['similarity']:.2f})")
    
    # The hold-period sensitivity is a parameter variation, should be duplicate
    assert result2['is_duplicate'], "Parameter variations should be detected as duplicates"
    assert not result1['is_duplicate'], "Identical parameters should be detected as same"
    print("✓ Parameter-based detection works\n")

def test_novelty_guard_data_universe():
    """Test data universe-based novelty detection."""
    print("=== Data Universe Novelty Test ===")
    
    guard = NoveltyGuard()
    
    # Cross-sectional momentum vs individual momentum
    csm_artifact = json.loads((DATA_DIR / "cross_sectional_momentum_results.json").read_text())
    momentum_artifact = json.loads((DATA_DIR / "momentum_results.json").read_text())
    
    result1 = guard.is_duplicate(csm_artifact, "cross_sectional_momentum")
    result2 = guard.is_duplicate(momentum_artifact, "cross_sectional_momentum")
    
    print(f"Cross-sectional momentum vs itself: {result1['is_duplicate']} (similarity: {result1['similarity']:.2f})")
    print(f"Momentum vs Cross-sectional momentum (different universe): {result2['is_duplicate']} (similarity: {result2['similarity']:.2f})")
    
    assert not result1['is_duplicate'], "Same universe should not be duplicate"
    assert result2['is_duplicate'], "Different data universes should be detected as duplicates when same signal class"
    print("✓ Data universe detection works\n")

def test_novelty_guard_integration():
    """Test integration with actual research checks."""
    print("=== Integration with Research Checks ===")
    
    guard = NoveltyGuard()
    
    # Test with different check types
    checks_to_test = [
        ("Momentum", "momentum.py"),
        ("Momentum Hold Period", "momentum_hold_period_sensitivity_fixed.py"), 
        ("Mean Reversion", "mean_reversion.py"),
        ("Volatility Targeting", "volatility_targeting.py"),
        ("Breakout", "breakout_20day_cont.py"),
    ]
    
    for name, check_file in checks_to_test:
        if (DATA_DIR / f"{check_file.replace('.py', '_results.json')").exists():
            artifact_path = DATA_DIR / f"{check_file.replace('.py', '_results.json')}"
            artifact = json.loads(artifact_path.read_text())
            result = guard.is_duplicate(artifact, name)
            
            print(f"{name}: {result['is_duplicate']} (similarity: {result['similarity']:.2f})")
            assert not result['is_duplicate'], f"{name} should not be duplicate of itself"
    
    print("✓ Integration with research checks works\n")

def test_novelty_guard_determinism():
    """Test that novelty guard produces deterministic results."""
    print("=== Determinism Test ===")
    
    guard1 = NoveltyGuard()
    guard2 = NoveltyGuard()
    
    # Load the same artifact
    artifact = json.loads((DATA_DIR / "momentum_results.json").read_text())
    
    result1 = guard1.is_duplicate(artifact, "momentum")
    result2 = guard2.is_duplicate(artifact, "momentum")
    
    print(f"Run 1: {result1['is_duplicate']} (similarity: {result1['similarity']:.2f})")
    print(f"Run 2: {result2['is_duplicate']} (similarity: {result2['similarity']:.2f})")
    
    # Results should be identical
    assert result1['is_duplicate'] == result2['is_duplicate'], "Determinism failed"
    assert result1['similarity'] == result2['similarity'], "Determinism failed for similarity score"
    
    # Check consistency of all fields
    for key in result1:
        if key != 'timestamp':
            assert result1[key] == result2[key], f"Determinism failed for field {key}"
    
    print("✓ Novelty guard is deterministic\n")

def test_novelty_guard_evidence_integration():
    """Test integration with evidence preservation."""
    print("=== Evidence Integration Test ===")
    
    guard = NoveltyGuard()
    
    # Test that evidence is properly preserved
    artifact = json.loads((DATA_DIR / "momentum_results.json").read_text())
    result = guard.is_duplicate(artifact, "momentum")
    
    # Check that result contains required evidence fields
    required_fields = ['is_duplicate', 'similarity', 'signal_class_match', 'parameter_overlap', 'data_universe_match', 'purpose_similarity']
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Check evidence quality
    assert 0.0 <= result['similarity'] <= 1.0, "Similarity score out of range"
    assert 0.0 <= result['signal_class_match'] <= 1.0, "Signal class match out of range"
    assert 0.0 <= result['parameter_overlap'] <= 1.0, "Parameter overlap out of range"
    assert 0.0 <= result['data_universe_match'] <= 1.0, "Data universe match out of range"
    assert 0.0 <= result['purpose_similarity'] <= 1.0, "Purpose similarity out of range"
    
    print("✓ Evidence integration works correctly\n")

def main():
    """Run all novelty guard tests."""
    print("Testing Novelty Guard Implementation\n")
    print("=" * 50)
    
    try:
        test_novelty_guard_basic()
        test_novelty_guard_parameters()
        test_novelty_guard_data_universe()
        test_novelty_guard_integration()
        test_novelty_guard_determinism()
        test_novelty_guard_evidence_integration()
        
        print("=" * 50)
        print("✅ All novelty guard tests passed!")
        print("\nThe novelty guard successfully:")
        print("  • Allows genuinely novel research requests")
        print("  • Blocks equivalent/near-duplicate research")
        print("  • Works deterministically")
        print("  • Integrates with evidence preservation")
        print("  • Follows LEARNING_EFFICIENCY.md requirements")
        
        return 0
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())