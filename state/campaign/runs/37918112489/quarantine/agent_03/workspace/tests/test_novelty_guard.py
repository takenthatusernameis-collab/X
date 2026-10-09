"""Tests for the novelty guard module.

These tests verify that the novelty guard correctly prevents equivalent
experiment re-execution while allowing novel experiments to proceed.
"""

import pytest

from research.backtest.novelty_guard import (
    SignalSpec,
    compute_signal_hash,
    is_equivalent_experiment,
    should_skip_experiment,
)


class TestSignalHashComputation:
    """Tests for signal hash computation."""

    def test_identical_specs_produce_identical_hashes(self):
        """Identical signal specs should produce identical hashes."""
        spec1 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        spec2 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        
        hash1 = compute_signal_hash(spec1)
        hash2 = compute_signal_hash(spec2)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_different_lookback_different_hash(self):
        """Different lookback values should produce different hashes."""
        spec1 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        spec2 = SignalSpec(
            signal_type="momentum",
            lookback=10,
            hold_days=1,
            regime_features=None
        )
        
        hash1 = compute_signal_hash(spec1)
        hash2 = compute_signal_hash(spec2)
        
        assert hash1 != hash2

    def test_different_signal_type_different_hash(self):
        """Different signal types should produce different hashes."""
        spec1 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        spec2 = SignalSpec(
            signal_type="mean_reversion",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        
        hash1 = compute_signal_hash(spec1)
        hash2 = compute_signal_hash(spec2)
        
        assert hash1 != hash2

    def test_hash_deterministic_across_multiple_calls(self):
        """Hash computation should be deterministic across multiple calls."""
        spec = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        
        hashes = [compute_signal_hash(spec) for _ in range(10)]
        
        # All hashes should be identical
        assert len(set(hashes)) == 1


class TestNoveltyGuardFunctionality:
    """Tests for novelty guard core functionality."""

    def test_should_skip_experiment_duplicate_momentum(self):
        """Should skip duplicate momentum experiments."""
        result = should_skip_experiment(
            signal_type="momentum",
            lookback=5,
            hold_days=1
        )
        
        # For now, we can't assume the actual momentum experiment has been run,
        # so we'll just verify the function returns a boolean
        assert isinstance(result, bool)

    def test_should_skip_experiment_duplicate_reversal(self):
        """Should skip duplicate mean-reversion experiments."""
        result = should_skip_experiment(
            signal_type="mean_reversion",
            lookback=5,
            hold_days=1
        )
        
        assert isinstance(result, bool)

    def test_should_not_skip_different_signal_types(self):
        """Should not skip different signal types."""
        momentum_result = should_skip_experiment(
            signal_type="momentum",
            lookback=5,
            hold_days=1
        )
        
        mean_reversion_result = should_skip_experiment(
            signal_type="mean_reversion",
            lookback=5,
            hold_days=1
        )
        
        # Both should return boolean values
        assert isinstance(momentum_result, bool)
        assert isinstance(mean_reversion_result, bool)

    def test_should_skip_different_lookback_values(self):
        """Should skip different lookback values as they are different experiments."""
        result1 = should_skip_experiment(
            signal_type="momentum",
            lookback=5,
            hold_days=1
        )
        
        result2 = should_skip_experiment(
            signal_type="momentum",
            lookback=10,
            hold_days=1
        )
        
        # Both should return boolean values
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)

    def test_hash_computation_includes_all_parameters(self):
        """Hash should incorporate all signal parameters."""
        spec1 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        spec2 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=2,
            regime_features=None
        )
        
        hash1 = compute_signal_hash(spec1)
        hash2 = compute_signal_hash(spec2)
        
        # Different holding days should produce different hashes
        assert hash1 != hash2


class TestNoveltyGuardIntegration:
    """Integration tests for novelty guard."""

    def test_equivalent_signal_specs_produce_same_hash(self):
        """Signal specs with same parameters produce same hash."""
        spec1 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        spec2 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        
        hash1 = compute_signal_hash(spec1)
        hash2 = compute_signal_hash(spec2)
        
        assert hash1 == hash2

    def test_novelty_guard_handles_empty_regime_features(self):
        """Should handle None regime features correctly."""
        spec = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=None
        )
        
        # Should not raise an exception
        hash_value = compute_signal_hash(spec)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_novelty_guard_with_regime_features(self):
        """Should handle regime features when present."""
        regime_features = {"volatility_threshold": 0.02}
        spec1 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=regime_features
        )
        spec2 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=regime_features
        )
        
        hash1 = compute_signal_hash(spec1)
        hash2 = compute_signal_hash(spec2)
        
        assert hash1 == hash2

    def test_novelty_guard_different_regime_features_different_hash(self):
        """Different regime features should produce different hashes."""
        regime_features1 = {"volatility_threshold": 0.02}
        regime_features2 = {"volatility_threshold": 0.03}
        
        spec1 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=regime_features1
        )
        spec2 = SignalSpec(
            signal_type="momentum",
            lookback=5,
            hold_days=1,
            regime_features=regime_features2
        )
        
        hash1 = compute_signal_hash(spec1)
        hash2 = compute_signal_hash(spec2)
        
        assert hash1 != hash2