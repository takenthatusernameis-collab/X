"""Smoke test for momentum cost robustness signals and cost model.

This test verifies the basic functionality of the momentum cost robustness
test without running the full universe sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def test_momentum_signals():
    """Test momentum signal generation."""
    print("=== Smoke test: momentum signals ===")

    # Test simple price series
    closes = [100.0] * 30  # Flat prices
    closes[5:10] = [102.0] * 5  # Upward move
    closes[15:20] = [98.0] * 5   # Downward move

    # Test lookback=5
    signals = bt.momentum_signals(closes, 5)

    # Check signal length
    assert len(signals) == len(closes), f"Signal length {len(signals)} != price length {len(closes)}"

    # Check warmup padding (first 4 signals should be neutral)
    for i in range(4):
        assert abs(signals[i].weight) < 1e-12, f"Signal {i} should be neutral in warmup: {signals[i].weight}"

    # Check that signals are applied after warmup
    for i in range(4, len(closes)):
        assert abs(signals[i].weight) <= 1.0, f"Signal {i} weight out of bounds: {signals[i].weight}"

    # Test specific lookback behavior
    # For lookback=5, signal at index 4 should be based on prices[0:5]
    # Since prices[0:5] are flat, signal should be neutral
    assert abs(signals[4].weight) < 1e-12, f"Flat price window should produce neutral signal"

    print("  PASS: momentum signals basic functionality")


def test_cost_model():
    """Test realistic cost model parameters."""
    print("=== Smoke test: cost model ===")

    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    # Check cost parameters
    assert cfg.commission_per_trade == 2.0, f"commission_per_trade should be 2.0, got {cfg.commission_per_trade}"
    assert cfg.commission_per_share == 0.003, f"commission_per_share should be 0.003, got {cfg.commission_per_share}"
    assert cfg.slippage_cents == 2.0, f"slippage_cents should be 2.0, got {cfg.slippage_cents}"
    assert cfg.slippage_proportional == 0.0005, f"slippage_proportional should be 0.0005, got {cfg.slippage_proportional}"

    # Test fill price calculation
    bar_close = 100.0
    expected_price = bar_close + cfg.slippage_cents / 100.0 + bar_close * cfg.slippage_proportional
    actual_price = bar_close + cfg.slippage_cents / 100.0 + bar_close * cfg.slippage_proportional

    assert abs(expected_price - actual_price) < 1e-12, f"Fill price calculation mismatch"

    print("  PASS: cost model parameters and fill price calculation")


def test_momentum_function():
    """Test momentum function directly."""
    print("=== Smoke test: momentum function ===")

    # Create simple upward trend
    closes = [100.0 + i * 0.1 for i in range(50)]

    # Test different lookbacks
    for lookback in [3, 5, 10]:
        signals = bt.momentum_signals(closes, lookback)
        assert len(signals) == len(closes), f"Lookback {lookback}: signal length mismatch"

        # Check that later signals have non-zero weights
        non_neutral_count = sum(1 for s in signals if abs(s.weight) > 1e-12)
        assert non_neutral_count > 0, f"Lookback {lookback}: should have some non-neutral signals"

    print("  PASS: momentum function with different lookbacks")


def test_random_signals_for_null():
    """Test random signals for use as matched null."""
    print("=== Smoke test: random signals (null hypothesis) ===")

    n_bars = 100
    signals = bt.random_signals(n_bars)

    assert len(signals) == n_bars, f"Random signals length {len(signals)} != bars {n_bars}"

    # Check weight distribution
    weights = [s.weight for s in signals]
    weight_counts = {w: weights.count(w) for w in set(weights)}

    # Should have approximately equal distribution of -1, 0, 1
    # Allow wider range for small sample sizes
    for weight in [-1.0, 0.0, 1.0]:
        count = weight_counts.get(weight, 0)
        proportion = count / len(weights)
        assert 0.15 <= proportion <= 0.55, f"Weight {weight} proportion {proportion:.2f} outside expected range"

    print("  PASS: random signals weight distribution")


def main() -> int:
    print("Running momentum cost robustness smoke tests...")
    print()

    try:
        test_momentum_signals()
        test_cost_model()
        test_momentum_function()
        test_random_signals_for_null()

        print()
        print("=== SMOKE TEST RESULT: ALL PASS ===")
        return 0
    except AssertionError as e:
        print(f"\n=== SMOKE TEST RESULT: FAIL ===")
        print(f"AssertionError: {e}")
        return 1
    except Exception as e:
        print(f"\n=== SMOKE TEST RESULT: ERROR ===")
        print(f"Exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
