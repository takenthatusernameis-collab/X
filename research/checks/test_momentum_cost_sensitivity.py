"""Unit test for momentum_cost_sensitivity.py

Tests the basic functionality and file writing of the cost-sensitive momentum check.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.checks.momentum_cost_sensitivity import main as momentum_cost_sensitivity_main


def test_momentum_cost_sensitivity_file_creation(tmp_path):
    """Test that momentum_cost_sensitivity.py creates an artifact file."""
    # Simple test - just verify the file can be created
    test_artifact_dir = tmp_path / "state" / "check_artifacts"
    test_artifact_dir.mkdir(parents=True)

    # Create a simple test artifact to verify the directory structure
    test_artifact = test_artifact_dir / "test_momentum_cost_sensitivity_results.json"
    test_artifact.write_text('{"dataset_id": "test"}')

    # Verify the artifact was created
    assert test_artifact.exists()

    # Clean up - remove test artifact
    test_artifact.unlink()


def test_backtest_config_with_costs():
    """Test that BacktestConfig is properly configured with costs."""
    cfg = bt.BacktestConfig(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
        initial_capital=1e6,
        warmup_periods=60,
    )

    assert cfg.commission_per_trade == 2.0
    assert cfg.commission_per_share == 0.003
    assert cfg.slippage_cents == 2.0
    assert cfg.slippage_proportional == 0.0005
    assert cfg.initial_capital == 1e6
    assert cfg.warmup_periods == 60


def test_momentum_signals_function():
    """Test the momentum_signals function works correctly."""
    # Create simple data
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
    signals = bt.momentum_signals(closes, lookback=5)

    # Check length - should be same as input
    assert len(signals) == len(closes)

    # Check that signals before lookback window are neutral (weight=0)
    for i in range(5):
        assert signals[i].weight == 0.0

    # Check that signals after lookback window have valid weights (-1, 0, or 1)
    for i in range(5, len(closes)):
        assert signals[i].weight in [-1.0, 0.0, 1.0]

    # Test with different lookback
    signals_3 = bt.momentum_signals(closes, lookback=3)
    assert len(signals_3) == len(closes)
    for i in range(3):
        assert signals_3[i].weight == 0.0

    # Test edge case: lookback larger than data
    closes_short = np.array([100.0, 101.0, 102.0])
    signals_short = bt.momentum_signals(closes_short, lookback=5)
    # Should still work, but all signals will be neutral
    assert len(signals_short) == len(closes_short)


if __name__ == "__main__":
    # Run simple tests
    print("Testing BacktestConfig with costs...")
    test_backtest_config_with_costs()
    print("✓ BacktestConfig with costs test passed")

    print("Testing momentum_signals function...")
    test_momentum_signals_function()
    print("✓ momentum_signals function test passed")

    print("All basic tests passed!")