"""Smoke test for momentum_lookback_sweep_cost_robustness.

Verifies that the new cost-robustness check imports and runs minimally.
"""
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import research.backtest as bt


def test_momentum_import():
    """Test that momentum_lookback_sweep_cost_robustness imports correctly."""
    from research.checks.momentum_lookback_sweep_cost_robustness import (
        DATASET_ID,
        SEED,
        TRAIN,
        TEST,
        WARM,
        OVERLAP,
        WINDOW,
        N_BLOCKS,
        MIN_SEGMENT_BARS,
        LOOKBACKS,
        segment_result_with_costs,
    )
    assert DATASET_ID == "yf-ohlcv-universe-2009-to-2026-10-03"
    assert SEED == 42
    assert TRAIN == 252
    assert TEST == 84
    assert WARM == 60
    assert OVERLAP == 60
    assert WINDOW == 60
    assert N_BLOCKS == 4
    assert MIN_SEGMENT_BARS == 400
    assert LOOKBACKS == (3, 5, 10)


def test_segment_result_with_costs():
    """Test segment_result_with_costs function."""
    bars, _ = bt.load_ticker("AMZN")
    result = segment_result_with_costs("AMZN", bars, 5)
    assert "segments" in result
    assert "medians" in result
    assert "null_medians" in result
    assert "verdict" in result
    assert len(result["medians"]) == 4  # 4 volatility blocks
    assert len(result["null_medians"]) == 4


def test_lookback_verdict():
    """Test lookback_verdict helper function."""
    from research.checks.momentum_lookback_sweep_cost_robustness import lookback_verdict

    # Positive REGIME_STABLE -> EDGE
    sr = {"verdict": "REGIME_STABLE", "medians": [0.05, 0.06, 0.07, 0.08]}
    assert lookback_verdict(sr) == "EDGE"

    # Loss -> LOSS
    sr = {"verdict": "REGIME_STABLE_LOSS", "medians": [-0.01, -0.02, -0.03, -0.04]}
    assert lookback_verdict(sr) == "LOSS"

    # Other verdicts -> NO_EDGE
    for verdict in ["REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE"]:
        sr = {"verdict": verdict, "medians": [0.01, 0.02, 0.03, 0.04]}
        assert lookback_verdict(sr) == "NO_EDGE"


def test_robustness_verdict_costs():
    """Test robustness_verdict_costs helper function."""
    from research.checks.momentum_lookback_sweep_cost_robustness import robustness_verdict_costs

    # ROBUST case
    results = {
        "3": {"verdict": "REGIME_STABLE", "medians": [0.05, 0.06, 0.07, 0.08]},
        "5": {"verdict": "REGIME_STABLE", "medians": [0.04, 0.05, 0.06, 0.07]},
        "10": {"verdict": "REGIME_STABLE", "medians": [0.03, 0.04, 0.05, 0.06]},
    }
    verdict, counts = robustness_verdict_costs(results)
    assert verdict == "ROBUST"
    assert counts["EDGE"] == 3
    assert counts["NO_EDGE"] == 0

    # SENSITIVE case
    results = {
        "3": {"verdict": "REGIME_STABLE", "medians": [0.05, 0.06, 0.07, 0.08]},
        "5": {"verdict": "CONSISTENT_WITH_NOISE", "medians": [0.01, 0.02, 0.03, 0.04]},
        "10": {"verdict": "CONSISTENT_WITH_NOISE", "medians": [-0.01, -0.02, -0.03, -0.04]},
    }
    verdict, counts = robustness_verdict_costs(results)
    assert verdict == "SENSITIVE"
    assert counts["EDGE"] == 1

    # NO_EDGE case
    results = {
        "3": {"verdict": "REGIME_STABLE_LOSS", "medians": [-0.01, -0.02, -0.03, -0.04]},
        "5": {"verdict": "CONSISTENT_WITH_NOISE", "medians": [0.01, 0.02, 0.03, 0.04]},
        "10": {"verdict": "CONSISTENT_WITH_NOISE", "medians": [-0.01, -0.02, -0.03, -0.04]},
    }
    verdict, counts = robustness_verdict_costs(results)
    assert verdict == "NO_EDGE"
    assert counts["EDGE"] == 0


def test_check_imports():
    """Test that the check module imports correctly."""
    # This import test ensures the module loads without error.
    from research.checks.momentum_lookback_sweep_cost_robustness import main
    assert callable(main)


def test_verifier_imports():
    """Test that the verifier module imports correctly."""
    from research.checks.verify_momentum_lookback_sweep_cost_robustness import main
    assert callable(main)