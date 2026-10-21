"""Past-only volatility gate for momentum signals (lookback-5).

This is a predeclared volatility gate that filters momentum signals based on
past-only realized volatility. The gate is designed to test whether a single
volatility threshold creates predeclared robustness improvement for the
qualified momentum effect.

Evidence basis: MA regime filters were previously falsified; this tests whether
the same class of conditioning helps the different momentum mechanism.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt


def momentum_signals(closes: np.ndarray, lookback: int = 5):
    """Past-only lookback-5 momentum signal (long previous positive return, short negative)."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return signals


def volatility_gate_momentum_signals(
    closes: np.ndarray,
    lookback: int = 5,
    vol_window: int = 20,
    vol_threshold: float = 0.20,
):
    """Past-only volatility gate for momentum signals.

    Past-only realized volatility is the standard deviation of log returns
    over `vol_window` bars, scaled to an annual basis. Momentum signals are
    only emitted in low-vol regimes (realized vol below `vol_threshold`);
    neutral otherwise. This is a predeclared gate, not tuned to results.

    This predeclared gate addresses whether past-only volatility conditioning
    helps the momentum mechanism differently from the previously falsified MA
    regime filters.

    Args:
        closes: closing prices (past-only available at each bar)
        lookback: lookback window for momentum calculation (fixed at 5)
        vol_window: trailing bars for realized volatility estimation
        vol_threshold: annualized vol threshold below which momentum trades

    Returns:
        List of Signal with same length as closes
    """
    n = len(closes)
    base_signals = momentum_signals(closes, lookback)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)

    for i in range(int(vol_window) - 1, n):
        log_returns = np.log(closes[i - vol_window + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        if realized_annual < vol_threshold and abs(base_signals[i].weight) > 1e-12:
            signals[i] = bt.Signal(date=i + 1, weight=base_signals[i].weight)
        else:
            signals[i] = bt.Signal(date=i + 1, weight=0.0)

    return signals


def predeclare_volatility_gate():
    """Explicitly predeclare the volatility gate parameters before execution.

    Returns:
        dict with gate parameters that are fixed a-priori
    """
    return {
        "signal_type": "momentum_lookback_5",
        "vol_threshold": 0.20,  # 20% annualized vol threshold
        "vol_window": 20,       # 20-bar trailing window for vol estimation
        "lookback": 5,          # lookback for momentum (fixed per objective)
        "gate_type": "past_only_volatility_filter",
        "declaration_timestamp": "2026-10-21T14:00:00Z",
    }


if __name__ == "__main__":
    # Quick test to verify the gate works
    np.random.seed(42)
    bars = bt.generate_bars(2500, seed=42)
    closes = bars.closes_array()

    gate_params = predeclare_volatility_gate()
    print("=== Predeclared Volatility Gate ===")
    print(f"Signal type: {gate_params['signal_type']}")
    print(f"Volatility threshold: {gate_params['vol_threshold']:.2%}")
    print(f"Volatility window: {gate_params['vol_window']} bars")
    print(f"Lookback: {gate_params['lookback']} bars")
    print()

    print("=== Testing Gate on Synthetic Data (seed 42) ===")
    gated_signals = volatility_gate_momentum_signals(
        closes,
        lookback=gate_params["lookback"],
        vol_window=gate_params["vol_window"],
        vol_threshold=gate_params["vol_threshold"],
    )

    # Basic integrity check
    bt.check_signal_integrity(gated_signals, [b.date for b in bars], warmup=0)
    print("[PASS] signal integrity checks passed")

    # Full-sample performance
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=5)
    res = bt.run_bars(list(bars), gated_signals, cfg0)
    bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
    print(f"[PASS] equity matches fills")
    print(f"Full-sample: {len([f for f in res.trades if abs(f.shares) > 0])} trades")

    # Walk-forward validation
    result = bt.walk_forward(
        list(bars),
        gated_signals,
        train_window=252,
        test_window=84,
        warmup=5,
        overlap_window=60,
        cfg=cfg0,
    )
    print(f"Walk-forward: {result.aggregate_metrics['n_folds']} folds, "
          f"{result.aggregate_metrics['total_oos_periods']} OOS periods")

    print("\nGate validation complete. Ready for regime-stability stress testing.")
