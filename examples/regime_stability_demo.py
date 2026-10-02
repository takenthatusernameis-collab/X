"""Regime-stability demonstration on synthetic data.

This example demonstrates the regime-stability stress test from
`research/backtest/regime_stability`:

- A long-only rule on extreme up/neutral/down regimes swings from
  large edge to large drag -> REGIME_DEPENDENT.
- A direction-following rule (long up, short down) earns a positive edge
  in both regimes -> REGIME_STABLE.
- A coin-flip signal shows no edge anywhere -> CONSISTENT_WITH_NOISE.

The extreme regime families are deliberately severe (documented) so that
regime dependence is easily measurable; they are not realistic
parameterizations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def _coin_flip_signals(closes, **params):
    """Adapt coin-flip signals to the (closes, **params) calling convention."""
    return bt.random_signals(len(closes), seed=int(sum(params.values()) * 1000 + 42))


def main():
    bars = bt.generate_bars(600, seed=42)
    grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))

    print("=== Regime-stability stress test (deterministic seed 42) ===")
    print()

    print("--- Long-only rule on extreme regimes (expect REGIME_DEPENDENT) ---")
    result = bt.regime_stress(
        bt.always_long_signal, list(bars), grid,
        (("fast", 20.0), ("slow", 60.0)),
        scenarios=bt.stress_regime_scenarios(),
        train_window=60, test_window=20, warmup=0,
    )
    print(result.inspect())
    print()

    print("--- Direction-following rule on up/down regimes (expect REGIME_STABLE) ---")
    stress = bt.stress_regime_scenarios()
    up_down = [s for s in stress if s.name in ("strong_up", "strong_down")]
    result = bt.regime_stress(
        bt.direction_signal, list(bars), grid,
        (("fast", 20.0), ("slow", 60.0)),
        scenarios=up_down,
        train_window=60, test_window=20, warmup=0,
    )
    print(result.inspect())
    print()

    print("--- Coin-flip signal on the canonical regime family (expect CONSISTENT_WITH_NOISE) ---")
    result = bt.regime_stress(
        _coin_flip_signals, list(bars), grid,
        (("fast", 20.0), ("slow", 60.0)),
        scenarios=bt.canonical_regime_scenarios(42),
        train_window=60, test_window=20, warmup=0,
    )
    print(result.inspect())
    print()

    print("--- MA crossover on the canonical regime family (exploratory result) ---")
    from examples.ma_crossover import ma_crossover_signals

    result = bt.regime_stress(
        ma_crossover_signals, list(bars), grid,
        (("fast", 20.0), ("slow", 60.0)),
        scenarios=bt.canonical_regime_scenarios(42),
        train_window=60, test_window=20, warmup=0,
    )
    print(result.inspect())
    print()

    print("NOTE: data is synthetic and intended only for tooling validation.")


if __name__ == "__main__":
    main()
