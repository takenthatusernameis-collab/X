"""Asset-universe robustness sweep on synthetic data (walk-forward IS/OOS).

Demonstrates the asset-universe sweep from `research/backtest/universe`:

- A family of 7 synthetic assets sharing one regime structure but with
  drifting annual drifts from -6% to +12%.
- The MA crossover is run walk-forward across the whole family.
- The sweep reports per-asset OOS medians, a null (coin-flip) benchmark,
  and the verdict: CONSISTENT / CONCENTRATED / NO_EDGE.

Expected pattern on this family: the MA crossover has no robust edge, so the
sweep reports NO_EDGE and the medians are flat across the drift gradient.
A strategy whose edge lives in only the high-drift assets would be flagged
CONCENTRATED instead.

Research/simulation only. The data is synthetic and must not be
interpreted as evidence about live markets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from examples.ma_crossover import ma_crossover_signals


def main():
    np.random.seed(42)
    base_regimes = [
        bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
        bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
    ]
    drift_offsets = [-0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12]
    assets = bt.uniform_regime_assets(
        base_regimes, n_assets=7, drift_offsets=drift_offsets, seed=42, n_bars=1000
    )

    fast, slow = 20, 60
    grid = bt.parameter_grid_around((("fast", fast), ("slow", slow)))
    warmup = slow

    result = bt.sweep_across_assets(
        signals_fn=ma_crossover_signals,
        assets=assets,
        param_grid=grid,
        baseline=(("fast", fast), ("slow", slow)),
        train_window=252,
        test_window=84,
        warmup=warmup,
        overlap_window=60,
    )
    summary = bt.asset_sweep_summary(result, baseline=(("fast", fast), ("slow", slow)))
    print(summary.inspect())

    print()
    print(
        "Interpretation: if a strategy's edge is real and general, the sweep "
        "should be CONSISTENT across assets. A single-asset anomaly shows up "
        "as CONCENTRATED. On pure synthetic noise the tooling reports NO_EDGE."
    )
    print()
    print("NOTE: data is synthetic and intended only for tooling validation.")


if __name__ == "__main__":
    main()
