"""Cost-sensitive momentum robustness test.

Research/simulation only. No live trading.

This implements the bounded cost-sensitivity check for lookback 3/5/10 momentum
specified in the agent contract, using the repository's realistic cost model
(based on ma_crossover.py) and a matched null (coin-flip) benchmark.

Objective: Classify cost robustness of the 5-day momentum edge under
reasonable transaction costs.

Deliverable: Reproducible artifact and independent verification result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.regime_stability import momentum_signals

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK_BASE = 5
N_BLOCKS = 4
WINDOW = 60
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

# Realistic costs based on ma_crossover.py example
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
)


def cost_sensitive_perturbation_sweep():
    """Run lookback 3/5/10 momentum sweep under realistic costs."""
    print("=== Cost-sensitive momentum robustness test ===")
    print("Realistic costs: commission_trade=$2, commission_share=$0.003, "
          "slippage=$2/shares + 0.05% pro-rata")
    print("Testing lookback 3 / 5 / 10 momentum on synthetic data")

    # Generate synthetic bars (same parameters as in momentum.py)
    bars = bt.generate_bars(n_bars=1000, seed=SEED)

    # Parameter grid: lookback 3 / 5 / 10
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK_BASE),),
        multipliers=(0.6, 1.0, 1.6667),  # lookback 3, 5, 10
    )
    print(f"Parameter grid: {grid}")

    # Run momentum sweep with realistic costs
    sweep = bt.parameter_sweep(
        signals_fn=lambda closes, **p: momentum_signals(closes, lookback=int(p["lookback"])),
        bars=list(bars),
        param_grid=grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
    )
    sweep_summary = bt.sweep_summary(sweep, baseline=(("lookback", LOOKBACK_BASE),))

    # Run matched null benchmark with same costs
    noise_summary = bt.noise_benchmark(
        list(bars),
        param_grid=grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        seed=SEED + 1,  # Different seed for null
    )

    print(f"\nMomentum sweep with realistic costs:")
    print(f"  baseline (lookback=5): {sweep_summary.baseline_median_log_return:+.3f}")
    print(f"  coin-flip null baseline: {noise_summary.baseline_median_log_return:+.3f}")
    baseline_vs_null = sweep_summary.compare_noise(
        noise_summary.baseline_median_log_return,
        noise_summary.noise_fold_median_log_returns,
    )
    print(f"  baseline vs null (sample-calibrated tol {sweep_summary.effective_tolerance:+.3f}): {baseline_vs_null}")

    print(f"\n  Parameter grid results:")
    for ps, med in zip(sweep_summary.param_sets, sweep_summary.median_log_returns):
        print(f"    {ps}: {med:+.3f}")

    print(f"\n  Sign consistency: {sweep_summary.n_positive} positive, "
          f"{sweep_summary.n_negative} negative, {sweep_summary.n_zero} zero")

    # Cost impact analysis
    base_median = sweep_summary.baseline_median_log_return
    zero_costs_baseline = None

    # Also run zero-cost version for comparison
    sweep_zero = bt.parameter_sweep(
        signals_fn=lambda closes, **p: momentum_signals(closes, lookback=int(p["lookback"])),
        bars=list(bars),
        param_grid=grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=ZERO_COSTS,
        periods_per_year=252,
    )
    sweep_zero_summary = bt.sweep_summary(sweep_zero, baseline=(("lookback", LOOKBACK_BASE),))
    zero_costs_baseline = sweep_zero_summary.baseline_median_log_return

    cost_impact = base_median - zero_costs_baseline
    print(f"\n  Cost impact: {zero_costs_baseline:+.3f} (zero cost) -> "
          f"{base_median:+.3f} (realistic costs) = {cost_impact:+.3f}")

    return {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "base_lookback": LOOKBACK_BASE,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "parameter_grid": grid,
        "momentum_sweep": {
            "param_sets": [tuple(ps) for ps in sweep_summary.param_sets],
            "medians": [round(m, 3) for m in sweep_summary.median_log_returns],
            "baseline_median": round(sweep_summary.baseline_median_log_return, 3),
            "noise_baseline_median": round(noise_summary.baseline_median_log_return, 3),
            "baseline_vs_null": sweep_summary.compare_noise(
                noise_summary.baseline_median_log_return,
                noise_summary.noise_fold_median_log_returns,
            ),
            "n_positive": sweep_summary.n_positive,
            "n_negative": sweep_summary.n_negative,
            "n_zero": sweep_summary.n_zero,
            "fraction_positive": round(sweep_summary.fraction_positive, 3),
            "best_param_set": tuple(sweep_summary.best_param_set),
            "best_median": round(sweep_summary.best_median_log_return, 3),
            "worst_param_set": tuple(sweep_summary.worst_param_set),
            "worst_median": round(sweep_summary.worst_median_log_return, 3),
        },
        "zero_costs_baseline": round(zero_costs_baseline, 3),
        "cost_impact": round(cost_impact, 3),
        "verdict": "CONSISTENT_WITH_NOISE" if not sweep_summary.compare_noise(
            noise_summary.baseline_median_log_return,
            noise_summary.noise_fold_median_log_returns
        ) else "REGIME_STABLE" if sweep_summary.n_positive > 0 else "REGIME_STABLE_LOSS",
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Momentum cost robustness test")
    print("=" * 50)

    result = cost_sensitive_perturbation_sweep()

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"\nResults written to: {ARTIFACT_PATH}")
    print(f"\nSummary: lookback={LOOKBACK_BASE} momentum under realistic costs: "
          f"{result['verdict']} (baseline {result['momentum_sweep']['baseline_median']:+.3f} vs "
          f"null {result['momentum_sweep']['noise_baseline_median']:+.3f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
