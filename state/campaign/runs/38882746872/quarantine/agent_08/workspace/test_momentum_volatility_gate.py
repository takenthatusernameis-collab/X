#!/usr/bin/env python3
"""Test momentum_signals with past-only volatility gate through regime-stability framework.

This tests the primary question: Does a single past-only volatility gate 
create a predeclared robustness improvement for the qualified momentum effect?

The gate: Trade only in turbulent regimes (realized vol > 25% annualized).
The signal: lookback-5 momentum (long previous 5-day return, short otherwise).
The universe: 10 assets from research/data/manifest.json.
The method: regime-stability across contiguous volatility segments.

One volatility feature, one threshold rule declared before execution.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
WARMUP = 60
MOMENTUM_LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60


def main():
    np.random.seed(42)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight ===")
    from importlib import import_module
    pf = import_module("research.data.preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting run.")
        sys.exit(1)

    print("\n=== 3. Momentum signals with past-only volatility gate ===")
    print(f"  Gate: active_when_turbulent (realized vol > 25% annualized)")
    print(f"  Signal: momentum_signals with lookback={MOMENTUM_LOOKBACK}")
    print(f"  Method: conditional_signal wrapping with volatility condition")

    universe = bt.load_universe()
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARMUP)

    for ticker, (bars, dates) in universe.items():
        closes = bars.closes_array()
        
        # Create momentum signals with volatility gate
        from research.backtest.regime_stability import conditional_signal, momentum_signals, active_when_turbulent
        gated_momentum_signals = conditional_signal(
            momentum_signals, 
            active_when_turbulent
        )
        
        signals = gated_momentum_signals(closes, lookback=MOMENTUM_LOOKBACK)
        
        # Verify signals integrity
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        
        # Test full-sample equity matching
        res = bt.run_bars(list(bars), signals, cfg0)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        
        # Count non-neutral signals
        nonneutral = sum(1 for s in signals if abs(s.weight) > 1e-12)
        print(f"  {ticker}: {bars.n_bars} bars, {nonneutral} non-neutral signals")

    print("\n=== 4. Regime-stability stress test across universe (momentum with volatility gate) ===")
    baseline = (("lookback", MOMENTUM_LOOKBACK),)
    grid = [{"lookback": MOMENTUM_LOOKBACK}]

    # Create a wrapper that passes the lookback parameter to momentum_signals
    def gated_momentum_wrapper(closes, **params):
        from research.backtest.regime_stability import conditional_signal, momentum_signals, active_when_turbulent
        gated = conditional_signal(momentum_signals, active_when_turbulent)
        return gated(closes, **params)

    summary = bt.stress_segments_across_tickers(
        {t: b for t, (b, _d) in universe.items()},
        signals_fn=gated_momentum_wrapper,
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=4),
        baseline=baseline,
        param_grid=grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg0,
        min_segment_bars=400,
    )

    print(summary.inspect())

    print("\n=== 5. Interpretation ===")
    vc = summary.verdict_counts
    print(f"  Verdict counts: CONSISTENT_WITH_NOISE={vc['CONSISTENT_WITH_NOISE']}, "
          f"REGIME_STABLE={vc['REGIME_STABLE']}, REGIME_STABLE_LOSS={vc['REGIME_STABLE_LOSS']}, "
          f"REGIME_DEPENDENT={vc['REGIME_DEPENDENT']}")

    if vc["REGIME_STABLE"] > vc["CONSISTENT_WITH_NOISE"]:
        print("  A majority of assets show REGIME_STABLE behavior, suggesting the volatility gate "
              "creates a reproducible improvement in robustness for the momentum effect.")
    elif vc["REGIME_DEPENDENT"] > 0:
        print("  Some assets show REGIME_DEPENDENT behavior, suggesting the momentum effect "
              "depends on specific volatility regimes rather than being robust across all regimes.")
    else:
        print("  Most assets show CONSISTENT_WITH_NOISE across regime segments; "
              "the momentum effect with the volatility gate is indistinguishable from noise.")

    print("\n=== 6. Baseline null comparison ===")
    # For null comparison, we need to create synthetic bars with the same length
    # Run noise benchmark on the same gated signal framework
    from research.backtest.perturbation import noise_benchmark
    from research.backtest.data import generate_bars
    
    # Use the first ticker to get the series length
    first_ticker = list(universe.keys())[0]
    first_bars, _ = universe[first_ticker]
    n_bars = len(first_bars)
    
    # Generate synthetic bars for noise benchmark
    synthetic_bars = generate_bars(n_bars=n_bars, seed=99)
    
    noise_summary = noise_benchmark(
        bars=list(synthetic_bars),
        param_grid=grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg0,
        seed=99,
    )
    
    # Check if baseline is within tolerance of noise
    is_indistinguishable = summary.compare_noise(
        noise_summary.baseline_median_log_return,
        noise_summary.noise_fold_median_log_returns,
    )
    print(f"  Momentum with volatility gate baseline within {summary.effective_tolerance:.3f} of noise: {is_indistinguishable}")
    print(f"  Gate baseline median log return: {summary.baseline_median_log_return:+.3f}")
    print(f"  Gate noise benchmark median log return: {noise_summary.baseline_median_log_return:+.3f}")

    print("\n=== 7. Cross-reference ===")
    print("  MA crossover (MA 20/60) with volatility gate: exploratory evidence from prior activations")
    print("  Momentum without gate: exploratory evidence from prior activations")
    print("  This test: momentum with volatility gate (lookback=5, vol>25%) across 10 assets")

    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum with past-only volatility gate regime-stability on the collected universe:")
    print(f"  Gate condition: active_when_turbulent (early_vol > 0.25)")
    print(f"  Signal class: momentum_signals (lookback={MOMENTUM_LOOKBACK})")
    print(f"  Overall verdict: {summary.overall_verdict}")


if __name__ == "__main__":
    main()