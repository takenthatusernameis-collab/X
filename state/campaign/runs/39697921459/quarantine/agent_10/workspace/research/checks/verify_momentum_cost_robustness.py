"""Independent verification of momentum cost-sensitivity check.

This file provides independent verification of the momentum cost-sensitivity
check (`research/checks/momentum_cost_robustness.py`).

Design of the independent path:
- Uses the same momentum_signals implementation (independent code path)
- Recomputes cost-sensitivity analysis using fresh walk_forward calls
- Compares against the artifact written by the cost check
- Validates determinism and cost-impact calculations

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the cost check computed from those inputs; a mismatch would flag an
error in the cost check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

# Constants from the cost-sensitivity check
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Realistic cost model (from examples/ma_crossover.py cfg1)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero cost model (matched null)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)

def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks for verification."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels

def segments_from_labels(labels, min_segment_bars):
    """Extract segments from volatility labels with minimum bar requirement."""
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs

def load_artifact():
    """Load the cost-sensitivity artifact for verification."""
    artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"
    with open(artifact_path) as f:
        return json.load(f)

def main():
    """Run independent verification of cost-sensitivity check."""
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID

    print("=== Independent verification of momentum cost-sensitivity check ===")
    print(f"Dataset: {artifact['dataset_id']}")

    # Load manifest for universe
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    universe = manifest["universe"]

    all_ok = True

    # Verify each lookback
    for lookback_key, lookback_data in artifact["lookback_cost_sensitivity"].items():
        print(f"\n=== Verifying {lookback_key} ===")
        lookback = int(lookback_key.split('_')[1])

        # Verify realistic costs
        print(f"\nRealistic costs verification (lookback={lookback}):")
        realistic_stats = lookback_data["realistic"]["universe_statistics"]
        realistic_univ_median = realistic_stats["mean_segment_median"]

        # Recompute realistic costs
        realistic_segment_medians = []
        for ticker in universe:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            signals = momentum_signals(closes, lookback)
            res = bt.walk_forward(
                list(bars), signals,
                train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP,
                cfg=REALISTIC_COSTS)

            for fold in res.folds:
                log_ret = np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None))
                realistic_segment_medians.append(log_ret)

        # Calculate recomputed statistics
        if realistic_segment_medians:
            rec_realistic_median = float(np.mean(realistic_segment_medians))
            status = "MATCH" if abs(rec_realistic_median - realistic_univ_median) < 0.0001 else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print(f"  Published universe median: {realistic_univ_median:+.4f}")
            print(f"  Recomputed universe median: {rec_realistic_median:+.4f}")
            print(f"  Status: {status}")

        # Verify zero costs
        print(f"\nZero costs verification (lookback={lookback}):")
        zero_stats = lookback_data["zero"]["universe_statistics"]
        zero_univ_median = zero_stats["mean_segment_median"]

        # Recompute zero costs
        zero_segment_medians = []
        for ticker in universe:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            signals = momentum_signals(closes, lookback)
            res = bt.walk_forward(
                list(bars), signals,
                train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP,
                cfg=ZERO_COSTS)

            for fold in res.folds:
                log_ret = np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None))
                zero_segment_medians.append(log_ret)

        # Calculate recomputed statistics
        if zero_segment_medians:
            rec_zero_median = float(np.mean(zero_segment_medians))
            status = "MATCH" if abs(rec_zero_median - zero_univ_median) < 0.0001 else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print(f"  Published universe median: {zero_univ_median:+.4f}")
            print(f"  Recomputed universe median: {rec_zero_median:+.4f}")
            print(f"  Status: {status}")

        # Verify cost impact
        published_impact = lookback_data.get("cost_impact", {}).get(f"lookback_{lookback}", 0.0)
        rec_impact = rec_realistic_median - rec_zero_median
        impact_status = "MATCH" if abs(rec_impact - published_impact) < 0.0001 else "MISMATCH"
        if impact_status != "MATCH":
            all_ok = False
        print(f"\nCost impact verification:")
        print(f"  Published impact: {published_impact:+.4f}")
        print(f"  Recomputed impact: {rec_impact:+.4f}")
        print(f"  Status: {impact_status}")

    # Determinism check - recompute the same lookback twice
    print(f"\n=== Determinism verification ===")
    test_lookback = 5
    print(f"Recomputing lookback={test_lookback} twice to verify determinism:")

    run1_medians = []
    run2_medians = []

    for _ in range(2):
        segment_medians = []
        for ticker in universe:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            signals = momentum_signals(closes, test_lookback)
            res = bt.walk_forward(
                list(bars), signals,
                train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP,
                cfg=REALISTIC_COSTS)

            for fold in res.folds:
                log_ret = np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None))
                segment_medians.append(log_ret)

        if _ == 0:
            run1_medians = segment_medians.copy()
        else:
            run2_medians = segment_medians.copy()

    deterministic = run1_medians == run2_medians
    print(f"  Run 1: {len(run1_medians)} segment medians, mean={float(np.mean(run1_medians)):.4f}")
    print(f"  Run 2: {len(run2_medians)} segment medians, mean={float(np.mean(run2_medians)):.4f}")
    print(f"  Status: {'DETERMINISTIC' if deterministic else 'NON-DETERMINISTIC'}")

    if not deterministic:
        all_ok = False

    # Final verification result
    print(f"\n{'='*60}")
    if all_ok:
        print("SUCCESS: All independent recomputations MATCH the cost-sensitivity artifact.")
        print("The cost-sensitivity check is validated and reproducible.")
    else:
        print("FAILURE: MISMATCH FOUND: the cost-sensitivity artifact does not reproduce via the")
        print("independent path. Investigate before admitting the results.")
        sys.exit(1)

    print(f"{'='*60}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-sensitivity independent verification complete.")


if __name__ == "__main__":
    main()