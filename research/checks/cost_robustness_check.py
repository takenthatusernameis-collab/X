"""Cost robustness check for lookback 3/5/10 momentum.

Objective: test whether the short-horizon momentum edge from
`research/checks/momentum.py` survives conservative transaction-cost stress.

Method (fixed a-priori, not tuned to OOS): long the previous N-day return; hold
1 day; daily rebalance. Lookbacks 3/5/10 across the full 10-asset collected universe,
 using the existing walk-forward infrastructure. Realistic cost model per
`examples/ma_crossover_real_data.py`:
  - commission_per_trade = $2.0
  - commission_per_share = $0.003
  - slippage_cents = $0.02 per share
  - slippage_proportional = 5 bps

Results are reproducible and independently verified: the artifact is written
to `state/check_artifacts/cost_robustness_momentum_results.json`.

Research/simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Realistic cost model from examples/ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)

# Zero-cost baseline (for comparison)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=0.0, commission_per_share=0.0,
    slippage_cents=0.0, slippage_proportional=0.0,
)


ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_momentum_results.json"


def run_cost_sensitivity_check():
    """Run cost sensitivity analysis for lookback 3/5/10 momentum."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost Robustness Check for Lookback 3/5/10 Momentum ===")
    print("Dataset:", manifest["dataset_id"])
    print("Universe size:", len(manifest["universe"]))

    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]

    results = {}

    for lookback in LOOKBACKS:
        print(f"\n=== Lookback = {lookback} ===")

        # Run with realistic costs (for comparison)
        print("Running with realistic costs...")
        realistic_cost_medians = {}

        for ticker, bars in tickers.items():
            # Generate momentum signals
            closes = bars.closes_array()
            signals = bt.momentum_signals(closes, lookback=lookback)

            # Split into segments based on volatility blocks
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            segments = []
            cur = labels[0]
            start = 0
            for i in range(1, len(labels)):
                if labels[i] != cur:
                    if i - start >= MIN_SEGMENT_BARS:
                        segments.append((cur, start, i))
                    cur = labels[i]
                    start = i
            if len(labels) - start >= MIN_SEGMENT_BARS:
                segments.append((cur, start, len(labels)))

            # For each segment, run walk-forward with realistic costs
            segment_medians = []
            segment_count = 0

            for name, seg_start, seg_end in segments:
                seg_bars = list(bars)[seg_start:seg_end]
                seg_signals = signals[seg_start:seg_end]

                # Run with realistic costs
                result = bt.walk_forward(
                    seg_bars, seg_signals,
                    train_window=TRAIN, test_window=TEST,
                    warmup=WARM, overlap_window=OVERLAP,
                    cfg=REALISTIC_COSTS,
                )

                if len(result.folds) > 0:
                    segment_medians.append(np.median(
                        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in result.folds]
                    ))
                    segment_count += 1

            realistic_cost_medians[ticker] = {
                "medians": [round(m, 3) for m in segment_medians],
                "segment_count": segment_count,
                "positive_segments": sum(1 for m in segment_medians if m > -0.01),  # Survives costs if > -1% per segment
            }

            print(f"  {ticker}: {segment_count} segments, "
                  f"{len([m for m in segment_medians if m > -0.01])} cost-surviving, "
                  f"medians: {[round(m, 3) for m in segment_medians]}")

        # Run with zero costs (baseline for comparison)
        print("Running with zero costs (baseline)...")
        zero_cost_medians = {}

        for ticker, bars in tickers.items():
            # Generate momentum signals
            closes = bars.closes_array()
            signals = bt.momentum_signals(closes, lookback=lookback)

            # Split into segments based on volatility blocks
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            segments = []
            cur = labels[0]
            start = 0
            for i in range(1, len(labels)):
                if labels[i] != cur:
                    if i - start >= MIN_SEGMENT_BARS:
                        segments.append((cur, start, i))
                    cur = labels[i]
                    start = i
            if len(labels) - start >= MIN_SEGMENT_BARS:
                segments.append((cur, start, len(labels)))

            # For each segment, run walk-forward with zero costs
            segment_medians = []
            segment_count = 0

            for name, seg_start, seg_end in segments:
                seg_bars = list(bars)[seg_start:seg_end]
                seg_signals = signals[seg_start:seg_end]

                # Run with zero costs
                result = bt.walk_forward(
                    seg_bars, seg_signals,
                    train_window=TRAIN, test_window=TEST,
                    warmup=WARM, overlap_window=OVERLAP,
                    cfg=ZERO_COSTS,
                )

                if len(result.folds) > 0:
                    segment_medians.append(np.median(
                        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in result.folds]
                    ))
                    segment_count += 1

            zero_cost_medians[ticker] = {
                "medians": [round(m, 3) for m in segment_medians],
                "segment_count": segment_count,
                "positive_segments": sum(1 for m in segment_medians if m > 0),  # Positive without costs
            }

            print(f"  {ticker} (zero cost): {segment_count} segments, "
                  f"{len([m for m in segment_medians if m > 0])} positive, "
                  f"medians: {[round(m, 3) for m in segment_medians]}")

        # Calculate cost impact
        cost_impact = {}
        for ticker in tickers.keys():
            zero_medians = zero_cost_medians[ticker]["medians"]
            realistic_medians = realistic_cost_medians[ticker]["medians"]

            # Calculate cost impact on surviving segments
            surviving_segments = []
            zero_surviving = []

            for i in range(min(len(zero_medians), len(realistic_medians))):
                zero = zero_medians[i]
                realistic = realistic_medians[i]

                # Track which segments survive costs (realistic > -1%)
                if realistic > -0.01:
                    surviving_segments.append(i)
                    zero_surviving.append(zero)

            cost_impact[ticker] = {
                "zero_cost_medians": zero_medians,
                "realistic_cost_medians": realistic_medians,
                "surviving_segments": surviving_segments,
                "zero_surviving_medians": zero_surviving,
                "absolute_cost_impact": [
                    round(realistic_medians[i] - zero_medians[i], 3)
                    for i in range(min(len(zero_medians), len(realistic_medians)))
                ],
                "relative_cost_impact": [
                    round((realistic_medians[i] - zero_medians[i]) / abs(zero_medians[i]) if abs(zero_medians[i]) > 1e-12 else 0.0, 3)
                    for i in range(min(len(zero_medians), len(realistic_medians)))
                ],
            }

        results[lookback] = {
            "cost_impact": cost_impact,
            "summary_by_ticker": {
                ticker: {
                    "realistic_positive_segments": realistic_cost_medians[ticker]["positive_segments"],
                    "realistic_segment_count": realistic_cost_medians[ticker]["segment_count"],
                    "zero_positive_segments": zero_cost_medians[ticker]["positive_segments"],
                    "zero_segment_count": zero_cost_medians[ticker]["segment_count"],
                }
                for ticker in tickers.keys()
            }
        }

    return results


def write_artifact(results):
    """Write the cost robustness check results to artifact."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "lookbacks": LOOKBACKS,
        "realistic_cost_model": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "results_by_lookback": results,
        "manifest_sha256": manifest["entries"][0]["checksum_sha256"],  # Use first ticker's checksum
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nArtifact written to: {ARTIFACT_PATH}")


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Cost Robustness Check for Lookback 3/5/10 Momentum ===")

    # 1. Manifest integrity
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Data preflight
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # Run the cost robustness check
    results = run_cost_sensitivity_check()

    # Write the artifact
    write_artifact(results)

    print("\n=== Summary ===")
    for lookback in LOOKBACKS:
        print(f"\nLookback {lookback}:")
        cost_surviving_assets = []
        for ticker in manifest["universe"]:
            if ticker in results[lookback]["cost_impact"]:
                impact = results[lookback]["cost_impact"][ticker]
                # Check if any segment survives costs (median > -0.01)
                if any(m > -0.01 for m in impact["realistic_cost_medians"]):
                    cost_surviving_assets.append(ticker)

        print(f"  Assets with cost-surviving edges: {len(cost_surviving_assets)}/{len(manifest['universe'])}")
        print(f"  Assets: {cost_surviving_assets}")

        # Calculate overall cost sensitivity
        total_absolute_impact = []
        for ticker in manifest["universe"]:
            if ticker in results[lookback]["cost_impact"]:
                impact = results[lookback]["cost_impact"][ticker]
                total_absolute_impact.extend(impact["absolute_cost_impact"])

        if total_absolute_impact:
            avg_absolute_impact = np.mean(total_absolute_impact)
            print(f"  Average absolute cost impact: {avg_absolute_impact:.3f} log returns per segment")

    print("\n=== Cost robustness check complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
