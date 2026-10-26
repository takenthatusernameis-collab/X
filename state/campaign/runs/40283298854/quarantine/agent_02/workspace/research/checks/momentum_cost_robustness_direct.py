#!/usr/bin/env python3
"""Direct cost-sensitivity analysis for momentum lookback 3/5/10.

This script tests the short-horizon momentum edge under realistic costs
by running the same walk-forward logic as the momentum check but with
the realistic cost model.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = [3, 5, 10]
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_direct_results.json"

# Realistic cost model based on test_engine.py unit tests
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.003,  # 3 basis points per share
    slippage_cents=5.0,           # 5 cents per share
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

# Zero-cost model (benchmark - same as momentum.py)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

def volatility_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks for direct use."""
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
    """Re-implementation of segment detection."""
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

def direct_momentum_cost_analysis():
    print("=== Direct Momentum Cost-Sensitivity Analysis ===")
    print(f"Lookback: {LOOKBACKS}")
    print(f"Realistic cost model: commission_per_share=0.003, slippage_cents=5.0")
    print(f"Zero-cost model: commission_per_share=0, slippage_cents=0")
    print(f"Test data: {DATASET_ID}")
    print()

    # Load manifest
    manifest = load_manifest()
    print("=== Manifest integrity ===")
    for entry in manifest["entries"]:
        print(f"  {entry['ticker']}: OK")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    # Run cost-sensitivity analysis
    cost_results = {}
    for lookback in LOOKBACKS:
        print(f"=== Lookback {lookback} days: momentum cost sensitivity ===")

        # Load all tickers
        tickers = {}
        for entry in manifest["entries"]:
            ticker = entry["ticker"]
            bars, _ = bt.load_ticker(ticker)
            tickers[ticker] = bars

        # Analysis per asset
        per_asset_results = {}

        for ticker in tickers:
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)

            # Generate noise benchmark for this lookback
            noise_bars = bt.generate_bars(600, seed=SEED)
            noise_signals = bt.momentum_signals(noise_bars.closes_array(), lookback=lookback)
            noise_res = bt.walk_forward(
                list(noise_bars), noise_signals, train_window=60, test_window=20,
                warmup=10, overlap_window=10, cfg=ZERO_COSTS)
            noise_medians = []
            for f in noise_res.folds:
                log_ret = np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                noise_medians.append(log_ret)
            noise_median = np.median(noise_medians) if noise_medians else 0.0

            # Momentum signals for this ticker
            signals = bt.momentum_signals(closes, lookback=lookback)

            # Run with zero costs (baseline)
            zero_cfg = bt.BacktestConfig(
                initial_capital=1e6,
                target_exposure=1.0,
                commission_per_trade=0.0,
                commission_per_share=0.0,
                slippage_cents=0.0,
                slippage_proportional=0.0,
                warmup_periods=0,
                risk_free=0.0,
                periods_per_year=252,
                margin_rate=0.0,
                margin_call_liquidate=True,
            )
            zero_res = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=zero_cfg)

            # Run with realistic costs
            real_cfg = bt.BacktestConfig(
                initial_capital=1e6,
                target_exposure=1.0,
                commission_per_trade=0.0,
                commission_per_share=0.003,
                slippage_cents=5.0,
                slippage_proportional=0.0,
                warmup_periods=0,
                risk_free=0.0,
                periods_per_year=252,
                margin_rate=0.0,
                margin_call_liquidate=True,
            )
            real_res = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=real_cfg)

            # Extract medians
            zero_medians = []
            real_medians = []

            for f in zero_res.folds:
                log_ret = np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                zero_medians.append(log_ret)

            for f in real_res.folds:
                log_ret = np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                real_medians.append(log_ret)

            # Calculate statistics
            zero_median = np.median(zero_medians) if zero_medians else 0.0
            real_median = np.median(real_medians) if real_medians else 0.0
            cost_impact = real_median - zero_median

            # Determine verdict based on conservative tolerance
            def get_verdict(medians, noise_median):
                if all(abs(m) <= 0.05 for m in medians):  # Conservative tolerance
                    return "CONSISTENT_WITH_NOISE"
                elif any(m > 2.0 * abs(noise_median) if noise_median != 0 else m > 0.05 for m in medians):
                    return "REGIME_DEPENDENT_OR_STABLE"
                elif all(m < 0 for m in medians):
                    return "REGIME_STABLE_LOSS"
                else:
                    return "REGIME_STABLE_POSITIVE"

            zero_verdict = get_verdict(zero_medians, noise_median)
            real_verdict = get_verdict(real_medians, noise_median)

            verdict_changed = zero_verdict != real_verdict

            per_asset_results[ticker] = {
                "lookback": lookback,
                "zero_cost_median": float(zero_median),
                "real_cost_median": float(real_median),
                "cost_impact_log_return": float(cost_impact),
                "zero_cost_verdict": zero_verdict,
                "real_cost_verdict": real_verdict,
                "verdict_changed": verdict_changed,
                "n_folds": len(zero_medians),
                "noise_median_log_return": float(noise_median),
                "zero_cost_mean": float(np.mean(zero_medians) if zero_medians else 0.0),
                "real_cost_mean": float(np.mean(real_medians) if real_medians else 0.0),
                "zero_cost_std": float(np.std(zero_medians, ddof=1) if len(zero_medians) > 1 else 0.0),
                "real_cost_std": float(np.std(real_medians, ddof=1) if len(real_medians) > 1 else 0.0),
            }

        # Summary for this lookback
        verdict_changes = sum(1 for r in per_asset_results.values() if r["verdict_changed"])
        total_assets = len(per_asset_results)
        verdict_preserved_pct = (total_assets - verdict_changes) / total_assets * 100 if total_assets > 0 else 0.0

        avg_cost_impact = np.mean([r["cost_impact_log_return"] for r in per_asset_results.values()]) if per_asset_results else 0.0
        max_cost_impact = max([abs(r["cost_impact_log_return"]) for r in per_asset_results.values()]) if per_asset_results else 0.0

        cost_results[f"lookback_{lookback}"] = {
            "lookback": lookback,
            "cost_model": {
                "commission_per_share": 0.003,
                "slippage_cents": 5.0,
            },
            "per_asset": per_asset_results,
            "summary": {
                "total_assets": total_assets,
                "verdict_changes": verdict_changes,
                "verdict_preserved_pct": round(verdict_preserved_pct, 2),
                "avg_cost_impact_log_return": round(avg_cost_impact, 6),
                "max_cost_impact_log_return": round(max_cost_impact, 6),
                "noise_median_log_return": round(noise_median, 6),
            },
        }

        print(f"  Lookback {lookback}: {total_assets} assets analyzed")
        print(f"    Verdicts changed by costs: {verdict_changes}/{total_assets} ({verdict_preserved_pct:.1f}% preserved)")
        print(f"    Average cost impact: {avg_cost_impact:.6f} log-return")
        print(f"    Max cost impact: {max_cost_impact:.6f} log-return")
        print()

    # Generate artifact
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "dataset_id": DATASET_ID,
        "lookbacks": LOOKBACKS,
        "cost_model": {
            "commission_per_share": 0.003,
            "slippage_cents": 5.0,
            "description": "Realistic cost model based on test_engine.py unit tests",
        },
        "zero_cost_model": {
            "commission_per_share": 0.0,
            "slippage_cents": 0.0,
        },
        "cost_sensitivity": cost_results,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print("=== Direct Cost-Sensitivity Analysis Complete ===")
    print(f"Artifact written to: {ARTIFACT_PATH}")
    print()
    print("Key findings:")
    for lb_key, results in cost_results.items():
        lb = results["lookback"]
        preserved = results["summary"]["verdict_preserved_pct"]
        impact = results["summary"]["avg_cost_impact_log_return"]
        print(f"  Lookback {lb}: {preserved:.1f}% verdicts preserved vs zero-cost")
        print(f"    Average log-return impact: {impact:.6f}")
        print(f"    Noise median: {results['summary']['noise_median_log_return']:.6f}")

    return 0

if __name__ == "__main__":
    sys.exit(direct_momentum_cost_analysis())
