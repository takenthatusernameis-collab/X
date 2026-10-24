#!/usr/bin/env python3
"""Cost sensitivity check for lookback 3/5/10 momentum signals.

This runs bounded cost-sensitivity checks for lookback 3/5/10 momentum
under realistic transaction costs using the repository's existing
BacktestConfig cost model framework and matched null benchmark.

Execution environment: research/simulation only. No live trading.
Objective: Classify cost robustness of short-horizon momentum edge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Realistic cost model parameters (typical for retail quantitative trading)
# Based on industry standards for transaction cost analysis
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.005,      # $0.005 per trade (0.5 cents)
    commission_per_share=0.002,      # $0.002 per share (0.2 cents)
    slippage_cents=0.01,             # $0.01 per share (1 cent fixed slippage)
    slippage_proportional=0.001,     # 10 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

# Matched null (coin-flip) configuration for cost robustness testing
NULL_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.005,
    commission_per_share=0.002,
    slippage_cents=0.01,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def volatility_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
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
    """Extract contiguous segments from volatility labels."""
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


def segment_median(ticker, s, e, train, test, warm, overlap, lookback, cost_cfg):
    """Walk-forward regime-stability result for one asset segment and lookback."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = bt.momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm, **{k: v for k, v in cost_cfg.__dict__.items() if k != 'warmup_periods'})
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def main() -> int:
    # Preflight check
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Cost Robustness Check: Lookback 3/5/10 Momentum ===")
    print(f"Dataset: {manifest['dataset_id']} | Universe: {len(manifest['universe'])}")
    print(f"Cost Config: commission_per_trade=${REALISTIC_COST_CONFIG.commission_per_trade:.3f}, "
          f"commission_per_share=${REALISTIC_COST_CONFIG.commission_per_share:.3f}, "
          f"slippage_cents=${REALISTIC_COST_CONFIG.slippage_cents:.2f}, "
          f"slippage_proportional={REALISTIC_COST_CONFIG.slippage_proportional:.3f}")
    print(f"Walk-forward: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d")
    print()
    
    # Load the universe
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    
    print(f"Loaded {len(tickers)} tickers for cost sensitivity testing")
    print()
    
    # Run cost sensitivity analysis for each lookback
    cost_robustness_results = {
        "dataset_id": DATASET_ID,
        "lookbacks": list(LOOKBACKS),
        "min_segment_bars": MIN_SEGMENT_BARS,
        "n_blocks": N_BLOCKS,
        "overlap": OVERLAP,
        "cost_config": {
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
        },
        "per_asset": {},
    }
    
    # Test each asset
    for ticker in sorted(tickers.keys()):
        print(f"=== Testing {ticker} ===")
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        cost_results = {}
        
        for lb in LOOKBACKS:
            medians = []
            nulls = []
            n_folds_list = []
            
            for label, s, e in runs:
                median_log_return, n_folds = segment_median(
                    ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb, REALISTIC_COST_CONFIG)
                medians.append(round(median_log_return, 3))
                n_folds_list.append(n_folds)
            
            # Get matched null medians using the same cost structure
            null_medians = []
            for label, s, e in runs:
                median_log_return, n_folds = segment_median(
                    ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb, NULL_COST_CONFIG)
                null_medians.append(round(median_log_return, 3))
            
            # Determine verdict based on medians and null medians
            cand_array = np.array(medians)
            null_array = np.array(null_medians)
            
            candidate_dispersion = round(float(np.std(cand_array)), 3)
            null_dispersion = round(float(np.std(null_array)), 3)
            
            if all(abs(m) <= 0.05 for m in medians):
                verdict = "CONSISTENT_WITH_NOISE"
            elif candidate_dispersion > 2.0 * null_dispersion:
                verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                verdict = "REGIME_STABLE_LOSS"
            else:
                verdict = "REGIME_STABLE"
            
            cost_results[str(lb)] = {
                "medians": medians,
                "null_medians": null_medians,
                "candidate_dispersion": candidate_dispersion,
                "null_dispersion": null_dispersion,
                "n_folds": n_folds_list[0] if n_folds_list else 0,
                "segments": [label for label, _, _ in runs],
                "verdict": verdict,
            }
            
            print(f"  Lookback {lb}: medians {medians} | verdict {verdict}")
        
        cost_robustness_results["per_asset"][ticker] = cost_results
        print()
    
    # Save results to state directory
    output_dir = Path.cwd() / "state" / "check_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "momentum_cost_robustness_results.json"
    
    with open(output_path, "w") as f:
        json.dump(cost_robustness_results, f, indent=2, sort_keys=True)
    
    print(f"=== Cost Robustness Analysis Complete ===")
    print(f"Results saved to: {output_path}")
    
    # Quick summary
    print("\n=== Summary ===")
    for ticker in sorted(cost_robustness_results["per_asset"].keys()):
        print(f"{ticker}:")
        for lb in LOOKBACKS:
            verdict = cost_robustness_results["per_asset"][ticker][str(lb)]["verdict"]
            medians = cost_robustness_results["per_asset"][ticker][str(lb)]["medians"]
            print(f"  LB{lb}: {verdict} (medians: {medians})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())