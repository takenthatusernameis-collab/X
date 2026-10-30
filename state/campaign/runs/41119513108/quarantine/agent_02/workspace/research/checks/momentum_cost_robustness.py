"""Momentum cost-robustness check across lookbacks 3/5/10 using segment-based validation.

Tests whether the momentum edge observed in momentum_lookback_sweep survives
realistic transaction-cost stress.

Method (fixed a-priori, not tuned to OOS):
- Long the previous lookback-day return, hold 1 day, daily rebalance.
- Walk-forward per segment train=252d / test=84d / warmup=60d / overlap=60d.
- Matched null: same signal timing, random signs.
- Conservative realistic transaction costs from BacktestConfig.
- Deterministic recomputation for AMZN/JPM subset.

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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def generate_matched_null_signals(closes, base_signals, seed=42, lookback=None):
    """Generate matched null signals with same timing as base signals.
    
    For each position in base_signals where weight != 0, assign a random sign
    (-1 or 1) with equal probability, preserving the timing of signals.
    Uses a unique seed per (base_signals, lookback) combination to avoid collisions.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    # Create unique seed for this (base_signals, lookback) combination
    signals_hash = hash(tuple(sig.weight for sig in base_signals))
    combined_seed = hash((signals_hash, lookback, seed)) & 0x7FFFFFFF
    
    rng = np.random.default_rng(combined_seed)
    for i in range(n):
        if base_signals[i].weight != 0.0:
            # Assign random sign, preserving the timing
            out[i] = bt.Signal(date=i + 1, weight=float(rng.choice([-1.0, 1.0])))
    return out


def run_segment_validation(ticker, bars, lookback, cfg, use_momentum=True):
    """Run segment-based validation for one asset/lookback.
    
    Uses the same approach as momentum_lookback_sweep.py but with custom cost model.
    """
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Generate signals
    if use_momentum:
        signals = bt.momentum_signals(closes, lookback=lookback)
        signal_name = "momentum"
    else:
        signals = generate_matched_null_signals(closes, bt.momentum_signals(closes, lookback=lookback), seed=SEED, lookback=lookback)
        signal_name = "matched_null"
    
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    
    # Use the same parameter grid and baseline as momentum_lookback_sweep
    param_grid = [{"lookback": lookback}]
    baseline = (("lookback", lookback),)
    
    # Run stress_segments with our cost model
    result = bt.stress_segments(
        signals_fn=bt.momentum_signals if use_momentum else (
            lambda c, **kw: generate_matched_null_signals(c, bt.momentum_signals(c, lookback=lookback), seed=SEED, lookback=lookback)
        ),
        bars=bars,
        signals=signals,
        param_grid=param_grid,
        baseline=baseline,
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    # Extract key metrics from the result
    scen = result.scenarios[0]
    
    return {
        "signal_name": signal_name,
        "segments": [s.name for s in result.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in result.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in result.scenarios],
        "candidate_dispersion": round(result.candidate_dispersion, 3),
        "null_dispersion": round(result.null_dispersion, 3),
        "n_folds": result.n_folds,
        "verdict": result.overall_verdict,
        "baseline_median_log_return": round(scen.baseline_median_log_return, 3),
        "noise_median_log_return": round(scen.noise_median_log_return, 3),
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load manifest and verify
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Momentum Cost-Robustness Check ===")
    print(f"Universe: {len(manifest['universe'])} assets")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Cost model: conservative realistic")
    print(f"  - commission_per_trade: 0.0001")
    print(f"  - commission_per_share: 0.0001") 
    print(f"  - slippage_cents: 1.0")
    print(f"  - slippage_proportional: 0.001")
    print(f"  - initial_capital: 1e6")
    print(f"  - warmup_periods: {WARM}")
    print()
    
    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    
    ticker_order = list(tickers.keys())
    
    # Create cost model config
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=0.0001,
        commission_per_share=0.0001,
        slippage_cents=1.0,
        slippage_proportional=0.001,
        warmup_periods=WARM,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )
    
    # Results structure
    results = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": LOOKBACKS,
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_model": {
            "commission_per_trade": cfg.commission_per_trade,
            "commission_per_share": cfg.commission_per_share,
            "slippage_cents": cfg.slippage_cents,
            "slippage_proportional": cfg.slippage_proportional,
            "initial_capital": cfg.initial_capital,
            "target_exposure": cfg.target_exposure,
            "warmup_periods": cfg.warmup_periods,
        },
        "per_asset": {},
        "universe_summary": {},
    }
    
    # Run check for each asset and lookback
    print("=== Cost-Robustness Check Results ===")
    
    for ticker in ticker_order:
        ticker_results = {}
        for lookback in LOOKBACKS:
            # Run momentum with cost model
            momentum_result = run_segment_validation(ticker, tickers[ticker], lookback, cfg, use_momentum=True)
            
            # Run matched null with cost model
            null_result = run_segment_validation(ticker, tickers[ticker], lookback, cfg, use_momentum=False)
            
            # Calculate cost impact (difference in baseline medians)
            cost_impact = momentum_result["baseline_median_log_return"] - null_result["noise_median_log_return"]
            
            ticker_results[str(lookback)] = {
                "momentum": momentum_result,
                "matched_null": null_result,
                "cost_impact": round(cost_impact, 4),
                "edge_classification": "EDGE" if cost_impact > 0 else "NO_EDGE" if cost_impact == 0 else "LOSS",
            }
            
            # Print progress
            print(f"{ticker:6s} lookback={lookback:2d}: "
                  f"momentum={momentum_result['baseline_median_log_return']:6.3f}, "
                  f"null={null_result['noise_median_log_return']:6.3f}, "
                  f"edge={cost_impact:7.4f} ({ticker_results[str(lookback)]['edge_classification']})")
        
        results["per_asset"][ticker] = ticker_results
    
    # Universe-level aggregation
    universe_summary = {}
    lookback_edges = {lb: [] for lb in LOOKBACKS}
    
    for ticker, ticker_results in results["per_asset"].items():
        for lookback, result in ticker_results.items():
            lookback_edges[int(lookback)].append(result["cost_impact"])
    
    # Compute universe statistics
    universe_summary["lookback_edges"] = {}
    for lookback, edges in lookback_edges.items():
        if edges and not any(np.isnan(e) for e in edges):
            universe_summary["lookback_edges"][str(lookback)] = {
                "count": len(edges),
                "mean_edge": float(np.mean(edges)),
                "median_edge": float(np.median(edges)),
                "std_edge": float(np.std(edges)),
                "positive_count": sum(1 for e in edges if e > 0),
                "positive_pct": float(sum(1 for e in edges if e > 0) / len(edges) * 100),
            }
    
    results["universe_summary"] = universe_summary
    
    print()
    print("=== Universe Summary ===")
    for lookback, stats in universe_summary.get("lookback_edges", {}).items():
        print(f"Lookback {lookback}:")
        print(f"  Positive edges: {stats['positive_count']}/{stats['count']} "
              f"({stats['positive_pct']:.1f}%)")
        print(f"  Mean edge: {stats['mean_edge']:.4f} "
              f"(median: {stats['median_edge']:.4f}, std: {stats['std_edge']:.4f})")
    
    # Determinism test with AMZN/JPM subset
    print()
    print("=== Determinism Test ===")
    target = ["AMZN", "JPM"]
    determinism_results = {}
    
    # First run
    determinism_results["run1"] = {}
    for ticker in target:
        determinism_results["run1"][ticker] = {}
        for lookback in LOOKBACKS:
            determinism_results["run1"][ticker][str(lookback)] = \
                run_segment_validation(ticker, tickers[ticker], lookback, cfg, use_momentum=True)
    
    # Second run with same seed for true determinism
    determinism_results["run2"] = {}
    for ticker in target:
        determinism_results["run2"][ticker] = {}
        for lookback in LOOKBACKS:
            determinism_results["run2"][ticker][str(lookback)] = \
                run_segment_validation(ticker, tickers[ticker], lookback, cfg, use_momentum=True)
    
    # Compare results
    deterministic = True
    for ticker in target:
        for lookback in LOOKBACKS:
            key = str(lookback)
            run1 = determinism_results["run1"][ticker][key]
            run2 = determinism_results["run2"][ticker][key]
            
            # Compare key metrics (only those available in determinism test)
            for metric in ["baseline_median_log_return", "noise_median_log_return"]:
                if abs(run1[metric] - run2[metric]) > 1e-10:
                    deterministic = False
                    print(f"NON-DETERMINISTIC: {ticker} lookback={lookback} "
                          f"{metric}: {run1[metric]} != {run2[metric]}")
            
            # Also compare cost_impact if available
            if "cost_impact" in run1 and "cost_impact" in run2:
                if abs(run1["cost_impact"] - run2["cost_impact"]) > 1e-10:
                    deterministic = False
                    print(f"NON-DETERMINISTIC: {ticker} lookback={lookback} "
                          f"cost_impact: {run1['cost_impact']} != {run2['cost_impact']}")
    
    if deterministic:
        print("DETERMINISTIC: Both runs produce identical results")
    
    results["determinism_test"] = {
        "deterministic": deterministic,
        "target_assets": target,
        "runs": determinism_results
    }
    
    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print()
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. Momentum cost-robustness check: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())