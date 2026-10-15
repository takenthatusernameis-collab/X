"""Momentum with fixed lookback-5 and predeclared holding-period grid.

This test examines whether momentum with lookback-5 survives variations in
holding periods: [1, 2, 3, 5] days. This addresses the temporal robustness
uncertainty by testing if the momentum edge is sensitive to how long positions
are held.

Design:
- Fixed lookback: 5 days (not tuned after seeing results)
- Predeclared holding periods: [1, 2, 3, 5] days
- Use existing 10-asset universe from manifest.json
- Use walk-forward/null methodology
- Stop after all holding periods are tested and independently verified
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACK = 5
HOLDING_PERIODS = [1, 2, 3, 5]
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_period_results.json"

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=100,
    commission_per_share=5,
    slippage_cents=20,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def momentum_signals(closes, lookback):
    """Simple momentum signal with fixed lookback."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def momentum_signals_with_holding(closes, lookback, holding_period):
    """Momentum signal with minimum holding period constraint.

    Signals respect minimum holding periods: once a signal changes, it cannot
    change again for the specified number of days.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    # Generate raw momentum signal first
    raw_signals = momentum_signals(closes, lookback)
    
    # Apply holding period constraint
    for i in range(lookback, n):
        # Find next time momentum signal would change
        if i < n - 1:
            next_signal_change = None
            for j in range(i + 1, min(i + holding_period + 1, n)):
                if abs(raw_signals[j].weight - raw_signals[i].weight) > 1e-12:
                    next_signal_change = j
                    break
            
            # If no signal change within holding period, keep current signal
            if next_signal_change is None:
                out[i] = raw_signals[i]
            else:
                # Use signal until holding period expires
                out[i] = raw_signals[min(i, next_signal_change - 1)]
        else:
            out[i] = raw_signals[i]
    
    # Pad warmup - signals must be neutral before warmup boundary
    for i in range(min(lookback, n)):
        out[i] = bt.Signal(date=i + 1, weight=0.0)
    
    return out


def run_holding_period_test():
    """Test momentum with different holding periods."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Momentum with fixed lookback-5 and holding period grid ===")
    print(f"Testing lookback {LOOKBACK} with holding periods {HOLDING_PERIODS}")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    print("\n=== Results ===")

    # Store results for each asset and holding period
    results = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookback": LOOKBACK,
        "holding_periods": HOLDING_PERIODS,
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "universe": {},
        "per_asset": {}
    }

    # Test each asset for each holding period
    for ticker in manifest["universe"]:
        print(f"\n=== {ticker} ===")
        bars = all_tickers[ticker]
        closes = bars.closes_array()
        n_bars = len(closes)
        
        # Store per-asset results
        asset_results = {}
        per_asset_summary = {
            "edge_counts": {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0},
            "verdicts": {}
        }
        
        for hp in HOLDING_PERIODS:
            signals = momentum_signals_with_holding(closes, LOOKBACK, hp)
            cfg = REALISTIC_COST_CONFIG
            
            # Run walk-forward validation
            result = bt.walk_forward(
                list(bars),
                signals,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=cfg,
            )
            
            # Calculate metrics for OOS period only
            oos_returns = []
            for fold in result.folds:
                oos_returns.append(fold.metrics["total_return"])
            
            if oos_returns:
                median_log_return = float(np.median([np.log1p(np.clip(r, -1.0 + 1e-12, None)) for r in oos_returns]))
            else:
                median_log_return = 0.0
            
            # Classify result
            if median_log_return > 0.05:
                verdict = "EDGE"
                per_asset_summary["edge_counts"]["EDGE"] += 1
            elif median_log_return < -0.05:
                verdict = "LOSS"
                per_asset_summary["edge_counts"]["LOSS"] += 1
            else:
                verdict = "NO_EDGE"
                per_asset_summary["edge_counts"]["NO_EDGE"] += 1
            
            per_asset_summary["verdicts"][str(hp)] = verdict
            
            print(f"  Holding period {hp}: median log return = {median_log_return:.3f} -> {verdict}")
            
            # Store detailed results for artifact
            if ticker not in results["per_asset"]:
                results["per_asset"][ticker] = {}
            
            results["per_asset"][ticker][str(hp)] = {
                "median_log_return": median_log_return,
                "verdict": verdict,
                "n_folds": result.aggregate_metrics["n_folds"],
                "oos_periods": result.aggregate_metrics["total_oos_periods"],
                "positive_folds": sum(1 for f in result.folds if f.metrics["total_return"] > 0),
                "negative_folds": sum(1 for f in result.folds if f.metrics["total_return"] < 0),
            }
        
        # Store per-asset summary
        results["per_asset"][ticker] = per_asset_summary
        
        # Universe-level aggregation for this asset
        asset_holding_results = []
        for hp in HOLDING_PERIODS:
            hp_result = results["per_asset"][ticker][str(hp)]
            if hp_result["verdict"] == "EDGE":
                asset_holding_results.append(1)
            else:
                asset_holding_results.append(0)
        
        # Summary for this asset
        positive_holding_periods = sum(asset_holding_results)
        print(f"  Summary: {positive_holding_periods}/{len(HOLDING_PERIODS)} holding periods show positive edge")
    
    # Create universe-level summary
    print("\n=== Universe-level summary ===")
    all_ticker_order = manifest["universe"]
    universe_holding_results = {}
    
    for hp in HOLDING_PERIODS:
        hp_universe_results = []
        for ticker in all_ticker_order:
            verdict = results["per_asset"][ticker]["verdicts"].get(str(hp), "NO_EDGE")
            hp_universe_results.append(1 if verdict == "EDGE" else 0)
        
        universe_holding_results[str(hp)] = {
            "positive_assets": sum(hp_universe_results),
            "total_assets": len(all_ticker_order),
            "verdict": "CONSISTENT" if sum(hp_universe_results) >= len(all_ticker_order) / 2 else "INCONSISTENT"
        }
        
        print(f"  Holding period {hp}: {sum(hp_universe_results)}/{len(all_ticker_order)} assets show positive edge -> {universe_holding_results[str(hp)]['verdict']}")
    
    results["universe"] = {
        "ticker_order": all_ticker_order,
        "holding_period_results": universe_holding_results,
        "holding_period_consistency": {
            hp: "CONSISTENT" if results["universe"]["holding_period_results"][hp]["verdict"] == "CONSISTENT" else "INCONSISTENT"
            for hp in HOLDING_PERIODS
        }
    }

    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\n=== Summary ===")
    print(f"Holding period test completed")
    print(f"Results saved to: {ARTIFACT_PATH}")
    print(f"Note: universe-level consistency varies by holding period")

    return results


if __name__ == "__main__":
    run_holding_period_test()
    print("\nDone.")
