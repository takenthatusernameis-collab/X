from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Predeclared small holding-period grid (before execution, not tuned after inspection)
# These are "small" variations around the baseline 1-day hold to test robustness
HOLDING_PERIODS = [1, 2, 3, 5]

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_period_results.json"


def momentum_with_holding_period(closes, lookback=5, holding_period=1):
    """Momentum signal with configurable holding period.
    
    Long the previous lookback-day cumulative return, hold for holding_period days.
    
    Args:
        closes: array of closing prices (past-only)
        lookback: lookback window in bars for the return calculation
        holding_period: number of days to hold the position (1 = baseline)
    
    Returns:
        List[Signal] with signals assigned at signal bar, position held for holding_period days
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    for i in range(lookback, n):
        # Calculate cumulative log return over lookback period
        # This matches the standard momentum_signals: go long if sum(log returns) > 0, else short
        cum_log_ret = float(np.sum(np.log(closes[i - lookback + 1 : i + 1]) - np.log(closes[i - lookback])))
        
        # Assign position based on sign of cumulative return
        out[i] = bt.Signal(date=i + 1, weight=1.0 if cum_log_ret > 0 else -1.0)
    
    return out


def run_holding_period_check():
    """Test lookback-5 momentum across holding periods 1/2/3/5 days using walk-forward."""
    np.random.seed(SEED)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    print(f"Running holding-period robustness test for lookback=5 across {HOLDING_PERIODS}")
    
    # Build per-asset, per-holding-period results
    per_asset = {}
    
    for ticker, bars in tickers.items():
        print(f"  Processing {ticker}...")
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        asset_results = {}
        edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        lookback_medians = {}
        
        for hold_days in HOLDING_PERIODS:
            def create_signals(closes, **params):
                return momentum_with_holding_period(closes, lookback=5, holding_period=hold_days)
            
            signals = create_signals(closes)
            
            res = bt.stress_segments(
                signals_fn=create_signals,
                bars=list(bars),
                signals=signals,
                param_grid=[{"holding_period": hold_days}],
                baseline=(("holding_period", hold_days),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            
            # Extract results
            asset_results[str(hold_days)] = {
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "n_folds": res.n_folds,
                "null_dispersion": round(res.null_dispersion, 3),
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "segments": [s.name for s in res.scenarios],
                "verdict": res.overall_verdict,
            }
            
            # Count edge status for this holding period
            for s in res.scenarios:
                if abs(s.baseline_median_log_return) <= 0.05:  # EdgeFree threshold
                    edge_counts["NO_EDGE"] += 1
                elif s.baseline_median_log_return > 0:
                    edge_counts["EDGE"] += 1
                else:
                    edge_counts["LOSS"] += 1
            
            lookback_medians[str(hold_days)] = asset_results[str(hold_days)]["medians"]
        
        # Add edge_counts and lookback_medians to the asset results
        asset_results["edge_counts"] = edge_counts
        asset_results["lookback_medians"] = lookback_medians
        per_asset[ticker] = asset_results
    
    output = {
        "dataset_id": DATASET_ID,
        "holding_periods": HOLDING_PERIODS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "n_blocks": N_BLOCKS,
        "overlap": OVERLAP,
        "seed": SEED,
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "window": WINDOW,
        "ticker_order": list(tickers.keys()),
        "per_asset": per_asset,
        "task_contract": {
            "objective": "Test a fixed lookback-5 momentum signal with a predeclared small holding-period set using the existing walk-forward/null methodology.",
            "primary_question": "Does the qualified momentum effect survive a small predeclared holding-period variation without becoming a single-point timing artifact?",
            "scope": "Use only the existing 10-asset universe and a fixed hold-period grid declared before execution.",
            "out_of_scope": "Do not tune the lookback or choose a holding period after inspecting results.",
            "success_criterion": "The holding-period sensitivity is classified and any positive finding is reproducible.",
            "verification_requirement": "Matched null, leakage review, independent recomputation, determinism, regression suite.",
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {ARTIFACT_PATH}")
    
    # Print summary
    print("\n=== Momentum Holding Period Robustness Results ===")
    for ticker in tickers.keys():
        print(f"\n{ticker}:")
        for hold_days in HOLDING_PERIODS:
            result = per_asset[ticker][str(hold_days)]
            print(f"  Holding Period {hold_days} days: {result['verdict']} (median: {result['medians'][0]:.3f})")
    
    return output


if __name__ == "__main__":
    run_holding_period_check()
