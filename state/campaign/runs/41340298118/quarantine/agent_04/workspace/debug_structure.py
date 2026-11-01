"""Debug script to check the structure of lookback_results dictionaries."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Realistic cost model from examples/ma_crossover_real_data.py
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
)

ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)


def debug_segment_result(ticker, bars, lookback, cfg, name):
    """Walk-forward regime-stability result for one ticker and one lookback with given costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )
    result = {
        "segments": [s.name for s in res.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "n_folds": res.n_folds,
        "verdict": res.overall_verdict,
    }
    print(f"{name} ticker {ticker} lookback {lookback}:")
    print(f"  lookback_results_zero_cost keys: {list(result.keys())}")
    print(f"  segment_result keys: {list(result.keys())}")
    print()
    return result


def main() -> int:
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Debug segment_result structure ===")
    
    # Just debug with one ticker and one lookback
    ticker = "AAPL"
    bars = bt.load_ticker(ticker)[0]
    
    # Check zero cost
    lookback_results_zero_cost = {lb: debug_segment_result(ticker, bars, lb, ZERO_COST_CONFIG, "ZERO_COST") for lb in LOOKBACKS}
    
    # Check realistic cost
    lookback_results_costs = {lb: debug_segment_result(ticker, bars, lb, REALISTIC_COST_CONFIG, "REALISTIC_COST") for lb in LOOKBACKS}
    
    print("=== Compare dictionaries ===")
    for lb in LOOKBACKS:
        print(f"LOOKBACK {lb}:")
        print(f"  lookback_results_zero_cost[{lb}] type: {type(lookback_results_zero_cost[lb])}")
        print(f"  lookback_results_zero_cost[{lb}].keys(): {list(lookback_results_zero_cost[lb].keys())}")
        print(f"  lookback_results_zero_cost[{lb}]['{lb}'] type: {type(lookback_results_zero_cost[lb][str(lb)])}")
        print()

if __name__ == "__main__":
    main()