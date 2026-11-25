"""Diagnostic: does the cost cfg actually change momentum walk-forward results?

This is a narrow tool-integrity probe for the cost-sensitivity check. It runs a
single asset/segment momentum walk-forward at zero vs base cost and reports the
per-fold total returns and the aggregate median. If the medians are identical,
the cost cfg is not reaching the engine and the cost-sensitivity result is
invalid (must be repaired, not reported).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400

ZERO = dict(commission_per_trade=0.0, commission_per_share=0.0,
            slippage_cents=0.0, slippage_proportional=0.0)
BASE = dict(commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005)


def cfg(level):
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM, **level)


def run(asset, level):
    bars, _ = bt.load_ticker(asset)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=5)
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
        segment_fn=seg_fn, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg(level),
        periods_per_year=252, min_segment_bars=MIN_SEG)
    return [round(s.baseline_median_log_return, 6) for s in res.scenarios]


for asset in ["AAPL", "MSFT"]:
    z = run(asset, ZERO)
    b = run(asset, BASE)
    print(f"{asset}: zero={z}")
    print(f"{asset}: base={b}")
    print(f"{asset}: identical={z == b}")

# Direct single-fold probe: one asset, full-series walk_forward, zero vs base.
bars, _ = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
rz = bt.walk_forward(list(bars), sig, train_window=TRAIN, test_window=TEST,
                     warmup=WARM, overlap_window=OVERLAP, cfg=cfg(ZERO))
rb = bt.walk_forward(list(bars), sig, train_window=TRAIN, test_window=TEST,
                     warmup=WARM, overlap_window=OVERLAP, cfg=cfg(BASE))
print("full-series AAPL zero median:", round(rz.aggregate_metrics["median_log_total_return"], 6))
print("full-series AAPL base median:", round(rb.aggregate_metrics["median_log_total_return"], 6))
print("first 5 fold returns zero:", [round(f.metrics["total_return"], 6) for f in rz.folds[:5]])
print("first 5 fold returns base:", [round(f.metrics["total_return"], 6) for f in rb.folds[:5]])