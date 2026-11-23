"""Blast-radius check: rerun the older momentum checks with the corrected
signal in-memory (no artifact overwrite) to see whether their verdicts change.

This is diagnostic only: it does not write any artifact. It tells us whether the
signal correction invalidates only the momentum admission or also the
longer-horizon and lookback-sweep cells.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "JNJ", "XOM"]


def run(bars, lookback, cost_kw=None):
    cost_kw = cost_kw or {}
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN, test_window=TEST, warmup=WARM,
        overlap_window=OVERLAP,
        cfg=bt.BacktestConfig(initial_capital=1e6, **cost_kw),
        periods_per_year=252, min_segment_bars=MIN_SEG)
    meds = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    return meds, res.overall_verdict


tickers = {t: bt.load_ticker(t)[0] for t in UNIVERSE}

for lb in (3, 5, 10, 20, 60):
    counts = {}
    pos_assets = []
    for t in UNIVERSE:
        meds, v = run(tickers[t], lb)
        counts[v] = counts.get(v, 0) + 1
        if v == "REGIME_STABLE" and all(m > 0 for m in meds):
            pos_assets.append(t)
    print(f"lookback {lb:2d}: {counts} positive_stable={pos_assets}")

print()
print("=== per-asset detail, lookback 5 (corrected, zero cost) ===")
for t in UNIVERSE:
    meds, v = run(tickers[t], 5)
    print(f"  {t:6s} {str(meds):34s} {v}")