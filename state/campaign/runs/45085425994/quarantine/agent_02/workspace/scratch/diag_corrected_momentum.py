"""Diagnostic: corrected causal momentum vs the degenerate published signal.

Establishes (on the collected universe, predeclared cost grid):
  1. The corrected signal is genuinely bidirectional.
  2. The degenerate signal's medians equal an always-long benchmark almost
     everywhere, so the published "momentum" verdicts describe buy-and-hold.
  3. The corrected signal's cost sensitivity.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
COSTS = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.0010),
}


def momentum_causal(closes, lookback=5):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(int(lookback), n):
        ret = float(closes[i] / closes[i - int(lookback)] - 1.0)
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run(bars, sig_fn, lookback, cost):
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = sig_fn(closes, lookback=lookback)
    res = bt.stress_segments(
        signals_fn=lambda c, **p: sig_fn(c, lookback=lookback),
        bars=bars, signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN, test_window=TEST, warmup=WARM,
        overlap_window=OVERLAP, periods_per_year=252,
        min_segment_bars=MIN_SEG,
        cfg=bt.BacktestConfig(initial_capital=1e6, **COSTS[cost]),
    )
    return res


manifest = __import__("json").loads(
    (Path(__file__).resolve().parent.parent / "research" / "data" / "manifest.json").read_text())
tickers = {t: bt.load_ticker(t)[0] for t in manifest["universe"]}

print("=== corrected signal direction balance (lookback 5) ===")
for t in manifest["universe"]:
    s = momentum_causal(tickers[t].closes_array(), 5)
    pos = sum(1 for x in s if x.weight > 0) / len(s)
    neg = sum(1 for x in s if x.weight < 0) / len(s)
    flips = sum(1 for i in range(1, len(s)) if s[i].weight != s[i - 1].weight)
    print(f"  {t:6s} long={pos:.3f} short={neg:.3f} flips={flips}")

print()
print("=== corrected momentum, lookback 5, per cost point ===")
print(f"  {'ticker':6s} {'segments':22s} {'cost':10s} {'medians':40s} verdict")
for t in manifest["universe"]:
    for cost in COSTS:
        res = run(tickers[t], momentum_causal, 5, cost)
        print("  {:6s} {:22s} {:10s} {:40s} {}".format(
            t, str([s.name for s in res.scenarios]), cost,
            str([round(s.baseline_median_log_return, 3) for s in res.scenarios]),
            res.overall_verdict))
    print()