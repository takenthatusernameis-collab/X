"""Diagnostic: turnover and net cost-inclusive medians for momentum vs null.

Answers: does the iid coin-flip null churn more than momentum, and what
happens to the candidate's own cost-inclusive per-segment median?
"""
import sys

import numpy as np

sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
from research.backtest.perturbation import random_signals

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400

COST = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.001),
}


def cfg(level):
    c = COST[level]
    return bt.BacktestConfig(initial_capital=1e6, **c)


def turnover(sig):
    return sum(1 for i in range(1, len(sig))
               if abs(sig[i].weight - sig[i - 1].weight) > 1e-12)


def seg_medians(ticker, lookback, level):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": lookback}], baseline=(("lookback", lookback),),
        segment_fn=seg_fn, train_window=TRAIN, test_window=TEST, warmup=WARM,
        overlap_window=OVERLAP, periods_per_year=252, min_segment_bars=MIN_SEG,
        cfg=cfg(level))
    return ([round(s.baseline_median_log_return, 4) for s in res.scenarios],
            [round(s.noise_median_log_return, 4) for s in res.scenarios],
            res.overall_verdict)


for ticker in ["AAPL", "NVDA", "TSLA", "AMZN"]:
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    print("==", ticker, "bars", len(closes))
    for lb in (3, 5, 10):
        mom = bt.momentum_signals(closes, lookback=lb)
        null = random_signals(len(closes), seed=42 + int(round(lb)) * 1000)
        print("  lb {} mom_turnover={} null_turnover={}".format(
            lb, turnover(mom), turnover(null)))
        for level in ("zero", "realistic", "stress2x"):
            cand, nul, v = seg_medians(ticker, lb, level)
            print("    {:9s} cand={} null={} verdict={}".format(level, cand, nul, v))