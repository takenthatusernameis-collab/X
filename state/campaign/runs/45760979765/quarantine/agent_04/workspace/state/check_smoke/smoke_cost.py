"""Smoke test: does the realistic cost cfg change momentum results, and how
large is the turnover / runtime? Read-only research smoke helper.

Runs AMZN lookback=5 through stress_segments under zero, realistic (1x) and
conservative (2x) cost configs and prints per-segment medians + verdicts, plus
a full-sample turnover count for each cost level.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
LOOKBACK = 5

ZERO = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
REALISTIC = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)
CONSERVATIVE = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=4.0, commission_per_share=0.006,
    slippage_cents=4.0, slippage_proportional=0.0010,
)


def run(ticker, cfg):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=LOOKBACK)
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": LOOKBACK}],
        baseline=(("lookback", LOOKBACK),), segment_fn=seg_fn,
        train_window=TRAIN, test_window=TEST, warmup=WARM,
        overlap_window=OVERLAP, periods_per_year=252,
        min_segment_bars=MIN_SEG, cfg=cfg,
    )
    meds = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    nulls = [round(s.noise_median_log_return, 3) for s in res.scenarios]
    return meds, nulls, res.overall_verdict


def turnover(ticker, cfg):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=LOOKBACK)
    res = bt.run_bars(list(bars), sig, cfg)
    return len(res.trades)


t0 = time.time()
for name, cfg in (("zero", ZERO), ("realistic", REALISTIC), ("conservative", CONSERVATIVE)):
    meds, nulls, verdict = run("AMZN", cfg)
    print("{:12s} medians={} nulls={} verdict={}".format(name, meds, nulls, verdict))
    print("             full-sample fills={}".format(turnover("AMZN", cfg)))
print("elapsed {:.1f}s".format(time.time() - t0))
print("SMOKE DONE")