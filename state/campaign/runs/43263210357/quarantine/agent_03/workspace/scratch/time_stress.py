"""Timing probe: how long does one stress_segments call take at one cost level?"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
REALISTIC = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM,
                              commission_per_trade=2.0, commission_per_share=0.003,
                              slippage_cents=2.0, slippage_proportional=0.0005)


def correct_momentum_signals(closes, lookback=5):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        out[i] = bt.Signal(date=i + 1, weight=1.0 if closes[i] > closes[i - lookback] else -1.0)
    return out


def one(bars, sig, cfg):
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    res = bt.stress_segments(signals_fn=bt.momentum_signals, bars=bars, signals=sig,
                             param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
                             segment_fn=seg_fn, train_window=TRAIN, test_window=TEST,
                             warmup=WARM, overlap_window=OVERLAP, cfg=cfg,
                             min_segment_bars=MIN_SEG)
    return res


def main():
    bars = bt.load_ticker("AAPL")[0]
    closes = bars.closes_array()
    for label, sig in (("existing", bt.momentum_signals(closes, 5)),
                       ("correct", correct_momentum_signals(closes, 5))):
        for cname, cfg in (("zero", bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)),
                           ("realistic", REALISTIC)):
            t0 = time.time()
            r = one(bars, sig, cfg)
            dt = time.time() - t0
            print(f"{label:9s} {cname:9s} {dt:6.2f}s  verdict={r.overall_verdict:22s} "
                  f"medians={[round(s.baseline_median_log_return,3) for s in r.scenarios]} "
                  f"null={[round(s.noise_median_log_return,3) for s in r.scenarios]}")


if __name__ == "__main__":
    main()