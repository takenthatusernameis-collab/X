"""Diagnostic: NVDA null and candidate behavior at 0x/1x/2x costs."""
import sys

import numpy as np

sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
BASE = dict(commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005)


def cfg(mult):
    return bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=BASE["commission_per_trade"] * mult,
        commission_per_share=BASE["commission_per_share"] * mult,
        slippage_cents=BASE["slippage_cents"] * mult,
        slippage_proportional=BASE["slippage_proportional"] * mult,
    )


def segments_of(labels):
    segs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= MIN_SEGMENT_BARS:
                segs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= MIN_SEGMENT_BARS:
        segs.append((cur, start, len(labels)))
    return segs


def main():
    ticker = "NVDA"
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    segs = segments_of(labels)
    print(ticker, "segments:", segs)
    for lab, s, e in segs:
        seg_bars = list(bars)[s:e]
        seg_closes = closes[s:e]
        sig = bt.momentum_signals(seg_closes, lookback=5)
        print(f"\nsegment {lab} [{s},{e}) bars={len(seg_bars)} "
              f"close {seg_closes[0]:.2f}->{seg_closes[-1]:.2f} "
              f"weights={ {w: sum(1 for x in sig if x.weight == w) for w in (-1.0, 0.0, 1.0)} }")
        for mult in (0.0, 1.0, 2.0):
            res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                                  warmup=WARM, overlap_window=OVERLAP, cfg=cfg(mult))
            rets = [f.metrics["total_return"] for f in res.folds]
            med = float(np.median(np.log1p(np.clip(rets, -1 + 1e-12, None))))
            full = bt.run_bars(seg_bars, sig, cfg(mult))
            comm = sum(t.commission for t in full.trades)
            print(f"  cand {mult}: folds={len(rets)} med={med:+.4f} "
                  f"mean={np.mean(rets):+.4f} equity={full.equity_curve[-1]:.0f} comm={comm:.0f}")
        for mult in (0.0, 1.0, 2.0):
            noise = bt.noise_benchmark(
                seg_bars, param_grid=[{"lookback": 5}],
                train_window=TRAIN, test_window=TEST, warmup=0,
                overlap_window=0, periods_per_year=252, cfg=cfg(mult))
            print(f"  null {mult}: med={noise.baseline_median_log_return:+.4f} "
                  f"folds={noise.n_folds}")
        # Inspect the null signal weights and full-sample run at 0x vs 1x.
        n = len(seg_bars)
        rng = np.random.default_rng(42 + int(round(5.0) * 1000))
        weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1/3, 1/3, 1/3])
        print("  null weights:", {w: int(np.sum(weights == w)) for w in (-1.0, 0.0, 1.0)})
        null_sig = [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]
        for mult in (0.0, 1.0):
            full = bt.run_bars(seg_bars, null_sig, cfg(mult))
            comm = sum(t.commission for t in full.trades)
            print(f"  null full {mult}: equity={full.equity_curve[-1]:.0f} "
                  f"trades={len(full.trades)} comm={comm:.0f}")


if __name__ == "__main__":
    main()