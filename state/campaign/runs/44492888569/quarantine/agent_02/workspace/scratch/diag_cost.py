"""Diagnostic: compare candidate and null walk-forward results at 0x vs 1x costs."""
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


def main():
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
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
    print("segments:", segs)
    lab, s, e = segs[0]
    seg_bars = list(bars)[s:e]
    seg_closes = closes[s:e]
    sig = bt.momentum_signals(seg_closes, lookback=5)
    print("segment", lab, s, e, "bars", len(seg_bars), "signal weights",
          {w: sum(1 for x in sig if x.weight == w) for w in (-1.0, 0.0, 1.0)})
    for mult in (0.0, 1.0, 2.0):
        res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                              warmup=WARM, overlap_window=OVERLAP, cfg=cfg(mult))
        rets = [f.metrics["total_return"] for f in res.folds]
        med = float(np.median(np.log1p(np.clip(rets, -1 + 1e-12, None))))
        print(f"candidate mult={mult}: folds={len(rets)} median_log={med:+.4f} "
              f"mean={np.mean(rets):+.4f} first3={[round(r, 4) for r in rets[:3]]}")
        full = bt.run_bars(seg_bars, sig, cfg(mult))
        n_trades = len(full.trades)
        total_comm = sum(t.commission for t in full.trades)
        print(f"  full-sample: trades={n_trades} total_commission={total_comm:.2f} "
              f"equity_end={full.equity_curve[-1]:.2f}")
    for mult in (0.0, 1.0, 2.0):
        noise = bt.noise_benchmark(
            seg_bars, param_grid=[{"lookback": 5}],
            train_window=TRAIN, test_window=TEST, warmup=0,
            overlap_window=0, periods_per_year=252, cfg=cfg(mult))
        print(f"null mult={mult}: median={noise.baseline_median_log_return:+.4f} "
              f"folds={noise.n_folds}")
    full = bt.run_bars(seg_bars, sig, cfg(1.0))
    print("first 10 fills at 1x:")
    for f in full.trades[:10]:
        print(f"  date={f.date} shares={f.shares:+.2f} price={f.price:.4f} comm={f.commission:.4f}")


if __name__ == "__main__":
    main()