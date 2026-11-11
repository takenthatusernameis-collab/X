import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400

ZERO = bt.BacktestConfig()
REAL = bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                         slippage_cents=2.0, slippage_proportional=0.0005)
CONS = bt.BacktestConfig(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.001)


def seg_medians(ticker, lookback, cfg):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    runs = []
    cur = labels[0]; start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= MIN_SEG:
                runs.append((cur, start, i))
            cur = labels[i]; start = i
    if len(labels) - start >= MIN_SEG:
        runs.append((cur, start, len(labels)))
    out = []
    for lab, s, e in runs:
        sig = bt.momentum_signals(closes, lookback=lookback)[s:e]
        res = bt.walk_forward(list(bars)[s:e], sig, train_window=TRAIN,
                              test_window=TEST, warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        lr = np.array([np.log1p(np.clip(f.metrics["total_return"], -1 + 1e-12, None))
                       for f in res.folds])
        out.append((lab, len(res.folds), float(np.mean(lr)), float(np.median(lr)),
                    float(np.min(lr)), float(np.max(lr))))
    return out


for ticker in ["AAPL", "NVDA"]:
    print("===", ticker, "lookback 5: per-segment mean/median log return by cost ===")
    for name, cfg in [("zero", ZERO), ("real", REAL), ("cons", CONS)]:
        rows = seg_medians(ticker, 5, cfg)
        print(" ", name)
        for lab, nf, mean, med, lo, hi in rows:
            print("    {:10s} folds={:2d} mean={:+.4f} median={:+.4f} min={:+.4f} max={:+.4f}".format(
                lab, nf, mean, med, lo, hi))

print()
print("=== Full-sample cost accounting (momentum lookback 5, whole series) ===")
for ticker in ["AAPL", "NVDA"]:
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    for name, cfg in [("zero", ZERO), ("real", REAL), ("cons", CONS)]:
        res = bt.run_bars(list(bars), sig, cfg)
        n_trades = len(res.trades)
        total_comm = float(sum(f.commission for f in res.trades))
        eq0 = float(res.equity_curve[0]); eq1 = float(res.equity_curve[-1])
        gross_units = float(sum(abs(f.shares) for f in res.trades))
        print("  {:6s} {:5s} trades={:5d} commission=${:,.0f} gross_shares={:,.0f} eq={:.4g}->{:.4g}".format(
            ticker, name, n_trades, total_comm, gross_units, eq0, eq1))

print()
print("=== Constant-long and random-signal cost accounting (AAPL) ===")
bars, _ = bt.load_ticker("AAPL")
closes = bars.closes_array()
n = len(closes)
long_sig = [bt.Signal(date=i + 1, weight=1.0) for i in range(n)]
for name, sig in [("const_long", long_sig),
                  ("random", bt.random_signals(n, seed=5042))]:
    for cname, cfg in [("zero", ZERO), ("real", REAL), ("cons", CONS)]:
        res = bt.run_bars(list(bars), sig, cfg)
        lr = np.log1p(np.clip([f for f in [res.equity_curve[-1] / res.equity_curve[0] - 1.0]],
                              -1 + 1e-12, None))
        print("  {:10s} {:5s} trades={:5d} commission=${:,.0f} total_ret={:+.4f}".format(
            name, cname, len(res.trades), float(sum(f.commission for f in res.trades)),
            float(res.equity_curve[-1] / res.equity_curve[0] - 1.0)))