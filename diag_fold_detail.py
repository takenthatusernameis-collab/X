"""Deep dive: per-bar P&L for one fold of momentum vs reversal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt
import numpy as np

ticker = "AAPL"
bars, _ = bt.load_ticker(ticker)
closes = bars.closes_array()

def segments(closes, n_blocks, window, min_seg):
    labels = bt.volatility_blocks(closes, n_blocks=n_blocks, window=window)
    runs = []
    cur = labels[0]; rs = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - rs >= min_seg:
                runs.append((cur, rs, i))
            cur = labels[i]; rs = i
    if len(labels) - rs >= min_seg:
        runs.append((cur, rs, len(labels)))
    return runs

runs = segments(closes, 4, 60, 400)
train, test, warm, overlap = 252, 84, 60, 60

def walk_fold(ticker, sig_fn, s, e, train, test, warm):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    seg_signals = sig_fn(closes)[s:e]
    res = bt.walk_forward(seg_bars, seg_signals, train_window=train,
                          test_window=test, warmup=warm,
                          overlap_window=overlap,
                          cfg=bt.BacktestConfig(warmup_periods=warm))
    return res

# First segment (index 0)
lab, s, e = runs[0]
print("Segment {} bars {}-{} (local window length {})".format(lab, s, e, e-s))

res_m = walk_fold(ticker, lambda c: bt.momentum_signals(c, 5), s, e, train, test, warm)
res_r = walk_fold(ticker, lambda c: bt.mean_reversion_signals(c, 5), s, e, train, test, warm)

print("MOMENTUM folds:", len(res_m.folds))
print("REVERSAL folds:", len(res_r.folds))
print("MOMENTUM sample metrics keys:", res_m.folds[0].metrics.keys())
print("REVERSAL sample metrics keys:", res_r.folds[0].metrics.keys())
print()
print("MOMENTUM per-fold log returns:")
for i, f in enumerate(res_m.folds):
    m = f.metrics
    print("  f{}: logtr={:+.4f}  n_trades={}".format(i, m["total_return"], m.get("n_trades", "?")))
print()
print("REVERSAL per-fold log returns:")
for i, f in enumerate(res_r.folds):
    m = f.metrics
    print("  f{}: logtr={:+.4f}  n_trades={}".format(i, m["total_return"], m.get("n_trades", "?")))
print()
print("MOMENTUM fold medians: [{}]".format(
    ", ".join("{:+.3f}".format(np.log1p(f.metrics["total_return"])) for f in res_m.folds)))
print("REVERSAL fold medians: [{}]".format(
    ", ".join("{:+.3f}".format(np.log1p(f.metrics["total_return"])) for f in res_r.folds)))

# Compare the last fold equity paths bar by bar
last_m = res_m.folds[-1]
last_r = res_r.folds[-1]
m_eq = last_m.equity_curve
r_eq = last_r.equity_curve
print()
print("Last fold equity length: mom={}, rev={}".format(len(m_eq), len(r_eq)))
print("Last fold metric total_return: mom={:+.4f} rev={:+.4f}".format(
    last_m.metrics["total_return"], last_r.metrics["total_return"]))
# test segment is the LAST test_window bars
tw = 84
print("mom ratio test[-1]/test[0] (last 84 bars) = {:+.4f}".format(m_eq[-tw:][-1]/m_eq[-tw:][0]))
print("rev ratio test[-1]/test[0] (last 84 bars) = {:+.4f}".format(r_eq[-tw:][-1]/r_eq[-tw:][0]))
mom_daily = np.diff(m_eq[-tw:])/m_eq[-tw:][:-1]
rev_daily = np.diff(r_eq[-tw:])/r_eq[-tw:][:-1]
print("momentum daily P&Ls (last 10 of test seg):", np.round(mom_daily[-10:], 4))
print("reversal daily P&Ls (last 10 of test seg):", np.round(rev_daily[-10:], 4))
print("corr(mom_daily, rev_daily) ~ -1 if opposite:", np.corrcoef(mom_daily, rev_daily)[0,1])
print("sum(log(1+mom_daily)) =", np.sum(np.log1p(mom_daily)), " vs log(ratio) =", np.log(m_eq[-tw:][-1]/m_eq[-tw:][0]))
print("sum(log(1+rev_daily)) =", np.sum(np.log1p(rev_daily)), " vs log(ratio) =", np.log(r_eq[-tw:][-1]/r_eq[-tw:][0]))
