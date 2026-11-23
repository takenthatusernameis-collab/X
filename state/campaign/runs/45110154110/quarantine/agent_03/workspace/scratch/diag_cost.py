"""Diagnostic: quantify transaction-cost drag in the walk-forward path.

For AMZN lookback 5 (and a second asset), report:
- full-sample zero vs realistic: total return, n fills, total commissions,
  total traded notional (turnover proxy);
- a representative walk-forward fold's OOS segment: candidate log return at
  zero vs realistic, and the cost drag.

Purpose: confirm costs are materially applied in the same cfg path used by
stress_segments, and quantify how much they reduce a fold's log return. A
COST_ROBUST label is only credible if the per-fold drag is measured, not
assumed.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60


def summarize(label, bars, signals, cfg):
    res = bt.run_bars(list(bars), signals, cfg)
    eq = res.equity_curve
    total_return = (eq[-1] - cfg.initial_capital) / cfg.initial_capital
    n_fills = len(res.trades)
    total_comm = sum(f.commission for f in res.trades)
    # traded notional = sum |shares|*price
    traded_notional = sum(abs(f.shares) * f.price for f in res.trades)
    gross = cfg.initial_capital
    print("  {:10s}: total_return={:+.4f} n_fills={:4d} total_comm=${:,.0f} "
          "traded_notional=${:,.0f} ({:.1f}x capital)".format(
              label, total_return, n_fills, total_comm, traded_notional,
              traded_notional / gross))
    return total_return, total_comm, traded_notional


def main():
    for ticker in ["AMZN", "AAPL"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        print("=== {} ({} bars) full-sample, lookback 5 ===".format(ticker, bars.n_bars))
        zero = bt.BacktestConfig()
        real = bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                                 slippage_cents=2.0, slippage_proportional=0.0005)
        rz = summarize("zero", bars, sig, zero)
        rr = summarize("realistic", bars, sig, real)
        print("  -> realistic minus zero total_return = {:+.4f}; extra commissions = ${:,.0f}".format(
            rr[0] - rz[0], rr[1] - rz[1]))
        print()

    # Representative walk-forward folds: cost drag on OOS test segments.
    print("=== Walk-forward per-fold OOS drag (AMZN, lookback 5) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    zero = bt.BacktestConfig(warmup_periods=WARM)
    real = bt.BacktestConfig(warmup_periods=WARM, commission_per_trade=2.0,
                             commission_per_share=0.003, slippage_cents=2.0,
                             slippage_proportional=0.0005)
    rz = bt.walk_forward(list(bars), sig, TRAIN, TEST, WARM, OVERLAP, cfg=zero)
    rr = bt.walk_forward(list(bars), sig, TRAIN, TEST, WARM, OVERLAP, cfg=real)
    drags = []
    for fz, fr in zip(rz.folds, rr.folds):
        tz = np.log1p(max(fz.metrics["total_return"], -1 + 1e-12))
        tr = np.log1p(max(fr.metrics["total_return"], -1 + 1e-12))
        drags.append(tz - tr)
    drags = np.array(drags)
    print("  n_folds={}  mean OOS drag={:.4f}  median drag={:.4f}  max drag={:.4f}".format(
        len(drags), float(np.mean(drags)), float(np.median(drags)), float(np.max(drags))))
    print("  drag as fraction of |zero-cost fold log ret| (median): {:.2%}".format(
        float(np.median(drags)) / max(abs(float(np.median([np.log1p(max(f.metrics['total_return'], -1+1e-12)) for f in rz.folds]))), 1e-9)))


if __name__ == "__main__":
    main()