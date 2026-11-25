"""Diagnostic: do transaction costs actually change AMZN momentum fold returns?

Compares a single walk_forward run at zero vs conservative cost and prints
per-fold total returns and trade counts. If the candidate fold returns are
identical across cost levels, costs are not reaching the candidate path and the
cost-sensitivity result would be invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60

COST = {
    "zero": dict(),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
}


def main() -> int:
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    for name, cost in COST.items():
        cfg = bt.BacktestConfig(initial_capital=1e6, **cost)
        res = bt.walk_forward(list(bars), sig, train_window=TRAIN,
                              test_window=TEST, warmup=WARM,
                              overlap_window=OVERLAP, cfg=cfg)
        rets = [f.metrics["total_return"] for f in res.folds]
        trades = [f.metrics["n_trades"] for f in res.folds]
        print("{}: n_folds={} mean={:.6f} median={:.6f} first5_ret={} first5_trades={}".format(
            name, len(rets), float(np.mean(rets)), float(np.median(rets)),
            [round(r, 6) for r in rets[:5]], trades[:5]))
    # Full-sample single run to see fill-level cost impact.
    for name, cost in COST.items():
        cfg = bt.BacktestConfig(initial_capital=1e6, **cost)
        fr = bt.run_bars(list(bars), sig, cfg)
        total = (fr.equity_curve[-1] - 1e6) / 1e6
        n_fills = len(fr.trades)
        total_comm = sum(f.commission for f in fr.trades)
        print("full-sample {}: total_return={:.6f} n_fills={} total_commission={:.2f}".format(
            name, total, n_fills, total_comm))
    return 0


if __name__ == "__main__":
    sys.exit(main())