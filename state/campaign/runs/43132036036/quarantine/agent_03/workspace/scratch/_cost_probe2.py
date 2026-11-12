"""Diagnose why candidate medians appear cost-insensitive."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
REALISTIC = dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005)


def main() -> int:
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    flips = sum(1 for i in range(1, len(sig)) if sig[i].weight != sig[i - 1].weight)
    print("bars={} momentum flips={}".format(len(bars), flips))
    for name, cfg in [("zero", bt.BacktestConfig()), ("realistic", bt.BacktestConfig(**REALISTIC))]:
        res = bt.walk_forward(list(bars), sig, train_window=TRAIN, test_window=TEST,
                              warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        rets = [f.metrics["total_return"] for f in res.folds]
        trades = [f.metrics["n_trades"] for f in res.folds]
        log = np.log1p(np.clip(rets, -1 + 1e-12, None))
        print("{}: median_log={:.4f} mean_log={:.4f} rets[:4]={} ntrades={}".format(
            name, float(np.median(log)), float(np.mean(log)),
            [round(r, 4) for r in rets[:4]], trades[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())