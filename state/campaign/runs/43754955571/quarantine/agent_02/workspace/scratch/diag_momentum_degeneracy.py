"""Diagnostic: is `momentum_signals` a genuine momentum signal or a constant long?

Read-only probe of the framework signal definitions used by the momentum cell.
No artifact is written; this only measures the emitted weight series and the
sign of the quantity the signal thresholds on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

MANIFEST = json.loads(Path("research/data/manifest.json").read_text())


def main() -> int:
    print("signal definition under test:")
    print("  regime_stability.momentum_signals: ret = mean(log(closes[i-lb+1:i+1]))")
    print("  regime_stability.mean_reversion_signals: same quantity, negated sign")
    print("")
    for ticker in MANIFEST["universe"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        mom = bt.momentum_signals(closes, lookback=5)
        rev = bt.mean_reversion_signals(closes, lookback=5)
        w = np.array([s.weight for s in mom])
        wr = np.array([s.weight for s in rev])
        # Independent computation of the thresholded quantity.
        q = np.array([
            float(np.mean(np.log(closes[i - 4:i + 1]))) if i >= 5 else np.nan
            for i in range(len(closes))
        ])
        qsub = q[~np.isnan(q)]
        # Intended quantity: lookback log return (price ratio).
        r = np.array([
            float(np.log(closes[i] / closes[i - 5])) if i >= 5 else np.nan
            for i in range(len(closes))
        ])
        rsub = r[~np.isnan(r)]
        print("{:6s} n={:5d} price[min={:.2f},max={:.2f}] mom_weights +1:{:5d} -1:{:5d} 0:{:5d} "
              "| mean(logp) min={:+.3f} frac>0={:.3f} | mean(logret) frac>0={:.3f}".format(
                  ticker, len(closes), float(np.min(closes)), float(np.max(closes)),
                  int(np.sum(w > 0)), int(np.sum(w < 0)), int(np.sum(w == 0)),
                  float(np.min(qsub)), float(np.mean(qsub > 0)),
                  float(np.mean(rsub > 0))))
    return 0


if __name__ == "__main__":
    sys.exit(main())