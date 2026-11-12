"""Inspect the existing momentum_signals implementation across the universe."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest


def main() -> int:
    manifest = load_manifest()
    tickers = manifest["universe"]
    for t in tickers:
        bars = bt.load_ticker(t)[0]
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        w = np.array([s.weight for s in sig])
        flips = int(np.sum(w[1:] != w[:-1]))
        vals, cnts = np.unique(w, return_counts=True)
        # correct momentum: sign of lookback return
        correct = np.zeros(len(closes))
        for i in range(5, len(closes)):
            r = closes[i] / closes[i - 5] - 1.0
            correct[i] = 1.0 if r > 0 else -1.0
        cflips = int(np.sum(correct[1:] != correct[:-1]))
        print("{:6s} min_close={:8.3f} max_close={:9.3f} weights={} flips={} | correct-lb5 flips={}".format(
            t, float(np.min(closes)), float(np.max(closes)),
            {float(v): int(c) for v, c in zip(vals, cnts)}, flips, cflips))
    return 0


if __name__ == "__main__":
    sys.exit(main())