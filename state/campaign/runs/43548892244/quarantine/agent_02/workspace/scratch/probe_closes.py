"""Read-only probe: inspect loaded closes and momentum signal weights."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

for ticker in ("AAPL", "JPM", "JNJ"):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    print(ticker, "n", len(closes), "first", closes[:3], "last", closes[-3:])
    print("  dates", dates[:2], dates[-2:])
    w = [s.weight for s in bt.momentum_signals(closes, 3)]
    print("  weights first 10", w[:10], "counts", {v: w.count(v) for v in set(w)})
    print("  close diffs sign counts",
          {int(np.sign(d)): int(c) for d, c in zip(*np.unique(np.sign(np.diff(closes)), return_counts=True))})