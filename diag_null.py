import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt
from research.backtest.real_data import load_ticker
from research.data.preflight import load_manifest

manifest = load_manifest()
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
assert manifest["dataset_id"] == DATASET_ID
tickers = {}
for e in manifest["entries"]:
    tickers[e["ticker"]] = load_ticker(e["ticker"])[0]
trunc = min(len(tb) for tb in tickers.values())
class _TruncatedTicker:
    def __init__(self, bars, n):
        self._bars = bars
        self._n = n
    def closes_array(self):
        return self._bars.closes_array()[:self._n]
    def __len__(self):
        return self._n
tickers = {t: _TruncatedTicker(tb, trunc) for t, tb in tickers.items()}

# Check's null helper (full series)
null_check = bt.csrs_null_spread_daily_returns(tickers, 20, 3, 3, 42)
print("check null[0:5]:", null_check[0:5])

# Verifier-style null: per-bar sign applied to the SPREAD
def spread(tickers, lookback, top_k, bottom_k):
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        rets = {t: float(tb.closes_array()[i] / tb.closes_array()[i - lookback] - 1)
                for t, tb in tickers.items()}
        order = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = order[:top_k]
        shorts = order[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        long_ret = float(np.mean([tb.closes_array()[i + 1] / tb.closes_array()[i] - 1
                                  for t, tb in tickers.items() if t in longs]))
        short_ret = float(np.mean([tb.closes_array()[i + 1] / tb.closes_array()[i] - 1
                                   for t, tb in tickers.items() if t in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl

daily_pnl = spread(tickers, 20, 3, 3)
rng = np.random.default_rng(42)
signs = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(len(daily_pnl))])
null_verifier = daily_pnl * signs
print("verifier null[0:5]:", null_verifier[0:5])
print("equal:", np.allclose(null_check, null_verifier))

# Verifier-style null per segment (as in verify_cross_sectional)
aapl_closes = tickers["AAPL"].closes_array()
def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        bar_vols.append(0.0 if i < window else
                        float(np.std(np.log(closes[i - window: i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = (float(np.median([v for v in bar_vols[s:e] if v > 0]))
                    if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels

def segments_from_labels(labels, min_segment_bars):
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        segments.append((cur, start, len(labels)))
    return segments

aapl_labels = vol_blocks(aapl_closes, 4, 60)
segs = segments_from_labels(aapl_labels, 400)
for name, s, e in segs:
    seg_pnl = daily_pnl[s:e].copy()
    seg_rng = np.random.default_rng(42)
    seg_signs = np.array([1.0 if seg_rng.random() < 0.5 else -1.0 for _ in range(len(seg_pnl))])
    seg_null = seg_pnl * seg_signs
    print(f"segment {name}: first 5 null = {seg_null[0:5]}")
