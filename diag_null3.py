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

null_check = bt.csrs_null_spread_daily_returns(tickers, 20, 3, 3, 42)

def spread(tickers, lookback, top_k, bottom_k):
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        closes = {t: tb.closes_array()[:min_len] for t, tb in tickers.items()}
        rets = {t: float(closes[t][i] / closes[t][i - lookback] - 1) for t in closes}
        order = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = order[:top_k]
        shorts = order[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        leg = lambda t: float(closes[t][i + 1] / closes[t][i] - 1)
        daily_pnl[i] = np.mean([leg(t) for t in longs]) - np.mean([leg(t) for t in shorts])
    return daily_pnl

daily_pnl = spread(tickers, 20, 3, 3)
rng = np.random.default_rng(42)
signs = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(len(daily_pnl))])
null_verifier = daily_pnl * signs

diff = np.where(np.abs(null_check - null_verifier) > 1e-12)[0]
print("first diff index:", diff[0] if len(diff) else "none")
print("null_check at diff:", null_check[diff[:3]])
print("null_verifier at diff:", null_verifier[diff[:3]])
print("total differing elements:", len(diff))
