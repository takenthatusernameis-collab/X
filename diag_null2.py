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

rng_check = np.random.default_rng(42)
rng_ver = np.random.default_rng(42)

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

# Walk bar by bar, replicating check-null order, and collect check-vs-verifier RNG draws
lookback = 20
n = trunc
for i in range(20, n - 1):
    closes = {t: tb.closes_array()[:n] for t, tb in tickers.items()}
    rets = {t: float(closes[t][i] / closes[t][i - lookback] - 1) for t, tb in tickers.items()}
    order = sorted(rets, key=lambda t: rets[t], reverse=True)
    longs = order[:3]
    shorts = order[-3:]
    if len(longs) < 3 or len(shorts) < 3:
        continue
    leg = lambda t: float(closes[t][i + 1] / closes[t][i] - 1)
    spread_val = float(np.mean([leg(t) for t in longs]) - np.mean([leg(t) for t in shorts]))
    coin_c = rng_check.random() < 0.5
    coin_v = rng_ver.random() < 0.5
    if i < 25 or (coin_c != coin_v):
        print(f"i={i}: coin_check={coin_c} coin_ver={coin_v} spread={spread_val:+.5f}")
