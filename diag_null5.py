import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt
from research.backtest.real_data import load_ticker
from research.data.preflight import load_manifest

manifest = load_manifest()
assert manifest["dataset_id"] == "yf-ohlcv-universe-2009-to-2026-10-03"
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

check_spread = bt.csrs_spread_daily_returns(tickers, 20, 3, 3)
print("check spread[22:27]:", check_spread[22:27])

# Verifier's spread implementation
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

verifier_spread = spread(tickers, 20, 3, 3)
print("verifier spread[22:27]:", verifier_spread[22:27])
print("equal:", np.allclose(check_spread, verifier_spread))
