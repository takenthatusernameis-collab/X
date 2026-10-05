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

i = 24
closes = {t: tb.closes_array()[:trunc] for t, tb in tickers.items()}
rets = {t: float(closes[t][i] / closes[t][i - 20] - 1) for t, tb in tickers.items()}
print("rets at i=24:", {t: round(r, 6) for t, r in rets.items()})
order = sorted(rets, key=lambda t: rets[t], reverse=True)
print("order desc:", order)
longs = order[:3]
shorts = order[-3:]
leg = lambda t: float(closes[t][i + 1] / closes[t][i] - 1)
spread_val = float(np.mean([leg(t) for t in longs]) - np.mean([leg(t) for t in shorts]))
print("spread at i=24:", spread_val)

rng_check = np.random.default_rng(42)
# replicate check-null loop exactly
for j in range(20, 25):
    j_closes = {t: tb.closes_array()[:trunc] for t, tb in tickers.items()}
    j_rets = {t: float(j_closes[t][j] / j_closes[t][j - 20] - 1) for t, tb in tickers.items()}
    j_order = sorted(j_rets, key=lambda t: j_rets[t], reverse=True)
    j_longs = j_order[:3]
    j_shorts = j_order[-3:]
    if len(j_longs) < 3 or len(j_shorts) < 3:
        continue
    j_leg = lambda t: float(j_closes[t][j + 1] / j_closes[t][j] - 1)
    j_spread = float(np.mean([j_leg(t) for t in j_longs]) - np.mean([j_leg(t) for t in j_shorts]))
    coin = rng_check.random() < 0.5
    print(f"j={j}: spread={j_spread:+.6f} coin={coin} null={j_spread if coin else -j_spread:+.6f}")
