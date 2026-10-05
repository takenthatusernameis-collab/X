import numpy as np
import research.backtest as bt
import sys
sys.path.insert(0, "/home/runner/work/X/X/research/checks")
import smoke_cs_momentum as sm

scenario = bt.canonical_regime_scenarios()[0]
family = [bt.generate_bars(800, regimes=scenario.regimes,
                          p_transition=scenario.p_transition,
                          start_price=100.0, seed=42 + i * 1000)
          for i in range(6)]
cand = sm.fresh_spread(family, sm.LOOKBACK, sm.TOP_K, sm.BOTTOM_K)
fw = bt.cs_momentum_spread_family(family, sm.LOOKBACK, sm.TOP_K, sm.BOTTOM_K)

diff = np.where(cand != fw)
print("n diffs:", len(diff[0]))
for idx in diff[0][:40]:
    print(idx, "fresh:", cand[idx], "fw:", fw[idx])
print("cand[55:95]:", [round(float(x), 6) for x in cand[55:95]])
print("fw[55:95]  :", [round(float(x), 6) for x in fw[55:95]])
