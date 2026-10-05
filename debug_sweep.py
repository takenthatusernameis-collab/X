"""Debug: reproduce the check and verifier sweep-family paths and compare per-scenario medians."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

SEED = 42
LOOKBACK, TOP_K, BOTTOM_K = 63, 3, 3

offsets = [-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.06, 0.08, 0.10]

def family_via_functional(n_assets=10, scenario=None, base_seed=SEED):
    """Check's make_synthetic_family."""
    family = []
    for i in range(n_assets):
        regimes = [bt.Regime(
            drift_annual=float(r.drift_annual + offsets[i]),
            vol_annual=float(r.vol_annual),
            mean_reversion_speed=float(r.mean_reversion_speed),
            mean_reversion_level=float(r.mean_reversion_level),
            intraday_range_scale=float(r.intraday_range_scale),
            p_transition=float(r.p_transition),
        ) for r in scenario.regimes]
        family.append(bt.generate_bars(
            n_bars=800, regimes=regimes, p_transition=float(scenario.p_transition),
            start_price=100.0, seed=base_seed + i * 1000))
    return family


def family_inline(n_assets=10, scenario=None, base_seed=SEED):
    """Verifier's inline family build."""
    family = []
    for i in range(n_assets):
        family.append(bt.generate_bars(
            n_bars=800,
            regimes=[bt.Regime(
                drift_annual=float(r.drift_annual + offsets[i]),
                vol_annual=float(r.vol_annual),
                mean_reversion_speed=float(r.mean_reversion_speed),
                mean_reversion_level=float(r.mean_reversion_level),
                intraday_range_scale=float(r.intraday_range_scale),
                p_transition=float(r.p_transition),
            ) for r in scenario.regimes],
            p_transition=float(scenario.p_transition),
            start_price=100.0,
            seed=base_seed + i * 1000,
        ))
    return family


def pnl_spread(family, lookback, top_k, bottom_k):
    first_len = len(family[0])
    pnl = np.zeros(first_len, dtype=np.float64)
    for i in range(lookback, first_len - 1):
        rets = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            rets[a] = float(np.log(c[i]) - np.log(c[i - lookback]))
        order = sorted(rets, key=lambda a: rets[a], reverse=True)
        longs = order[:top_k]
        shorts = order[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        rl = float(np.mean([np.log(family[a].closes_array()[i + 1])
                            - np.log(family[a].closes_array()[i]) for a in longs]))
        rs = float(np.mean([np.log(family[a].closes_array()[i + 1])
                            - np.log(family[a].closes_array()[i]) for a in shorts]))
        pnl[i] = rl - rs
    return pnl


def folds_lrets(pnl, train=60, test=20, warm=10, overlap=10):
    n = len(pnl)
    step = test - overlap
    fold_start = 0
    lrets = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n:
            break
        oos = pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets


scenarios = bt.canonical_regime_scenarios()

print("=== per-scenario family medians (lookback=63, 3/3), seed offset per scenario ===")
for s_idx, s in enumerate(scenarios):
    fam = family_via_functional(n_assets=10, scenario=s, base_seed=SEED + s_idx)
    pnl = pnl_spread(fam, LOOKBACK, TOP_K, BOTTOM_K)
    lrets = folds_lrets(pnl)
    print("  {} (seed_base={}): median={:+.5f}, mean={:+.5f}, n_folds={} "
          "| first 3 assets closes[100]: {}".format(
              s.name, SEED + s_idx, np.median(lrets), np.mean(lrets), len(lrets),
              [round(float(fam[k].closes_array()[100]), 3) for k in range(3)]))

print("\n=== per-scenario family medians (same logic, verifier-style: no per-scenario seed offset) ===")
for s_idx, s in enumerate(scenarios):
    fam = family_inline(n_assets=10, scenario=s, base_seed=SEED)
    pnl = pnl_spread(fam, LOOKBACK, TOP_K, BOTTOM_K)
    lrets = folds_lrets(pnl)
    print("  {}: median={:+.5f}".format(s.name, np.median(lrets)))

print("\n=== is family identical across scenarios? check vs verifier ===")
# same seed, same regimes -> check (seed+0) vs verifier (seed) for calm should be identical
fam1 = family_via_functional(n_assets=10, scenario=scenarios[0], base_seed=SEED + 0)
fam2 = family_inline(n_assets=10, scenario=scenarios[0], base_seed=SEED)
diff = any(not np.isclose(fam1[k].closes_array(), fam2[k].closes_array()).all() for k in range(10))
print("  calm family via_functional(seed=42) vs inline(seed=42) differ:", diff)

print("\n=== does the family change across scenarios with SAME seed? (verifier behavior) ===")
fams = [family_inline(n_assets=10, scenario=s, base_seed=SEED) for s in scenarios]
print("  families equal pairwise:", all(
    np.isclose(fams[i][k].closes_array(), fams[j][k].closes_array()).all()
    for i in range(4) for j in range(i + 1, 4) for k in range(10)))
print("  calm vs mean_reverting asset0 closes[200]:",
      round(float(fams[0][0].closes_array()[200]), 3), "vs",
      round(float(fams[2][0].closes_array()[200]), 3))
