"""Smoke test for the volatility-targeting signal class.

A cheap representative computation on synthetic data exercising:
initialize -> use required framework primitives -> representative computation
-> produce an artifact -> validate and reload it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import research.backtest as bt


SEED = 42
LOOKBACK = 60
TOP_K = BOTTOM_K = 3
N_BARS = 800


def make_synthetic_family(n_assets=10, scenario=None, base_seed=SEED):
    """Build a family of n_assets synthetic BarSequences sharing a regime
    structure with shifted drifts (cross-sectional dispersion)."""
    drift_offsets = [-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.06, 0.08, 0.10]
    family = []
    for i in range(n_assets):
        regimes = [bt.Regime(
            drift_annual=float(r.drift_annual + drift_offsets[i]),
            vol_annual=float(r.vol_annual),
            mean_reversion_speed=float(r.mean_reversion_speed),
            mean_reversion_level=float(r.mean_reversion_level),
            intraday_range_scale=float(r.intraday_range_scale),
            p_transition=float(r.p_transition),
        ) for r in scenario.regimes]
        family.append(bt.generate_bars(
            n_bars=N_BARS,
            regimes=regimes,
            p_transition=float(scenario.p_transition),
            start_price=100.0,
            seed=base_seed + i * 1000,
        ))
    return family


def main() -> int:
    print("=== Smoke test: volatility-targeting (long low-vol / short high-vol) ===")
    print("Synthetic family: 10 assets, {} bars, seed {}"
          .format(N_BARS, SEED))

    family = make_synthetic_family(scenario=bt.canonical_regime_scenarios()[0])

    spread = bt.vol_rank_spread_family(family, LOOKBACK, TOP_K, BOTTOM_K)
    null = bt.vol_rank_null_spread_family(family, LOOKBACK, TOP_K, BOTTOM_K, SEED)

    # Determinism: rerun both and compare.
    spread_r2 = bt.vol_rank_spread_family(family, LOOKBACK, TOP_K, BOTTOM_K)
    null_r2 = bt.vol_rank_null_spread_family(family, LOOKBACK, TOP_K, BOTTOM_K, SEED)
    det = np.array_equal(spread, spread_r2) and np.array_equal(null, null_r2)
    print("  spread shape: {}, null shape: {}"
          .format(len(spread), len(null)))
    print("  spread: mean {:+.4f}, std {:+.4f}, median {:+.4f}"
          .format(float(np.mean(spread)), float(np.std(spread, ddof=1)),
                  float(np.median(spread))))
    print("  null  : mean {:+.4f}, std {:+.4f}, median {:+.4f}"
          .format(float(np.mean(null)), float(np.std(null, ddof=1)),
                  float(np.median(null))))
    print("  spread/std == null/std (same magnitude structure): {}"
          .format(np.allclose(np.abs(spread), np.abs(null))))
    print("  determinism r1 == r2: {}".format(det))

    # Self-consistency: build a synthetic equity asset from the spread and
    # confirm its compounded OOS total_return equals the spread's compounded
    # equity over the same window: eq[w] / eq[LOOKBACK] - 1 ==
    # prod(1 + spread[LOOKBACK+1 : w+1]) - 1 (build_synthetic_spread_asset
    # seeds equity at index 0 with spread[0], so eq[LOOKBACK] carries
    # spread[0:LOOKBACK+1] and the ratio cancels to spread[LOOKBACK+1:w+1]).
    synth = bt.build_synthetic_spread_asset(spread, start_price=1e6)
    w = LOOKBACK + 100
    print("  spread[LOOKBACK]={} spread[LOOKBACK+1]={} spread[w-1]={} "
          "spread[w]={}".format(spread[LOOKBACK], spread[LOOKBACK + 1],
                                spread[w - 1], spread[w]))
    print("  spread nan: {}, inf: {}".format(bool(np.any(np.isnan(spread))),
          bool(np.any(np.isinf(spread)))))
    eq_ratio = synth.closes_array()[w] / synth.closes_array()[LOOKBACK] - 1.0
    pnl_ratio = 1.0
    for r in spread[LOOKBACK + 1 : w + 1]:
        pnl_ratio *= 1.0 + r
    # Recompute equity directly to confirm build_synthetic_spread_asset's
    # compounding matches the spread value-by-value.
    start = 1e6
    direct = start
    for i, r in enumerate(spread[:w + 1]):
        direct = (start * (1.0 + r) if i == 0 else direct * (1.0 + r)) if r != 0.0 else direct
    direct_60 = start
    for i, r in enumerate(spread[:LOOKBACK + 1]):
        direct_60 = (start * (1.0 + r) if i == 0 else direct_60 * (1.0 + r)) if r != 0.0 else direct_60
    print("  synth eq[{}]={:+.6f} direct eq[{}]={:+.6f}".format(w, synth.closes_array()[w], w, direct))
    print("  synth eq[{}]={:+.6f} direct eq[{}]={:+.6f}".format(LOOKBACK, synth.closes_array()[LOOKBACK], LOOKBACK, direct_60))
    print("  synth eq[w]/eq[LOOKBACK]-1 = {:+.6f}; direct = {:+.6f}".format(
          synth.closes_array()[w] / synth.closes_array()[LOOKBACK] - 1.0,
          direct / direct_60 - 1.0))
    print("  prod(1+spread[61:161]) via numpy:      {:+.6f}".format(
          np.prod(1.0 + spread[61:161])))
    print("  prod(1+spread[61:161]) via loop+pnl+1: {:+.6f}".format(pnl_ratio + 1.0))
    print("  prod(1+spread[61:161]) via loop+direct: {:+.6f}".format(
          (direct / direct_60)))
    print("  spread[61:65]={} spread[156:161]={}".format(spread[61:65], spread[156:161]))
    print("  spread==0 in [61:161]: {}, spread<-1 in [61:161]: {}".format(
          int((spread[61:161] == 0.0).sum()), int((spread[61:161] < -1.0).sum())))
    consistent = abs((eq_ratio + 1.0) - pnl_ratio) < 1e-12
    print("  synthetic-equity OOS total return: {:+.4f}".format(eq_ratio))
    print("  spread compounded return:          {:+.4f}".format(pnl_ratio))
    print("  synthetic-equity path self-consistent: {}".format(
        "PASS" if consistent else "FAIL"))

    # Regime-gate representative: 4 volatility-block labels and a
    # walk-forward fold-log-return aggregation.
    labels = bt.volatility_blocks(synth.closes_array(), n_blocks=4, window=60)
    fold_step = 84 - 60
    lrets = []
    fs = 0
    while True:
        fe = fs + 60 + 252 + 84
        if fe > len(spread):
            break
        oos = spread[fe - 84 + 1:fe]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(comp - 1.0)))
        fs += fold_step
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= 400:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= 400:
        runs.append((cur, start, len(labels)))
    print("  walk-forward fold count: {} (train=252d/test=84d/warmup=60d)"
          .format(len(lrets)))
    print("  fold log-returns: mean {:+.4f}, median {:+.4f}, "
          "segments {}".format(float(np.mean(lrets)), float(np.median(lrets)),
                               runs))

    # Write the smoke artifact as a flat structure (no tuples) and reload it
    # to confirm round-trip integrity.
    artifact = dict(
        n_assets=len(family),
        n_bars=N_BARS,
        lookback=LOOKBACK,
        top_k=TOP_K,
        bottom_k=BOTTOM_K,
        spread_mean=float(np.mean(spread)),
        spread_std=float(np.std(spread, ddof=1)),
        null_mean=float(np.mean(null)),
        null_std=float(np.std(null, ddof=1)),
        spread_null_magnitude_match=bool(np.allclose(np.abs(spread), np.abs(null))),
        determinism=bool(det),
        equity_self_consistent=bool(consistent),
        walk_forward=dict(fold_count=len(lrets),
                          mean_log_return=float(np.mean(lrets)),
                          median_log_return=float(np.median(lrets)),
                          segments=["{},{},{}".format(n, s, e) for n, s, e in runs]),
    )
    artifact_path = Path.cwd() / "state" / "check_artifacts" / "volatility_targeting_smoke.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    with open(artifact_path) as f:
        reload = json.load(f)
    reload_ok = all(reload.get(k) == artifact[k] for k in artifact)
    print("  smoke artifact round-trip: {} -> {}".format(artifact_path,
          "OK" if reload_ok else "FAIL"))

    ok = det and consistent and reload_ok and len(lrets) > 0
    print()
    if ok:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
