"""Smoke test for the cross-sectional momentum primitives.

This exercises the new framework functions on synthetic data:
  initialize (generate a synthetic family) -> use the new tool (cs_momentum_*)
  -> perform representative computation (rank, long top-3 / short bottom-3, hold 1
  day) -> produce an artifact (small JSON) -> validate / reload it.

The core invariant being validated: the fresh implementation matches the
framework implementation element-wise, the spread is neutral before the
lookback window, and the coin-flip null preserves magnitude while randomizing
sign. Nothing here asserts any trading edge; this is pure smoke test/shape
validation.

Research/simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "smoke_cs_momentum.json"

LOOKBACK = 63
TOP_K = BOTTOM_K = 3
SEED = 42


def fresh_spread(family, lookback, top_k, bottom_k):
    """Fresh, self-contained implementation of the momentum spread (does not
    import research.backtest.regime_stability.cs_momentum_*)."""
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
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
        leg = lambda a: float(np.log(family[a].closes_array()[i + 1])
                              - np.log(family[a].closes_array()[i]))
        daily_pnl[i] = np.mean([leg(a) for a in longs]) - np.mean([leg(a) for a in shorts])
    return daily_pnl


def fresh_null(family, lookback, top_k, bottom_k, seed):
    """Fresh, self-contained coin-flip null (does not import the framework)."""
    rng = np.random.default_rng(seed)
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
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
        leg = lambda a: float(np.log(family[a].closes_array()[i + 1])
                              - np.log(family[a].closes_array()[i]))
        spread = np.mean([leg(a) for a in longs]) - np.mean([leg(a) for a in shorts])
        daily_pnl[i] = spread if rng.random() < 0.5 else -spread
    return daily_pnl


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    scenario = bt.canonical_regime_scenarios()[0]
    family = [bt.generate_bars(800, regimes=scenario.regimes,
                              p_transition=scenario.p_transition,
                              start_price=100.0, seed=42 + i * 1000)
              for i in range(6)]

    print("=== 1. Compute spread with fresh implementation ===")
    cand = fresh_spread(family, LOOKBACK, TOP_K, BOTTOM_K)
    print("  shape: {} | first {} bars neutral: {}".format(
        len(cand), LOOKBACK, bool(np.all(cand[:LOOKBACK] == 0.0))))

    print("=== 2. Match against framework implementation ===")
    fw = bt.cs_momentum_spread_family(family, LOOKBACK, TOP_K, BOTTOM_K)
    match = bool(np.array_equal(cand, fw))
    print("  fresh == framework element-wise: {}".format(match))
    if not match:
        print("  [DEBUG] fresh: {}  framework: {}".format(cand[:5], fw[:5]))
        return 1

    print("=== 3. Neutral-bar invariant and minimum-legs invariant ===")
    n_neutral = int(np.sum(cand == 0.0))
    print("  zero-P&L bars: {} / {} (expected >= {} lookback bars)".format(
        n_neutral, len(cand), LOOKBACK))

    print("=== 4. Coin-flip null magnitude/structure invariant ===")
    rng = np.random.default_rng(SEED)
    null = np.zeros(len(cand), dtype=np.float64)
    for i in range(LOOKBACK, len(cand) - 1):
        rets = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            rets[a] = float(c[i] / c[i - LOOKBACK] - 1)
        order = sorted(rets, key=lambda a: rets[a], reverse=True)
        longs = order[:TOP_K]
        shorts = order[-BOTTOM_K:]
        if len(longs) < TOP_K or len(shorts) < BOTTOM_K:
            continue
        leg = lambda a: float(np.log(family[a].closes_array()[i + 1])
                              - np.log(family[a].closes_array()[i]))
        spread = np.mean([leg(a) for a in longs]) - np.mean([leg(a) for a in shorts])
        null[i] = spread if rng.random() < 0.5 else -spread
    null_match = bool(np.allclose(np.abs(null), np.abs(fw), atol=1e-12))
    print("  |null| == |framework spread| bar-for-bar: {}".format(null_match))
    if not null_match:
        return 1

    print("=== 5. Framework null matches fresh null ===")
    fw_null = bt.cs_momentum_null_spread_family(family, LOOKBACK, TOP_K,
                                                BOTTOM_K, SEED)
    print("  fresh null == framework null element-wise: {}".format(
        bool(np.array_equal(null, fw_null))))
    if not np.array_equal(null, fw_null):
        return 1

    print("=== 6. Determinism: two runs identical ===")
    fw2 = bt.cs_momentum_spread_family(family, LOOKBACK, TOP_K, BOTTOM_K)
    print("  run1 == run2: {}".format(bool(np.array_equal(fw, fw2))))
    if not np.array_equal(fw, fw2):
        return 1

    artifact = dict(
        n_assets=6,
        n_bars=len(cand),
        lookback=LOOKBACK, top_k=TOP_K, bottom_k=BOTTOM_K, seed=SEED,
        shape_ok=bool(len(cand) == 800),
        neutral_bars_ok=bool(np.all(cand[:LOOKBACK] == 0.0)),
        fresh_framework_match=bool(np.array_equal(cand, fw)),
        null_magnitude_invariant=bool(null_match),
        framework_null_equal_fresh_null=bool(np.array_equal(null, fw_null)),
        determinism_ok=bool(np.array_equal(fw, fw2)),
        first_spread_bars=[round(float(x), 6) for x in fw[LOOKBACK:LOOKBACK + 5]],
        null_first_bars=[round(float(x), 6) for x in fw_null[LOOKBACK:LOOKBACK + 5]],
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\n=== 7. Artifact written and reloaded ===")
    with open(ARTIFACT_PATH) as f:
        reload = json.load(f)
    print("  artifact fields: {}".format(sorted(reload.keys())))
    print("  reloaded shape_ok: {} | determinism_ok: {}"
          .format(reload["shape_ok"], reload["determinism_ok"]))
    all_ok = all([reload["shape_ok"], reload["neutral_bars_ok"],
                  reload["fresh_framework_match"], reload["null_magnitude_invariant"],
                  reload["framework_null_equal_fresh_null"], reload["determinism_ok"]])
    print("\nSMOKE TEST: {}".format("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
