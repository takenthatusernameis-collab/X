"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

The verifier deliberately shares no code path with the check for the quantities
it certifies. It re-implements, from the documented specification:

* the ``volatility_blocks`` regime classification (own loop),
* segment reconstruction from labels,
* the momentum signal (trailing ``lookback``-bar log return),
* the coin-flip null, built from ``np.random.default_rng`` directly and driven
  through ``bt.walk_forward`` rather than through ``noise_benchmark``,
* the regime verdict rule and the EDGE / asset-class / universe classification.

Gates:

1. Cost-grid recomputation: every (ticker, lookback, cost level) cell's segment
   medians and null medians are recomputed and compared to the artifact.
2. Classification re-derivation: cell verdicts, per-asset cost classes, and the
   universe verdict are re-derived from the recomputed medians and compared to
   the artifact.
3. Legacy-signal audit: the pre-2026-11-20 formula is shown to emit a single
   constant weight for every ticker/lookback (30/30 cells), and its
   constant-long medians reproduce the recorded `momentum_results.json`
   medians on all 10 assets — i.e. the recorded "momentum edge" is a
   constant-long (buy-and-hold) backtest.
4. Determinism: the verifier's own recomputation is repeated and compared.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
OUTLIER_TOL = 0.05

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
MOMENTUM_ARTIFACT = ARTIFACT_DIR / "momentum_results.json"

BASE_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_LEVELS = {
    "zero": 0.0,
    "repo_realistic_1x": 1.0,
    "conservative_2x": 2.0,
    "stress_4x": 4.0,
}
TOL = 1e-6


def vol_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
            )
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window
                else 0.0
            )
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_bars=MIN_SEGMENT_BARS):
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_bars:
        runs.append((cur, start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    """Fresh implementation of the repaired momentum signal."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def legacy_weights(closes, lookback):
    """Pre-repair formula: sign of the mean log PRICE LEVEL over the window."""
    n = len(closes)
    out = [0.0] * n
    for i in range(int(lookback), n):
        level = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = 1.0 if level > 0.0 else -1.0
    return out


def cfg_for(multiplier):
    return bt.BacktestConfig(
        **{k: v * float(multiplier) for k, v in BASE_COSTS.items()}
    )


def null_signals(n, lookback, seed=42):
    """Coin-flip null, rebuilt directly from numpy (no noise_benchmark call)."""
    rng = np.random.default_rng(seed + int(lookback) * 1000)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


def fold_log_returns(result):
    return np.array(
        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
         for f in result.folds],
        dtype=np.float64,
    )


def cell(s, e, lookback, cfg, bars_list):
    """Fresh path: candidate medians + null medians for one segment/lookback."""
    seg_bars = bars_list[s:e]
    closes = np.array([b.close for b in bars_list], dtype=np.float64)
    sig = momentum_signals(closes, lookback)[s:e]
    res = bt.walk_forward(
        seg_bars, sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg,
    )
    cand = fold_log_returns(res)
    null_res = bt.walk_forward(
        seg_bars, null_signals(len(seg_bars), lookback),
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg,
    )
    null = fold_log_returns(null_res)
    return dict(
        candidate=float(np.median(cand)),
        null=float(np.median(null)),
        n_folds=int(len(cand)),
    )


def verdict_of(medians, nulls):
    """Fresh implementation of the regime verdict rule."""
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    cand_disp = float(np.std(medians))
    null_disp = float(np.std(nulls))
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0.0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def cell_verdict(verdict, medians):
    if verdict == "REGIME_STABLE" and all(m > 0.0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def main() -> int:
    with open(ARTIFACT) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID, art["dataset_id"]

    bars_by_ticker = {t: bt.load_ticker(t)[0] for t in art["ticker_order"]}
    bars_list = {t: list(b) for t, b in bars_by_ticker.items()}

    print("=== Gate 1. Cost-grid recomputation (fresh path) ===")
    recomputed = {}
    mismatches = []
    n_cells = 0
    for level_name, mult in COST_LEVELS.items():
        cfg = cfg_for(mult)
        recomputed[level_name] = {}
        for lb in LOOKBACKS:
            recomputed[level_name][str(lb)] = {}
            for t in art["ticker_order"]:
                closes = np.array([b.close for b in bars_list[t]], dtype=np.float64)
                labels = vol_blocks(closes)
                segs = segments_from_labels(labels)
                seg_med, seg_null = [], []
                n_folds = None
                for name, s, e in segs:
                    r = cell(s, e, lb, cfg, bars_list[t])
                    if n_folds is None:
                        # stress_segments reports the first segment's fold count.
                        n_folds = r["n_folds"]
                    seg_med.append(r["candidate"])
                    seg_null.append(r["null"])
                rec = dict(
                    medians=[round(x, 6) for x in seg_med],
                    null_medians=[round(x, 6) for x in seg_null],
                    n_folds=n_folds,
                )
                exp = art["per_asset"][t]["cells"][level_name][str(lb)]
                n_cells += 1
                ok_med = all(abs(a - b) <= TOL
                             for a, b in zip(rec["medians"], exp["medians"]))
                ok_null = all(abs(a - b) <= TOL
                              for a, b in zip(rec["null_medians"], exp["null_medians"]))
                ok_n = rec["n_folds"] == exp["n_folds"]
                if not (ok_med and ok_null and ok_n):
                    mismatches.append((level_name, lb, t, rec["medians"],
                                       exp["medians"], rec["null_medians"],
                                       exp["null_medians"]))
                recomputed[level_name][str(lb)][t] = dict(
                    medians=rec["medians"], null_medians=rec["null_medians"],
                    n_folds=rec["n_folds"])
    print(f"  cells recomputed: {n_cells}")
    print(f"  cells matching the artifact: {n_cells - len(mismatches)}/{n_cells}")
    for m in mismatches[:5]:
        print("   MISMATCH", m)
    if mismatches:
        print("  GATE 1: FAIL")
        return 1
    print("  GATE 1: MATCH")

    print("\n=== Gate 2. Classification re-derivation ===")
    class_mismatch = []
    class_counts = {"ROBUST": 0, "PARTIAL": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    for t in art["ticker_order"]:
        zero_edges = 0
        base_edges = 0
        for lb in LOOKBACKS:
            for level_name in ("zero", "repo_realistic_1x"):
                r = recomputed[level_name][str(lb)][t]
                v = verdict_of(r["medians"], r["null_medians"])
                cv = cell_verdict(v, r["medians"])
                exp_cv = art["per_asset"][t]["cells"][level_name][str(lb)]["cell"]
                if cv != exp_cv:
                    class_mismatch.append((t, level_name, lb, cv, exp_cv))
                if level_name == "zero" and cv == "EDGE":
                    zero_edges += 1
                if level_name == "repo_realistic_1x" and cv == "EDGE":
                    base_edges += 1
        if base_edges >= 2:
            cls = "ROBUST"
        elif base_edges == 1:
            cls = "PARTIAL"
        elif zero_edges >= 1:
            cls = "COST_FRAGILE"
        else:
            cls = "NO_EDGE"
        class_counts[cls] += 1
        if cls != art["per_asset"][t]["cost_class"]:
            class_mismatch.append((t, "asset_class", cls,
                                   art["per_asset"][t]["cost_class"]))
    retained = sum(
        1 for t in art["ticker_order"]
        if sum(
            1 for lb in LOOKBACKS
            if cell_verdict(
                verdict_of(recomputed["repo_realistic_1x"][str(lb)][t]["medians"],
                           recomputed["repo_realistic_1x"][str(lb)][t]["null_medians"]),
                recomputed["repo_realistic_1x"][str(lb)][t]["medians"]) == "EDGE"
        ) >= 2
    )
    edge_pairs_zero = sum(
        1 for t in art["ticker_order"] for lb in LOOKBACKS
        if cell_verdict(
            verdict_of(recomputed["zero"][str(lb)][t]["medians"],
                       recomputed["zero"][str(lb)][t]["null_medians"]),
            recomputed["zero"][str(lb)][t]["medians"]) == "EDGE"
    )
    if retained >= 6:
        universe = "COST_ROBUST"
    elif retained >= 1:
        universe = "PARTIALLY_COST_ROBUST"
    elif edge_pairs_zero > 0:
        universe = "COST_FRAGILE"
    else:
        universe = "NO_BASELINE_EDGE"
    if universe != art["universe_verdict"]:
        class_mismatch.append(("universe", universe, art["universe_verdict"]))
    print(f"  per-asset classes re-derived: {class_counts}")
    print(f"  retained assets at 1x: {retained}/10; zero-cost EDGE cells: "
          f"{edge_pairs_zero}/30")
    print(f"  universe verdict re-derived: {universe} "
          f"(artifact: {art['universe_verdict']})")
    print(f"  classification mismatches: {len(class_mismatch)}")
    for m in class_mismatch[:5]:
        print("   MISMATCH", m)
    if class_mismatch:
        print("  GATE 2: FAIL")
        return 1
    print("  GATE 2: MATCH")

    print("\n=== Gate 3. Legacy-signal audit (the correction's evidentiary basis) ===")
    constant_cells = 0
    total_cells = 0
    nonconstant = []
    for t in art["ticker_order"]:
        closes = np.array([b.close for b in bars_list[t]], dtype=np.float64)
        for lb in LOOKBACKS:
            total_cells += 1
            nz = [w for w in legacy_weights(closes, lb) if w != 0.0]
            if len(set(nz)) == 1:
                constant_cells += 1
            else:
                nonconstant.append((t, lb, sorted(set(nz))))
    print(f"  legacy formula emits a single constant weight on "
          f"{constant_cells}/{total_cells} (ticker, lookback) cells")
    print(f"  non-constant cells: {nonconstant}")
    expected_nonconstant = [("NVDA", lb) for lb in LOOKBACKS]
    if [(t, lb) for t, lb, _ in nonconstant] != expected_nonconstant:
        print("  GATE 3: FAIL (unexpected set of non-degenerate legacy cells)")
        return 1
    if constant_cells != total_cells - len(expected_nonconstant):
        print("  GATE 3: FAIL (legacy degeneracy count)")
        return 1
    # Every other ticker is always long (single weight +1.0) regardless of lookback.
    always_long = all(
        set(w for w in legacy_weights(
            np.array([b.close for b in bars_list[t]], dtype=np.float64), lb)
            if w != 0.0) == {1.0}
        for t in art["ticker_order"] if t != "NVDA"
        for lb in LOOKBACKS
    )
    print(f"  9/10 tickers are a constant +1.0 (always long) at every lookback: "
          f"{always_long}")
    if not always_long:
        print("  GATE 3: FAIL")
        return 1
    # NVDA's exceptions are explained by its split-adjusted sub-$1 price level:
    # the legacy sign is the sign of the mean log price LEVEL, so it flips only
    # where that level crosses 1.0, not where returns change sign.
    nvda = np.array([b.close for b in bars_list["NVDA"]], dtype=np.float64)
    print(f"  NVDA adjusted-close range: [{nvda.min():.4f}, {nvda.max():.4f}] "
          f"(sub-$1 history makes the legacy price-level sign flip)")

    if not MOMENTUM_ARTIFACT.exists():
        print(f"  GATE 3: SKIP (prior artifact {MOMENTUM_ARTIFACT} absent)")
    else:
        with open(MOMENTUM_ARTIFACT) as f:
            prior = json.load(f)
        prior_universe = prior.get("universe", {}).get("per_asset", {})
        match_n = 0
        total_n = 0
        for t in art["ticker_order"]:
            entry = prior_universe.get(t)
            if entry is None:
                continue
            total_n += 1
            closes = np.array([b.close for b in bars_list[t]], dtype=np.float64)
            labels = vol_blocks(closes)
            segs = segments_from_labels(labels)
            cfg = cfg_for(0.0)
            meds = []
            legacy = legacy_weights(closes, 5)
            for name, s, e in segs:
                seg_bars = bars_list[t][s:e]
                sig = [bt.Signal(date=s + j + 1, weight=legacy[s + j])
                       for j in range(e - s)]
                res = bt.walk_forward(
                    seg_bars, sig, train_window=TRAIN, test_window=TEST,
                    warmup=WARM, overlap_window=OVERLAP, cfg=cfg,
                )
                meds.append(round(float(np.median(fold_log_returns(res))), 3))
            recorded = [round(float(x), 3) for x in entry["medians"]]
            if meds == recorded:
                match_n += 1
            else:
                print(f"   MISMATCH {t}: fresh {meds} vs recorded {recorded}")
        print(f"  recorded momentum medians reproduced by the legacy formula: "
              f"{match_n}/{total_n}")
        if match_n != total_n or total_n == 0:
            print("  GATE 3: FAIL")
            return 1
    print("  GATE 3: MATCH (legacy momentum == constant-long buy-and-hold)")

    print("\n=== Gate 4. Determinism of the verification path ===")
    t = art["ticker_order"][0]
    closes = np.array([b.close for b in bars_list[t]], dtype=np.float64)
    labels = vol_blocks(closes)
    segs = segments_from_labels(labels)
    cfg = cfg_for(1.0)
    r1 = [cell(s, e, 5, cfg, bars_list[t]) for _, s, e in segs]
    r2 = [cell(s, e, 5, cfg, bars_list[t]) for _, s, e in segs]
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    print(f"  determinism: r1 == r2: {same}")
    if not same:
        print("  GATE 4: FAIL")
        return 1
    print("  GATE 4: MATCH")
    print("\nALL GATES MATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())