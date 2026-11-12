"""Independent verification of the momentum cost-sensitivity artifact.

This verifier does NOT call ``stress_segments`` (the framework path used by the
check). It independently recomputes, for a representative subset of cells:

  1. segment boundaries from ``volatility_blocks`` (re-implemented here);
  2. per-segment candidate medians via ``parameter_sweep`` + ``sweep_summary``;
  3. per-segment coin-flip null medians via ``noise_benchmark`` under the SAME
     cost config;
  4. the regime verdict with the documented threshold logic (re-implemented);
  5. the ``positive_edge`` flag and the cost-robustness classification.

It compares all of these to ``state/check_artifacts/momentum_cost_sensitivity_results.json``
and asserts agreement, then re-runs the recomputation to assert determinism.

It also independently re-derives the signal-correctness diagnostic (weight
composition and the ``existing == always_long`` claim) from the raw data.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
OUTLIER_TOL = 0.05

# Subset recomputed through the independent path (both variants, lb=5, all costs).
SUBSET_LOOKBACKS = (5,)
SUBSET_TICKERS = None  # all tickers in the artifact


def corrected_momentum_signals(closes, lookback=5):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        out[i] = bt.Signal(date=i + 1,
                           weight=1.0 if float(closes[i]) > float(closes[i - lookback]) else -1.0)
    return out


VARIANTS = {"existing": bt.momentum_signals, "corrected": corrected_momentum_signals}


def segment_bounds(labels, min_bars):
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_bars:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_bars:
        segments.append((cur, start, len(labels)))
    return segments


def cfg_for(cost_levels, level):
    c = cost_levels[level]
    return bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=c["commission_per_trade"],
        commission_per_share=c["commission_per_share"],
        slippage_cents=c["slippage_cents"],
        slippage_proportional=c["slippage_proportional"],
    )


def verdict_of(cand, null):
    cand = np.asarray(cand, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    cd = float(np.std(cand))
    nd = float(np.std(null))
    if all(abs(m) <= OUTLIER_TOL for m in cand):
        return "CONSISTENT_WITH_NOISE", cd, nd
    if cd > 2.0 * nd:
        return "REGIME_DEPENDENT", cd, nd
    if all(m < 0 for m in cand):
        return "REGIME_STABLE_LOSS", cd, nd
    return "REGIME_STABLE", cd, nd


def independent_cell(bars, signal_fn, lookback, cfg):
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    segs = segment_bounds(labels, MIN_SEG)
    bars_list = list(bars)
    cand, null, names, edges = [], [], [], []
    for name, s, e in segs:
        seg_bars = bars_list[s:e]
        sweep = bt.parameter_sweep(
            signals_fn=signal_fn, bars=seg_bars, param_grid=[{"lookback": lookback}],
            train_window=TRAIN, test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
            cfg=cfg, periods_per_year=252)
        summ = bt.sweep_summary(sweep, (("lookback", lookback),))
        noise = bt.noise_benchmark(
            seg_bars, param_grid=[{"lookback": lookback}], train_window=TRAIN,
            test_window=TEST, warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252)
        cm = float(summ.baseline_median_log_return)
        nm = float(noise.baseline_median_log_return)
        cand.append(cm)
        null.append(nm)
        names.append(name)
        edges.append(1 if abs(cm - nm) > OUTLIER_TOL else 0)
    v, cd, nd = verdict_of(cand, null)
    positive_edge = (v == "REGIME_STABLE" and all(m > 0 for m in cand))
    return dict(
        segments=names,
        medians=[round(m, 6) for m in cand],
        null_medians=[round(m, 6) for m in null],
        edge_status=edges,
        candidate_dispersion=round(cd, 6),
        null_dispersion=round(nd, 6),
        n_folds=sweep.n_folds,
        verdict=v,
        positive_edge=bool(positive_edge),
        above_null=bool(all(m > n for m, n in zip(cand, null))),
    )


def approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def classify(cells):
    if not cells["zero"]["positive_edge"]:
        return "NO_EDGE"
    if cells["realistic"]["positive_edge"] and cells["conservative"]["positive_edge"]:
        return "COST_ROBUST"
    return "COST_FRAGILE"


def main() -> int:
    with open(ARTIFACT_PATH) as fh:
        art = json.load(fh)
    cost_levels = art["cost_levels"]
    ticker_order = list(art["cells"]["existing"]["5"]["zero"].keys())
    if SUBSET_TICKERS:
        ticker_order = [t for t in ticker_order if t in SUBSET_TICKERS]

    manifest = load_manifest()
    universe = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    print("=== Gate 0: independent signal-correctness diagnostic ===")
    diag_ok = True
    for t in ticker_order:
        closes = universe[t].closes_array()
        ex = [s.weight for s in bt.momentum_signals(closes, 5)[5:]]
        al = [s.weight for s in bt.always_long_signal(closes, 5, 5)[5:]]
        is_al = all(a == b for a, b in zip(ex, al))
        rec = art["signal_diagnostic"][t]
        ok = (rec["existing"]["long"] == sum(1 for w in ex if w > 0)
              and rec["existing"]["short"] == sum(1 for w in ex if w < 0)
              and rec["existing_is_always_long"] == is_al)
        diag_ok = diag_ok and ok
        print(f"  {t:6s} existing_long={rec['existing']['long']:5d} "
              f"recomputed_long={sum(1 for w in ex if w > 0):5d} "
              f"always_long={is_al} match={ok}")
    print(f"  SIGNAL_DIAGNOSTIC_MATCH: {diag_ok}")
    assert diag_ok, "signal diagnostic mismatch"

    print("\n=== Gate 1: independent recomputation (both variants, lb=5, all costs, all tickers) ===")
    gate1_ok = True
    for v in VARIANTS:
        for lb in SUBSET_LOOKBACKS:
            for lvl in cost_levels:
                cfg = cfg_for(cost_levels, lvl)
                for t in ticker_order:
                    rec = independent_cell(universe[t], VARIANTS[v], lb, cfg)
                    ref = art["cells"][v][str(lb)][lvl][t]
                    ok = (rec["segments"] == ref["segments"]
                          and all(approx(a, b) for a, b in zip(rec["medians"], ref["medians"]))
                          and all(approx(a, b) for a, b in zip(rec["null_medians"], ref["null_medians"]))
                          and rec["verdict"] == ref["verdict"]
                          and rec["positive_edge"] == ref["positive_edge"]
                          and rec["above_null"] == ref["above_null"])
                    if not ok:
                        gate1_ok = False
                        print(f"  MISMATCH {v} lb={lb} {lvl} {t}:")
                        print(f"    recomputed: {rec}")
                        print(f"    artifact  : {ref}")
    print(f"  GATE1_RECOMPUTATION_MATCH: {gate1_ok}")
    assert gate1_ok, "independent recomputation does not match the artifact"

    print("\n=== Gate 2: independent classification of the recomputed subset ===")
    gate2_ok = True
    for v in VARIANTS:
        for lb in SUBSET_LOOKBACKS:
            for t in ticker_order:
                cells = {lvl: art["cells"][v][str(lb)][lvl][t] for lvl in cost_levels}
                # recompute cells from the independent path for classification
                icells = {lvl: independent_cell(universe[t], VARIANTS[v], lb, cfg_for(cost_levels, lvl))
                          for lvl in cost_levels}
                cls_ref = art["classification"][v][str(lb)]["per_asset"][t]["classification"]
                cls_ind = classify(icells)
                if cls_ref != cls_ind:
                    gate2_ok = False
                    print(f"  CLASS_MISMATCH {v} lb={lb} {t}: artifact={cls_ref} independent={cls_ind}")
    print(f"  GATE2_CLASSIFICATION_MATCH: {gate2_ok}")
    assert gate2_ok, "classification does not reproduce"

    print("\n=== Gate 3: determinism (recompute subset twice, compare) ===")
    v, lb, lvl, t = "corrected", 5, "realistic", "AMZN"
    r1 = independent_cell(universe[t], VARIANTS[v], lb, cfg_for(cost_levels, lvl))
    r2 = independent_cell(universe[t], VARIANTS[v], lb, cfg_for(cost_levels, lvl))
    det = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    print(f"  DETERMINISM_MATCH: {det}")
    assert det, "non-deterministic independent recomputation"

    print("\nALL GATES PASSED: artifact is independently reproduced and deterministic.")


if __name__ == "__main__":
    main()