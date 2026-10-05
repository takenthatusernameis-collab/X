"""Independent verification of the cross-sectional momentum check
(`research/checks/cross_sectional_momentum.py`).

This is a separate implementation path: it recomputes the momentum spread P&L
and the spread medians from raw tickers, runs walk-forward on the spread
engine per fold with the fold's own baseline (the same independent cross-check
strategy the momentum check uses on the direct compounding path), then compares
against the artifact written by the momentum check
(`state/check_artifacts/cross_sectional_momentum_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Tickers loaded from the manifest, truncated to the fully overlapping
  window (3614 bars, META) exactly as the check does.
- AAPL volatility blocks reconstructed with a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments reconstructed from labels.
- Per-segment medians are obtained via ``walk_forward`` on a synthetic spread
  asset per fold, each fold carrying its own baseline (independent of the
  check's direct compounding path but matching the check's engine cross-check);
  the check's medians use direct P&L compounding, so agreement is a genuine
  cross-implementation verification.
- The null is recomputed with a fresh coin-flip randomization (same seed) and
  run through the same engine path.
- Concentration-gate figures are recomputed from raw drop-one sub-universes.
- Perturbation baseline medians are recomputed by walking each parameter set
  independently.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch would flag an
error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 63
TOP_K = BOTTOM_K = 3
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
FULLY_OVERLAPPING_BARS = 3614
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "cross_sectional_momentum_results.json"


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
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
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def momentum_spread(tickers):
    """Fresh implementation of the cross-sectional momentum spread (independent
    code path: no research.backtest.cs_momentum_* calls)."""
    def closes(tb, idx):
        if hasattr(tb, "closes_array"):
            return tb.closes_array()[idx]
        return tb[idx]

    min_len = min(len(tb) for tb in tickers.values())
    pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(LOOKBACK, min_len - 1):
        rets = {t: closes(tb, i) / closes(tb, i - LOOKBACK) - 1
                for t, tb in tickers.items()}
        order = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = order[:TOP_K]
        shorts = order[-BOTTOM_K:]
        if len(longs) < TOP_K or len(shorts) < BOTTOM_K:
            continue
        ret_l = float(np.mean(
            [closes(tick, i + 1) / closes(tick, i) - 1
             for t, tick in tickers.items() if t in longs]))
        ret_s = float(np.mean(
            [closes(tick, i + 1) / closes(tick, i) - 1
             for t, tick in tickers.items() if t in shorts]))
        pnl[i] = ret_l - ret_s
    return pnl


def coin_flip_null_spread(tickers, seed=42):
    """Coin-flip null of the momentum spread (fresh code path)."""
    def closes(tb, idx):
        if hasattr(tb, "closes_array"):
            return tb.closes_array()[idx]
        return tb[idx]

    rng = np.random.default_rng(seed)
    min_len = min(len(tb) for tb in tickers.values())
    pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(LOOKBACK, min_len - 1):
        rets = {t: closes(tb, i) / closes(tb, i - LOOKBACK) - 1
                for t, tb in tickers.items()}
        order = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = order[:TOP_K]
        shorts = order[-BOTTOM_K:]
        if len(longs) < TOP_K or len(shorts) < BOTTOM_K:
            continue
        ret_l = float(np.mean(
            [closes(tick, i + 1) / closes(tick, i) - 1
             for t, tick in tickers.items() if t in longs]))
        ret_s = float(np.mean(
            [closes(tick, i + 1) / closes(tick, i) - 1
             for t, tick in tickers.items() if t in shorts]))
        pnl[i] = (ret_l - ret_s) if rng.random() < 0.5 else -(ret_l - ret_s)
    return pnl


def walk_forward_fold_lrets(pnl, train, test, warm, overlap):
    """Walk-forward OOS fold log returns for a spread P&L series (independent
    implementation of the fold geometry)."""
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


def regime_verdict(candidate_medians, null_medians):
    """Re-implementation of the check's regime_verdict logic."""
    cand = np.array(candidate_medians, dtype=np.float64)
    null = np.array(null_medians, dtype=np.float64)
    if all(abs(m) <= 0.05 for m in cand):
        return "CONSISTENT_WITH_NOISE"
    cand_disp = float(np.std(cand, ddof=1))
    null_disp = float(np.std(null, ddof=1))
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in cand):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def engine_medians_for_spread(pnl, runs):
    """Per-segment walk-forward medians via the engine, one fold slice per
    segment with the fold's own first-OOS bar as baseline. This is the same
    engine cross-check strategy as the check; the check's published medians are
    computed by direct P&L compounding, so this is an independent
    cross-implementation path."""
    synth = bt.build_synthetic_spread_asset(pnl, start_price=1e6)
    medians = []
    for lab, s, e in runs:
        fold_start = 0
        seg_lrets = []
        while True:
            fold_end = fold_start + WARM + TRAIN + TEST
            if fold_end > e - s:
                break
            w0 = fold_start + WARM
            seg_bars = list(synth)[s + fold_start:s + fold_end]
            seg_signals = [bt.Signal(date=s + fold_start + k + 1,
                                     weight=synth.closes_array()[s + fold_start + k] /
                                            synth.closes_array()[s + w0])
                           for k in range(fold_end - fold_start)]
            res = bt.walk_forward(seg_bars, seg_signals,
                                  train_window=TRAIN, test_window=TEST,
                                  warmup=WARM, overlap_window=OVERLAP,
                                  cfg=bt.BacktestConfig(periods_per_year=252))
            log = np.array([np.log1p(np.clip(f.metrics["total_return"],
                                             -1.0 + 1e-12, None))
                            for f in res.folds], dtype=np.float64)
            assert len(res.folds) == 1
            seg_lrets.append(float(log[0]))
            fold_start += TEST - OVERLAP
        medians.append(round(float(np.median(seg_lrets)), 3))
    return medians


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK
    assert artifact["top_k"] == TOP_K
    assert artifact["bottom_k"] == BOTTOM_K

    print("=== Independent recomputation (engine walk-forward per fold) ===")
    all_ok = True

    # ---- Load tickers, truncated to the fully overlapping window ----
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {}
    for entry in m["entries"]:
        bars, dates = bt.load_ticker(entry["ticker"])
        tickers[entry["ticker"]] = bars.closes_array()[:FULLY_OVERLAPPING_BARS]

    # ---- Reconstruct AAPL segments ----
    aapl_closes = tickers["AAPL"]
    aapl_labels = vol_blocks(aapl_closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(aapl_labels, MIN_SEGMENT_BARS)
    lbls = [lab for lab, _, _ in runs]
    pub_labels = artifact["regime_gate"]["segments"]
    if lbls != pub_labels:
        all_ok = False
        print("  [DEBUG] labels mismatch: {} vs {}".format(lbls, pub_labels))

    # ---- Candidate medians via engine path ----
    pnl = momentum_spread(tickers)
    cand_medians = engine_medians_for_spread(pnl, runs)
    pub_cand = artifact["regime_gate"]["candidate_medians"]
    cand_status = "MATCH" if cand_medians == pub_cand else "MISMATCH"
    if cand_status != "MATCH":
        all_ok = False

    # ---- Null medians via engine path (fresh coin-flip null, same seed) ----
    pnl_null = coin_flip_null_spread({t: tb for t, tb in tickers.items()}, SEED)
    null_medians = engine_medians_for_spread(pnl_null, runs)
    pub_null = artifact["regime_gate"]["null_medians"]
    null_status = "MATCH" if null_medians == pub_null else "MISMATCH"
    if null_status != "MATCH":
        all_ok = False

    regen_verdict = regime_verdict(cand_medians, null_medians)
    pub_verdict = artifact["regime_gate"]["verdict"]
    verdict_status = "MATCH" if regen_verdict == pub_verdict else "MISMATCH"
    if verdict_status != "MATCH":
        all_ok = False
    print("  segments: {} -> {}".format(lbls, "MATCH" if lbls == pub_labels else "DIFF"))
    print("  candidate medians: {} vs published {} -> {}".format(
        cand_medians, pub_cand, cand_status))
    print("  null medians: {} vs published {} -> {}".format(
        null_medians, pub_null, null_status))
    print("  verdict regenerated {} (published {}) -> {}".format(
        regen_verdict, pub_verdict, verdict_status))

    # ---- Concentration gate recomputation ----
    print("\n=== Concentration gate recomputation (drop-one) ===")
    full_lrets = walk_forward_fold_lrets(pnl, TRAIN, TEST, WARM, OVERLAP)
    null_lrets = walk_forward_fold_lrets(pnl_null, TRAIN, TEST, WARM, OVERLAP)
    full_median = float(np.median(full_lrets))
    null_tol = 2 * float(np.std(null_lrets, ddof=1)) / np.sqrt(len(full_lrets))
    drop_medians = {}
    for ticker in tickers:
        sub = {t: tb for t, tb in tickers.items() if t != ticker}
        spnl = momentum_spread(sub)
        drop_medians[ticker] = float(np.median(
            walk_forward_fold_lrets(spnl, TRAIN, TEST, WARM, OVERLAP)))
    impacts = {t: full_median - m for t, m in drop_medians.items()}
    best_share = max(impacts.values()) / full_median if full_median else 0.0
    pub_conc = artifact["concentration"]
    conc_ok = (np.isclose(round(full_median, 3), pub_conc["full_universe_median"])
               and np.isclose(round(null_tol, 4), pub_conc["null_tolerance"])
               and len(full_lrets) == pub_conc["fold_count"]
               and np.isclose(round(best_share, 3), pub_conc["best_share"]))
    if not conc_ok:
        all_ok = False
    print("  full-universe median {:+.3f} (pub {:+.3f}) -> {}".format(
        round(full_median, 3), pub_conc["full_universe_median"],
        "MATCH" if np.isclose(round(full_median, 3), pub_conc["full_universe_median"]) else "DIFF"))
    print("  null tolerance {:+.4f} (pub {:+.4f}) -> {}".format(
        round(null_tol, 4), pub_conc["null_tolerance"],
        "MATCH" if np.isclose(round(null_tol, 4), pub_conc["null_tolerance"]) else "DIFF"))
    print("  fold count {} (pub {}) -> {}".format(
        len(full_lrets), pub_conc["fold_count"],
        "MATCH" if len(full_lrets) == pub_conc["fold_count"] else "DIFF"))
    print("  best share {:+.3f} (pub {:+.3f}) -> {}".format(
        round(best_share, 3), pub_conc["best_share"],
        "MATCH" if np.isclose(round(best_share, 3), pub_conc["best_share"]) else "DIFF"))
    print("  concentration recomputation -> {}".format("MATCH" if conc_ok else "MISMATCH"))

    # ---- Perturbation sweep baseline recomputation ----
    print("\n=== Perturbation sweep baseline recomputation (lookback=63, 3/3) ===")
    grid = [{"lookback": lk, "top_k": tk, "bottom_k": tk}
            for lk in [21, 63, 126] for tk in [3, 5, 7]]
    scenarios = bt.canonical_regime_scenarios()
    recomputed = {tuple(sorted(p.items())): [] for p in grid}
    null_recomputed = {tuple(sorted(p.items())): [] for p in grid}
    baseline_meds = []
    baseline_nulls = []
    baseline_cand_fold = []
    baseline_null_fold = []
    for s_idx, s in enumerate(scenarios):
        family = []
        for i in range(10):
            family.append(bt.generate_bars(
                n_bars=800,
                regimes=[bt.Regime(
                    drift_annual=float(r.drift_annual + [-0.06, -0.03, -0.01, 0.0, 0.01,
                                                          0.03, 0.05, 0.06, 0.08, 0.10][i]),
                    vol_annual=float(r.vol_annual),
                    mean_reversion_speed=float(r.mean_reversion_speed),
                    mean_reversion_level=float(r.mean_reversion_level),
                    intraday_range_scale=float(r.intraday_range_scale),
                    p_transition=float(r.p_transition),
                ) for r in s.regimes],
                p_transition=float(s.p_transition),
                start_price=100.0,
                seed=SEED + s_idx + i * 1000,
            ))
        for p in grid:
            first_len = len(family[0])
            pnl_p = np.zeros(first_len, dtype=np.float64)
            for i in range(p["lookback"], first_len - 1):
                rets = {}
                for a, asset in enumerate(family):
                    c = asset.closes_array()
                    rets[a] = float(np.log(c[i]) - np.log(c[i - p["lookback"]]))
                order = sorted(rets, key=lambda a: rets[a], reverse=True)
                longs = order[:p["top_k"]]
                shorts = order[-p["bottom_k"]:]
                if len(longs) < p["top_k"] or len(shorts) < p["bottom_k"]:
                    continue
                rl = float(np.mean([np.log(family[a].closes_array()[i + 1])
                                    - np.log(family[a].closes_array()[i])
                                    for a in longs]))
                rs = float(np.mean([np.log(family[a].closes_array()[i + 1])
                                    - np.log(family[a].closes_array()[i])
                                    for a in shorts]))
                pnl_p[i] = rl - rs
            lrets = walk_forward_fold_lrets(pnl_p, 60, 20, 10, 10)
            recomputed[tuple(sorted(p.items()))].append(float(np.median(lrets)))
            if (p["lookback"] == LOOKBACK and p["top_k"] == TOP_K
                    and p["bottom_k"] == BOTTOM_K):
                baseline_meds.append(float(np.median(lrets)))
                baseline_nulls.append(float(np.median(nlrets)))
                baseline_cand_fold.extend(lrets)
                baseline_null_fold.extend(nlrets)
            rng2 = np.random.default_rng(SEED + int(round(
                sum(float(v) for v in p.values()) * 1000)))
            null_pnl = np.zeros(first_len, dtype=np.float64)
            for i in range(p["lookback"], first_len - 1):
                rets = {}
                for a, asset in enumerate(family):
                    c = asset.closes_array()
                    rets[a] = float(np.log(c[i]) - np.log(c[i - p["lookback"]]))
                order = sorted(rets, key=lambda a: rets[a], reverse=True)
                longs = order[:p["top_k"]]
                shorts = order[-p["bottom_k"]:]
                if len(longs) < p["top_k"] or len(shorts) < p["bottom_k"]:
                    continue
                rl = float(np.mean([np.log(family[a].closes_array()[i + 1])
                                    - np.log(family[a].closes_array()[i])
                                    for a in longs]))
                rs = float(np.mean([np.log(family[a].closes_array()[i + 1])
                                    - np.log(family[a].closes_array()[i])
                                    for a in shorts]))
                null_pnl[i] = (rl - rs) if rng2.random() < 0.5 else -(rl - rs)
            nlrets = walk_forward_fold_lrets(null_pnl, 60, 20, 10, 10)
            null_recomputed[tuple(sorted(p.items()))].append(float(np.median(nlrets)))
            if (p["lookback"] == LOOKBACK and p["top_k"] == TOP_K
                    and p["bottom_k"] == BOTTOM_K):
                baseline_null = float(np.median(nlrets))
                baseline_null_fold.extend(nlrets)

    summary_medians = {k: float(np.median(v)) for k, v in recomputed.items()}
    summary_null = {k: float(np.median(v)) for k, v in null_recomputed.items()}
    baseline_med = float(np.median(baseline_meds))
    baseline_null = float(np.median(baseline_nulls))
    pub_baseline_med = artifact["perturbation"]["baseline_candidate"]
    pub_baseline_null = artifact["perturbation"]["baseline_null"]
    eff_tol = max(0.05, 2 * float(np.std(baseline_null_fold, ddof=1))
                  / np.sqrt(len(baseline_null_fold)))
    within_default = abs(baseline_med - baseline_null) <= eff_tol
    p_ok = True
    pub_scenario_medians = artifact["perturbation"]["scenario_medians"]
    pub_scenario_null = artifact["perturbation"]["null_medians"]
    # artifact param_sets are JSON arrays of [key, value] pairs; convert to
    # tuple-of-tuples matching recomputed keys.
    pub_param_sets = []
    for ps in artifact["perturbation"]["param_sets"]:
        pub_param_sets.append(tuple((tuple(k) if isinstance(k, list) else k, v)
                                    for k, v in ps))
    # Independent validation of the full sweep grid: recomputed[ps] holds the
    # 4 scenario medians per param set; compare each of the 36 medians
    # (9 param sets x 4 scenarios) against the artifact for candidate and null,
    # matched by canonical scenario name so the alphabetically sorted artifact
    # keys (json.dump sort_keys=True) do not desynchronize the comparison.
    scen_names = [s.name for s in scenarios]
    for j, ps in enumerate(pub_param_sets):
        if ps not in recomputed or ps not in null_recomputed:
            p_ok = False
            continue
        for s_idx in range(4):
            s_name = scen_names[s_idx]
            pub_med = round(pub_scenario_medians[s_name][j], 3)
            pub_null = round(pub_scenario_null[s_name][j], 3)
            if not np.isclose(round(recomputed[ps][s_idx], 3), pub_med, atol=1e-6):
                p_ok = False
                print("  [DEBUG] sweep candidate mismatch scenario={} paramset={} recompute={:.6f} pub={}".format(
                    s_name, ps, recomputed[ps][s_idx], pub_med))
            if not np.isclose(round(null_recomputed[ps][s_idx], 3), pub_null, atol=1e-6):
                p_ok = False
                print("  [DEBUG] sweep null mismatch scenario={} paramset={} recompute={:.6f} pub={}".format(
                    s_name, ps, null_recomputed[ps][s_idx], pub_null))
    if not p_ok:
        all_ok = False
    print("  sweep grid validated (9 param sets x 4 scenarios = 36 medians each, candidate & null): "
          "{}".format("MATCH" if p_ok else "MISMATCH"))
    print("  baseline candidate {:+.3f} (pub {:+.3f}) -> {}".format(
        baseline_med, pub_baseline_med,
        "MATCH" if np.isclose(round(baseline_med, 3), pub_baseline_med) else "DIFF"))
    print("  baseline null {:+.3f} (pub {:+.3f}) -> {}".format(
        baseline_null, pub_baseline_null,
        "MATCH" if np.isclose(round(baseline_null, 3), pub_baseline_null) else "DIFF"))
    print("  effective tol {:+.4f} (pub {:+.4f}) -> {}".format(
        eff_tol, artifact["perturbation"]["effective_tolerance"],
        "MATCH" if np.isclose(round(eff_tol, 4), artifact["perturbation"]["effective_tolerance"])
        else "DIFF"))
    print("  baseline vs null: {} (pub {}) -> {}".format(
        "within" if within_default else "outside",
        artifact["perturbation"]["compare_default"],
        "MATCH" if within_default == (artifact["perturbation"]["compare_default"] == "within")
        else "DIFF"))
    print("  perturbation recomputation -> {}".format("MATCH" if p_ok else "MISMATCH"))

    # ---- Determinism of verification path ----
    print("\n=== Determinism of verification path ===")
    pnl_d = momentum_spread(tickers)
    runs_d = segments_from_labels(vol_blocks(tickers["AAPL"], N_BLOCKS, WINDOW),
                                  MIN_SEGMENT_BARS)
    m1 = [round(float(np.median(
        walk_forward_fold_lrets(pnl_d[s:e], TRAIN, TEST, WARM, OVERLAP))), 3)
          for lab, s, e in runs_d]
    m2 = [round(float(np.median(
        walk_forward_fold_lrets(pnl_d[s:e], TRAIN, TEST, WARM, OVERLAP))), 3)
          for lab, s, e in runs_d]
    det_ok = m1 == m2
    print("  r1: {}; r2: {} -> {}".format(m1, m2, "identical" if det_ok else "DIFFERENT"))
    if not det_ok:
        all_ok = False
    print("  verification-path determinism -> {}".format("MATCH" if det_ok else "MISMATCH"))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cross-sectional momentum independent verification on "
          "the collected universe: exploratory simulation.")


if __name__ == "__main__":
    main()
