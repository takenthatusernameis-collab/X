"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Separate implementation path: recomputes every segment's walk-forward median and
the matched coin-flip null from a fresh ``walk_forward`` + ``noise_benchmark``
pipeline (NOT ``stress_segments``), under each predeclared cost cfg, then
compares against the artifact written by the cost-sensitivity check
(`state/check_artifacts/momentum_cost_sensitivity_results.json`). A mismatch
would flag a defect in the check.

Design of the independent path:
- Regime labels reconstructed from ``closes`` via a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments reconstructed from labels via ``min_segment_bars``.
- Candidate medians computed by ``walk_forward`` directly on each segment slice
  with the same cost cfg (no call to ``stress_segments``/``parameter_sweep``).
- Coin-flip null recomputed through the framework's own ``noise_benchmark`` with
  the same cost cfg (a separate code path from ``stress_segments``).
- Verdicts regenerated from FRESH UNROUNDED medians/nulls via the documented
  rules, so 3-dp rounding cannot confuse a verdict decided on unrounded values.
- Cost-robustness classification and edge counts re-derived from fresh values.
- Each (asset, lookback, cost-level) is computed once and cached, then reused by
  both the AMZN/JPM spot gate and the all-asset gate.

This keeps the same inputs (dataset, seed, windows, segments, cost grid) so a
match confirms the check computed from those inputs; a mismatch flags an error
in the check's pipeline.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
CANONICAL_LOOKBACK = 5
SWEEP_VERDICT_TOL = 0.05
SURVIVAL_THRESHOLD = 0.6

COST_GRID = {
    "zero": {},
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.001),
}
COST_LEVELS = ("zero", "realistic", "conservative")

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


def cost_config(level):
    return bt.BacktestConfig(**COST_GRID[level])


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


def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, s, e, lookback, cfg):
    """Fresh (ticker, segment, lookback, cost) result via walk_forward + null.

    Returns (median, null_median, n_folds). No call to stress_segments.
    """
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252, cfg=cfg)
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def regime_verdict(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= SWEEP_VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def is_edge(medians, verdict):
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def classify(E0, ER):
    if E0 == 0:
        return "NO_ZERO_COST_EDGE"
    if ER >= SURVIVAL_THRESHOLD * E0:
        return "COST_ROBUST"
    if ER > 0:
        return "COST_FRAGILE"
    return "COST_ELIMINATED"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS), artifact["lookbacks"]
    assert artifact["cost_grid"] == COST_GRID, "cost grid drift"
    assert artifact["zero_cost_cross_check"] == "MATCH", \
        "artifact's own zero-cost cross-check did not match"

    all_ok = True

    # ---- Precompute every (ticker, lookback, cost) via the fresh path once. ----
    tickers = {}
    for entry in json.load(open(Path.cwd() / "research" / "data" / "manifest.json"))["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(tickers.keys())

    fresh = {}       # (ticker, lb, lv) -> (median, null, n_folds)
    fresh_seglabels = {}
    for t in ticker_order:
        closes = tickers[t].closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        fresh_seglabels[t] = [lab for lab, _, _ in runs]
        for lv in COST_LEVELS:
            cfg = cost_config(lv)
            for lb in LOOKBACKS:
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = segment_result(t, s, e, lb, cfg)
                    meds.append(m)
                    nulls.append(n)
                fresh[(t, lb, lv)] = (meds, nulls)

    # ---- Gate 1: AMZN/JPM spot check (all lookbacks, all cost levels) ----
    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks/costs) ===")
    for asset in ["AMZN", "JPM"]:
        for lv in COST_LEVELS:
            for lb in LOOKBACKS:
                meds, nulls = fresh[(asset, lb, lv)]
                pub = artifact["results"][lv][str(lb)][asset]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                ok = ([round(m, 3) for m in meds] == pub_medians
                      and [round(n, 3) for n in nulls] == pub_nulls
                      and fresh_seglabels[asset] == pub["segments"])
                if not ok:
                    all_ok = False
                    print("  MISMATCH {} lb{} lv{}: med {} vs {} | null {} vs {} | seg {} vs {}".format(
                        asset, lb, lv, [round(m, 3) for m in meds], pub_medians,
                        [round(n, 3) for n in nulls], pub_nulls,
                        fresh_seglabels[asset], pub["segments"]))
        print("  {}: all lookbacks/cost levels -> {}".format(
            asset, "MATCH" if all(
                [round(m, 3) for m in fresh[(asset, lb, lv)][0]] == [round(v, 3) for v in artifact["results"][lv][str(lb)][asset]["medians"]]
                and [round(n, 3) for n in fresh[(asset, lb, lv)][1]] == [round(v, 3) for v in artifact["results"][lv][str(lb)][asset]["null_medians"]]
                and fresh_seglabels[asset] == artifact["results"][lv][str(lb)][asset]["segments"]
                for lv in COST_LEVELS for lb in LOOKBACKS) else "MISMATCH"))

    # ---- Gate 2: all assets, medians/nulls + regenerated verdicts + classification ----
    print("\n=== Gate 2: all-asset fresh medians/nulls, verdict & classification regeneration ===")
    recomputed_edge = {lv: {lb: [] for lb in LOOKBACKS} for lv in COST_LEVELS}
    for t in ticker_order:
        asset_ok = True
        for lv in COST_LEVELS:
            for lb in LOOKBACKS:
                meds, nulls = fresh[(t, lb, lv)]
                pub = artifact["results"][lv][str(lb)][t]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                if [round(m, 3) for m in meds] != pub_medians or [round(n, 3) for n in nulls] != pub_nulls:
                    asset_ok = False
                    all_ok = False
                    print("  [DEBUG] {} lb{} lv{}: fresh med {} null {} vs pub med {} null {}".format(
                        t, lb, lv, [round(m, 3) for m in meds], [round(n, 3) for n in nulls],
                        pub_medians, pub_nulls))
                regen = regime_verdict(meds, nulls)
                if regen != pub["verdict"]:
                    asset_ok = False
                    all_ok = False
                    print("  [DEBUG] {} lb{} lv{}: regenerated verdict {} vs published {} "
                          "(fresh med {} null {})".format(t, lb, lv, regen, pub["verdict"],
                                                          [round(m, 3) for m in meds], [round(n, 3) for n in nulls]))
                if is_edge(meds, regen):
                    recomputed_edge[lv][lb].append(t)
        print("  {:6s}: medians/nulls/verdicts -> {}".format(t, "MATCH" if asset_ok else "MISMATCH"))

    # Re-derive edge counts + classification per level/lookback and compare.
    print("\n=== Gate 2b: edge counts + cost-robustness classification ===")
    for lb in LOOKBACKS:
        for lv in COST_LEVELS:
            mine = sorted(recomputed_edge[lv][lb])
            pub = sorted(artifact["per_lookback"][str(lb)]["{}_edge_assets".format(
                {"zero": "zero", "realistic": "realistic", "conservative": "conservative"}[lv])])
            if mine != pub:
                all_ok = False
                print("  [DEBUG] lb{} lv{}: edge assets {} vs published {}".format(lb, lv, mine, pub))
        E0 = len(recomputed_edge["zero"][lb])
        ER = len(recomputed_edge["realistic"][lb])
        EC = len(recomputed_edge["conservative"][lb])
        regen_cls = classify(E0, ER)
        pub = artifact["per_lookback"][str(lb)]
        cls_ok = (E0 == pub["E_zero"] and ER == pub["E_realistic"]
                  and EC == pub["E_conservative"] and regen_cls == pub["verdict"])
        if not cls_ok:
            all_ok = False
        print("  lookback {}: E0={} ER={} EC={} -> {} (pub E0={} ER={} EC={} {}) {}".format(
            lb, E0, ER, EC, regen_cls, pub["E_zero"], pub["E_realistic"],
            pub["E_conservative"], pub["verdict"], "MATCH" if cls_ok else "MISMATCH"))

    # Primary verdict consistency
    primary = artifact["per_lookback"][str(CANONICAL_LOOKBACK)]["verdict"]
    if primary != artifact["primary_verdict"]:
        all_ok = False
        print("  [DEBUG] primary verdict {} vs artifact primary {}".format(
            primary, artifact["primary_verdict"]))
    print("  primary cost-robustness verdict (lookback {}): {} | artifact primary: {}".format(
        CANONICAL_LOOKBACK, primary, artifact["primary_verdict"]))

    # ---- Gate 3: determinism of the verification path ----
    print("\n=== Gate 3: determinism of the verification path ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    cfg_real = cost_config("realistic")
    m1 = [round(segment_result("AMZN", s, e, 5, cfg_real)[0], 3) for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, cfg_real)[0], 3) for lab, s, e in runs]
    det_ok = (m1 == m2)
    if not det_ok:
        all_ok = False
    print("  AMZN lookback 5 realistic r1: {}; r2: {} -> {}".format(
        m1, m2, "identical" if det_ok else "DIFFERENT"))
    assert det_ok, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, fresh "
              "regenerated verdicts/classification match, and the verification "
              "path is deterministic.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()