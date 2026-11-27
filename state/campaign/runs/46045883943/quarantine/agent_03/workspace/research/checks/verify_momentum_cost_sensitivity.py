"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates, each an independent code path from the check:

1. Fresh recomputation of the AMZN/JPM per-segment medians AND coin-flip null
   medians for every (lookback, cost) pair, using re-implemented
   ``volatility_blocks`` / segment reconstruction / momentum signals,
   ``walk_forward`` with a cost cfg, and the framework ``noise_benchmark`` with
   the SAME cost cfg (a separate path from ``stress_segments``). Every value is
   compared to the artifact.

2. Cost=0 cross-consistency gate: the 0 bps / lookback-5 column of this
   artifact is an independent rerun of the zero-cost momentum class, so it must
   match `momentum_results.json` exactly (medians, nulls, verdicts) for all 10
   tickers. A mismatch would flag a change in the baseline computation.

3. Reproducibility gate: the per-asset cost-robustness classification and the
   per-(lookback,cost) edge status are re-derived from the artifact
   medians/nulls/verdicts via the documented rule (COST_ROBUST if a positive
   REGIME_STABLE edge at lookback 5 survives to >= DECISION_BPS), independently
   of the check's own classification code. Determinism is asserted by
   recomputing the AMZN lookback-5 / 25 bps medians twice.
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
COST_BPS = (0, 10, 25, 50, 100)
DECISION_BPS = 25
REFERENCE_LOOKBACK = 5
OUTLIER_TOL = 0.05
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"


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


def cost_cfg(bps, ref_price):
    return bt.BacktestConfig(commission_per_share=(float(bps) / 10000.0) * ref_price)


def segment_result(ticker, s, e, lookback, bps, ref_price):
    """Fresh per-segment candidate median + coin-flip null median under cost."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = cost_cfg(bps, ref_price)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252)
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def regen_verdict(medians, null_medians):
    cand = np.array(medians, dtype=np.float64)
    null = np.array(null_medians, dtype=np.float64)
    cand_disp = float(np.std(cand))
    null_disp = float(np.std(null))
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def edge_status(verdict, medians):
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def main():
    np.random.seed(42)
    sweep = load_artifact(SWEEP_ARTIFACT)
    assert sweep["dataset_id"] == DATASET_ID
    assert sweep["lookbacks"] == list(LOOKBACKS)
    assert sweep["cost_bps"] == list(COST_BPS)
    assert sweep["decision_bps"] == DECISION_BPS

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookback x cost) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        ref_price = float(np.median(closes))
        pub_ref = sweep["per_asset"][asset]["ref_price"]
        if round(ref_price, 4) != pub_ref:
            all_ok = False
            print("  [MISMATCH] {} ref_price {} vs published {}".format(
                asset, round(ref_price, 4), pub_ref))
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            for bps in COST_BPS:
                medians, nulls = [], []
                nfolds = None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, bps, ref_price)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf
                pub = sweep["per_asset"][asset]["by_lookback"][str(lb)][str(bps)]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                status = ("MATCH"
                          if medians == pub_medians
                          and nulls == pub_nulls
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    mismatch = True
                    all_ok = False
                    print("  {} lb={} bps={} -> {}".format(asset, lb, bps, status))
                    print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                    print("    nulls   recomputed {} vs published {}".format(nulls, pub_nulls))
        if not mismatch:
            print("  {}: all 15 (lookback x cost) median/null columns -> MATCH".format(asset))

    print("\n=== Gate 2: cost=0 lookback-5 cross-consistency vs momentum_results.json ===")
    mom = load_artifact(MOM_ARTIFACT)
    sweep_lp = sweep["per_asset"]
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False
    for ticker in mom_lp:
        pub5 = mom_lp[ticker]
        sw5 = sweep_lp[ticker]["by_lookback"]["5"]["0"]
        status = "MATCH" if (sw5["medians"] == pub5["medians"]
                             and sw5["null_medians"] == pub5["null_medians"]
                             and sw5["verdict"] == pub5["verdict"]) else "MISMATCH"
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: medians {} vs {} | verdict {} vs {} -> {}".format(
            ticker, sw5["medians"], pub5["medians"], sw5["verdict"],
            pub5["verdict"], status))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: cost-robustness classification reproducibility ===")
    def classify(by_lookback):
        ref = by_lookback[str(REFERENCE_LOOKBACK)]
        e0 = edge_status(ref[str(COST_BPS[0])]["verdict"],
                         ref[str(COST_BPS[0])]["medians"]) == "EDGE"
        if not e0:
            return "NO_EDGE"
        survivors = [b for b in COST_BPS
                     if edge_status(ref[str(b)]["verdict"], ref[str(b)]["medians"]) == "EDGE"]
        surv = max(survivors) if survivors else None
        return "COST_ROBUST" if (surv is not None and surv >= DECISION_BPS) else "COST_FRAGILE"

    rreclass_mismatch = False
    counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    for ticker in sweep_lp:
        by_lookback = sweep_lp[ticker]["by_lookback"]
        derived = classify(by_lookback)
        counts[derived] += 1
        status = "MATCH" if derived == sweep_lp[ticker]["classification"] else "MISMATCH"
        if status != "MATCH":
            rreclass_mismatch = True
            all_ok = False
        print("  {}: derived {} vs published {} -> {}".format(
            ticker, derived, sweep_lp[ticker]["classification"], status))
    pub_counts = sweep["universe"]["classification_counts"]
    counts_status = "MATCH" if counts == pub_counts else "MISMATCH"
    if counts != pub_counts:
        all_ok = False
    print("  classification counts derived {} vs published {} -> {}".format(
        counts, pub_counts, counts_status))

    print("\nDeterminism: recomputing AMZN lookback 5 / 25 bps medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    ref_price = float(np.median(closes))
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, 5, 25, ref_price)[0], 3) for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, 25, ref_price)[0], 3) for lab, s, e in runs]
    det_ok = m1 == m2
    if not det_ok:
        all_ok = False
    print("  AMZN lb5/25bps r1: {}; r2: {} -> {}".format(
        m1, m2, "identical" if det_ok else "DIFFERENT"))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the cost=0 "
              "column cross-checks against momentum_results.json, and the cost-"
              "robustness classification reproduces.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()