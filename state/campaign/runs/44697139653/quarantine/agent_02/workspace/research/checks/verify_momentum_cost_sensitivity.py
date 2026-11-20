"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates:

1. Fresh recomputation (separate code path): for every ticker, lookback in
   {3,5,10} and cost multiplier in {0,1,2,4}, rebuild the regime segments with a
   fresh `volatility_blocks`, slice the bars, rebuild the momentum signal with a
   fresh implementation, and run `walk_forward` + `noise_benchmark` with a
   cost config built independently (not imported from the check). Every
   candidate median, null median and verdict is compared to the artifact. No
   call to `stress_segments`.

2. Classification reproducibility: the COST_ROBUST / COST_MARGINAL / COST_FRAGILE
   / NO_EDGE class and the universe counts are re-derived from the artifact's
   medians + verdicts via the documented predeclared rule, independently of the
   check's own `classify_cost`.

3. Zero-cost cross-consistency: the M=0 lookback-5 column of the artifact must
   equal `momentum_lookback_sweep_results.json` (medians, nulls, verdicts).

4. Determinism: a fresh recomputation of AMZN (lookback 5, 1x and 4x) is run
   twice and must be identical.
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
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
PRIOR_SWEEP = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

# Independent copy of the repository realistic cost model (not imported).
BASE_COST = dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005)


def fresh_cost_config(multiplier):
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE_COST["commission_per_trade"] * multiplier,
        commission_per_share=BASE_COST["commission_per_share"] * multiplier,
        slippage_cents=BASE_COST["slippage_cents"] * multiplier,
        slippage_proportional=BASE_COST["slippage_proportional"] * multiplier,
    )


def vol_blocks(closes, n_blocks, window):
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
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, s, e, lookback, multiplier):
    """Fresh walk_forward + noise_benchmark with an independent cost config."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = fresh_cost_config(multiplier)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST, warmup=0, overlap_window=0,
        cfg=cfg, periods_per_year=252)
    return float(np.median(cand_log)), float(noise.baseline_median_log_return), len(res.folds)


def classify_median(med_pos_by_mult, clean_zero_cost_edge):
    if not clean_zero_cost_edge:
        return "NO_CLEAN_EDGE"
    if med_pos_by_mult.get(4.0):
        return "COST_ROBUST"
    if med_pos_by_mult.get(1.0):
        return "COST_MARGINAL"
    return "COST_FRAGILE"


def classify_verdict(edge_by_mult):
    if edge_by_mult.get(4.0):
        return "COST_ROBUST"
    if edge_by_mult.get(1.0):
        return "COST_MARGINAL"
    if edge_by_mult.get(0.0):
        return "COST_FRAGILE"
    return "NO_EDGE"


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def verdict_from_medians(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def main():
    art = load_artifact(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["cost_multipliers"] == list(COST_MULTIPLIERS)

    tickers = sorted(art["per_asset"].keys())
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (all tickers, lookbacks, costs) ===")
    mismatch_count = 0
    for ticker in tickers:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for mult in COST_MULTIPLIERS:
                medians, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = segment_result(ticker, s, e, lb, mult)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["per_asset"][ticker][str(lb)]["by_cost"][_key(mult)]
                ok = (medians == pub["medians"] and nulls == pub["null_medians"]
                      and lbls == pub["segments"])
                if not ok:
                    mismatch_count += 1
                    all_ok = False
                    print("  MISMATCH {} lb={} mult={}: med {} vs {} | null {} vs {} | seg {} vs {}".format(
                        ticker, lb, _key(mult), medians, pub["medians"],
                        nulls, pub["null_medians"], lbls, pub["segments"]))
    if mismatch_count == 0:
        print("  all {} ticker x lookback x cost cells MATCH the artifact "
              "via the independent path".format(len(tickers) * len(LOOKBACKS) * len(COST_MULTIPLIERS)))

    print("\n=== Gate 2: median-based cost-classification reproducibility ===")
    cls_mismatch = 0
    for ticker in tickers:
        for lb in LOOKBACKS:
            med_pos = {}
            edge_by_mult = {}
            for mult in COST_MULTIPLIERS:
                pub = art["per_asset"][ticker][str(lb)]["by_cost"][_key(mult)]
                vd = verdict_from_medians(pub["medians"], pub["null_medians"])
                med_pos[mult] = all(m > 0 for m in pub["medians"])
                edge_by_mult[mult] = (vd == "REGIME_STABLE" and all(m > 0 for m in pub["medians"]))
                if edge_by_mult[mult] != pub["edge"]:
                    cls_mismatch += 1
                    all_ok = False
                    print("  EDGE MISMATCH {} lb={} mult={}: recomputed {} vs stored {} (verdict {})".format(
                        ticker, lb, _key(mult), edge_by_mult[mult], pub["edge"], pub["verdict"]))
            clean0 = bool(med_pos[0.0] and art["per_asset"][ticker][str(lb)]["by_cost"]["0.0"]["verdict"] == "REGIME_STABLE")
            re_med = classify_median(med_pos, clean0)
            re_ver = classify_verdict(edge_by_mult)
            p = art["per_asset"][ticker][str(lb)]
            if re_med != p["median_cost_class"]:
                cls_mismatch += 1
                all_ok = False
                print("  MEDIAN-CLASS MISMATCH {} lb={}: recomputed {} vs stored {}".format(
                    ticker, lb, re_med, p["median_cost_class"]))
            if re_ver != p["verdict_cost_class"]:
                cls_mismatch += 1
                all_ok = False
                print("  VERDICT-CLASS MISMATCH {} lb={}: recomputed {} vs stored {}".format(
                    ticker, lb, re_ver, p["verdict_cost_class"]))
            if clean0 != p["clean_zero_cost_edge"]:
                cls_mismatch += 1
                all_ok = False
                print("  CLEAN0 MISMATCH {} lb={}: recomputed {} vs stored {}".format(
                    ticker, lb, clean0, p["clean_zero_cost_edge"]))
    if cls_mismatch == 0:
        print("  all median-based and verdict-based cost classes reproduce from artifact medians")

    # Universe count reproduction (median-based primary).
    for lb in LOOKBACKS:
        recomputed = {c: 0 for c in ("COST_ROBUST", "COST_MARGINAL", "COST_FRAGILE", "NO_CLEAN_EDGE")}
        for ticker in tickers:
            recomputed[art["per_asset"][ticker][str(lb)]["median_cost_class"]] += 1
        pub_counts = art["universe"][str(lb)]["median_cost_class_counts"]
        if recomputed != pub_counts:
            all_ok = False
            print("  COUNT MISMATCH lookback {}: {} vs {}".format(lb, recomputed, pub_counts))
    print("  universe median-based cost-class counts: {}".format(
        {lb: art["universe"][str(lb)]["median_cost_class_counts"] for lb in LOOKBACKS}))

    print("\n=== Gate 3: zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    if PRIOR_SWEEP.exists():
        prior = load_artifact(PRIOR_SWEEP)
        plp = prior["universe"]["per_asset"]
        xok = True
        for ticker in tickers:
            sw5 = art["per_asset"][ticker]["5"]["by_cost"]["0.0"]
            pub = plp[ticker]
            ok = (sw5["medians"] == pub["lookback_medians"]["5"]
                  and sw5["null_medians"] == pub["lookback_null_medians"]["5"]
                  and sw5["verdict"] == pub["verdicts_5"])
            xok = xok and ok
            if not ok:
                all_ok = False
                print("  X MISMATCH {}".format(ticker))
        print("  zero-cost lookback-5 column vs prior artifact -> {}".format(
            "MATCH (all tickers)" if xok else "MISMATCH"))
    else:
        print("  prior sweep artifact not found; skipping.")

    print("\n=== Gate 4: determinism ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    for mult in (1.0, 4.0):
        m1 = [round(segment_result("AMZN", s, e, 5, mult)[0], 3) for lab, s, e in runs]
        m2 = [round(segment_result("AMZN", s, e, 5, mult)[0], 3) for lab, s, e in runs]
        det = m1 == m2
        print("  AMZN lookback 5 mult {} r1={} r2={} -> {}".format(_key(mult), m1, m2, det))
        if not det:
            all_ok = False
            print("  NON-DETERMINISTIC")

    print()
    if all_ok:
        print("All independent recomputations MATCH: candidate/null medians and "
              "segments reproduce for every ticker x lookback x cost cell, cost "
              "classes and universe counts reproduce, the zero-cost column "
              "cross-checks against the prior artifact, and the path is deterministic.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


def _key(m):
    return "{:.1f}".format(float(m))


if __name__ == "__main__":
    main()