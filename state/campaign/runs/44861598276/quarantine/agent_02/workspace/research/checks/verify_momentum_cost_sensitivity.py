"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates:

1. Fresh recomputation of the AMZN/JPM segment medians and matched-null
   medians via a separate implementation path (re-implemented
   ``volatility_blocks`` / segment reconstruction / momentum signals,
   aggregating fold log returns with ``walk_forward`` and the framework's
   ``noise_benchmark``), for every lookback (3/5/10) and every predeclared
   cost level (zero/realistic/stress). No call to ``stress_segments``.

2. Zero-cost cross-consistency: the zero-cost column of this artifact is an
   independent rerun of the momentum lookback sweep, so its medians, null
   medians, and verdicts must match
   `state/check_artifacts/momentum_lookback_sweep_results.json` exactly for
   all 10 tickers and all three lookbacks.

3. Classification reproducibility: the per-asset cost-robustness classes,
   lookback-robustness labels, and the aggregate class are re-derived from the
   artifact medians/verdicts using the documented rules, independently of the
   check's own classification code.

4. Cost-grid integrity: the artifact's recorded cost grid must equal the
   predeclared grid (the repository's realistic model and its 2x stress).

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
COST_LEVELS = ("zero", "realistic", "stress")
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
VERDICT_TOL = 0.05


def cost_config(level):
    base = dict(initial_capital=1e6, target_exposure=1.0)
    if level == "zero":
        return bt.BacktestConfig(**base)
    if level == "realistic":
        return bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                                 slippage_cents=2.0, slippage_proportional=0.0005, **base)
    if level == "stress":
        return bt.BacktestConfig(commission_per_trade=4.0, commission_per_share=0.006,
                                 slippage_cents=4.0, slippage_proportional=0.0010, **base)
    raise ValueError(level)


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
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0])
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


def segment_result(bars, s, e, lookback, cfg):
    """Fresh path: walk_forward for the candidate, noise_benchmark for the
    matched null, both under the same cost config."""
    closes = bars.closes_array()
    sig = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = sig[s:e]
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
    return float(np.median(cand_log)), float(noise.baseline_median_log_return), len(res.folds)


def is_positive_edge(medians, verdict):
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def classify(edge_flags):
    e0, er, es = edge_flags["zero"], edge_flags["realistic"], edge_flags["stress"]
    if not e0:
        return "NO_ZERO_COST_EDGE"
    if er and es:
        return "COST_ROBUST"
    if er and not es:
        return "COST_FRAGILE_STRESS"
    return "COST_ELIMINATED"


def classify_aggregate(edge_counts):
    n0, nr, ns = edge_counts["zero"], edge_counts["realistic"], edge_counts["stress"]
    if n0 == 0:
        return "NO_ZERO_COST_EDGE"
    if nr == n0 and ns == n0:
        return "COST_ROBUST"
    if nr == n0 and ns < n0:
        return "COST_FRAGILE_STRESS"
    return "COST_ELIMINATED"


def main():
    np.random.seed(42)
    art = json.load(open(SWEEP_ARTIFACT))
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["cost_levels"] == list(COST_LEVELS)

    all_ok = True

    print("=== Gate 4: cost-grid integrity ===")
    expected = {
        "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                     slippage_cents=0.0, slippage_proportional=0.0),
        "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                          slippage_cents=2.0, slippage_proportional=0.0005),
        "stress": dict(commission_per_trade=4.0, commission_per_share=0.006,
                       slippage_cents=4.0, slippage_proportional=0.0010),
    }
    for level in COST_LEVELS:
        rec = art["cost_grid"][level]
        ok = all(abs(float(rec[k]) - float(v)) < 1e-12 for k, v in expected[level].items())
        if not ok:
            all_ok = False
        print("  {}: {} -> {}".format(level, rec, "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks/costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for level in COST_LEVELS:
                cfg = cost_config(level)
                medians, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = segment_result(bars, s, e, lb, cfg)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["results"][level][str(lb)][asset]
                ok = (medians == [round(v, 3) for v in pub["medians"]]
                      and nulls == [round(v, 3) for v in pub["null_medians"]]
                      and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                print("  {:4s} lb={:2d} {:9s} -> {} | med {} vs {} | null {} vs {}".format(
                    asset, lb, level, "MATCH" if ok else "MISMATCH",
                    medians, pub["medians"], nulls, pub["null_medians"]))

    print("\n=== Gate 2: zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    lb_art = json.load(open(LOOKBACK_ARTIFACT))
    assert lb_art["dataset_id"] == DATASET_ID
    mism = False
    for ticker in lb_art["universe"]["per_asset"]:
        for lb in LOOKBACKS:
            pub = lb_art["per_asset"][ticker][str(lb)]
            got = art["results"]["zero"][str(lb)][ticker]
            ok = (got["medians"] == [round(v, 3) for v in pub["medians"]]
                  and got["null_medians"] == [round(v, 3) for v in pub["null_medians"]]
                  and got["verdict"] == pub["verdict"]
                  and got["segments"] == pub["segments"])
            if not ok:
                mism = True
                all_ok = False
            print("  {:6s} lb={:2d}: med {} vs {} | null {} vs {} | verdict {} vs {} -> {}".format(
                ticker, lb, got["medians"], pub["medians"],
                got["null_medians"], pub["null_medians"],
                got["verdict"], pub["verdict"], "MATCH" if ok else "MISMATCH"))
    if not mism:
        print("  all 10 tickers x 3 lookbacks cross-checked -> MATCH")

    print("\n=== Gate 3: classification reproducibility ===")
    edge_l5 = {level: 0 for level in COST_LEVELS}
    medpos_l5 = {level: 0 for level in COST_LEVELS}
    for ticker in art["universe"]["ticker_order"]:
        entry = art["per_asset"][ticker]["lookbacks"]["5"]
        flags = {}
        medpos = {}
        for level in COST_LEVELS:
            sr = entry[level]
            med = [round(v, 3) for v in sr["medians"]]
            flags[level] = is_positive_edge(med, sr["verdict"])
            medpos[level] = all(m > 0 for m in med)
            if flags[level]:
                edge_l5[level] += 1
            if medpos[level]:
                medpos_l5[level] += 1
        cls = classify(flags)
        ok = (cls == entry["cost_robustness"] and flags == entry["edge_flags"]
              and medpos == entry["median_positive_flags"])
        if not ok:
            all_ok = False
        print("  {:6s}: flags={} medpos={} class={} vs published {} -> {}".format(
            ticker, flags, medpos, cls, entry["cost_robustness"],
            "MATCH" if ok else "MISMATCH"))
    agg = classify_aggregate(edge_l5)
    ok = agg == art["universe"]["aggregate_cost_robustness"]
    if not ok:
        all_ok = False
    print("  aggregate edges l5={} -> {} vs published {} -> {}".format(
        edge_l5, agg, art["universe"]["aggregate_cost_robustness"],
        "MATCH" if ok else "MISMATCH"))
    ok = medpos_l5 == art["universe"]["median_positive_counts_l5"]
    if not ok:
        all_ok = False
    print("  median-positive counts l5={} vs published {} -> {}".format(
        medpos_l5, art["universe"]["median_positive_counts_l5"],
        "MATCH" if ok else "MISMATCH"))

    # Lookback-robustness counts.
    for level in COST_LEVELS:
        n_rob = 0
        for ticker in art["universe"]["ticker_order"]:
            cnt = sum(1 for lb in LOOKBACKS
                      if is_positive_edge(art["results"][level][str(lb)][ticker]["medians"],
                                          art["results"][level][str(lb)][ticker]["verdict"]))
            if cnt >= 2:
                n_rob += 1
        pub = art["universe"]["lookback_robust_counts"][level]
        ok = n_rob == pub
        if not ok:
            all_ok = False
        print("  lookback-ROBUST count cost={:9s}: {} vs published {} -> {}".format(
            level, n_rob, pub, "MATCH" if ok else "MISMATCH"))

    print("\nDeterminism: recomputing AMZN lookback=5 (realistic) twice.")
    bars, dates = bt.load_ticker("AMZN")
    labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    r1 = [round(segment_result(bars, s, e, 5, cost_config("realistic"))[0], 3)
          for lab, s, e in runs]
    r2 = [round(segment_result(bars, s, e, 5, cost_config("realistic"))[0], 3)
          for lab, s, e in runs]
    print("  AMZN lb5 realistic r1: {}; r2: {} -> {}".format(
        r1, r2, "identical" if r1 == r2 else "DIFFERENT"))
    assert r1 == r2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity artifact, "
              "the zero-cost column cross-checks against "
              "momentum_lookback_sweep_results.json, and the classification "
              "reproduces.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()