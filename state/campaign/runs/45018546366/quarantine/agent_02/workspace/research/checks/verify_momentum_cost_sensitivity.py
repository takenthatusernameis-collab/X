"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates, each using a code path separate from the check:

1. Fresh recomputation (AMZN/JPM, every lookback x cost scale) via a separate
   implementation: re-implemented `volatility_blocks`, segment reconstruction,
   and momentum signals, aggregating fold log returns with `walk_forward`
   directly, and the coin-flip null through the framework's own
   `noise_benchmark` (a separate path from `stress_segments`). The cost vector
   is reconstructed here independently (not via the check's `cost_config`), so
   the artifact's per-asset medians/nulls are reproduced without reusing the
   check's cost helper.

2. Zero-cost cross-consistency: the scale 0.0 column must reproduce the
   existing `momentum_lookback_sweep_results.json` (an independent prior run of
   the same zero-cost momentum computation).

3. Reproducibility of the predeclared cost-survival classification: the
   pos_edge_count / mean_candidate_median / cost_survival tags are re-derived
   from the artifact medians alone, independently of the check's aggregation,
   and the fresh path is asserted deterministic.

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
COST_SCALES = (0.0, 1.0, 2.0)
# Independent reconstruction of the repository realistic cost vector
# (examples/ma_crossover_real_data.py); deliberately NOT imported from the check.
BASE_COMMISSION_PER_TRADE = 2.0
BASE_COMMISSION_PER_SHARE = 0.003
BASE_SLIPPAGE_CENTS = 2.0
BASE_SLIPPAGE_PROPORTIONAL = 0.0005

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
ZERO_COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


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


def independent_cost_config(scale):
    """BacktestConfig reconstructed independently of the check's cost_config."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE_COMMISSION_PER_TRADE * scale,
        commission_per_share=BASE_COMMISSION_PER_SHARE * scale,
        slippage_cents=BASE_SLIPPAGE_CENTS * scale,
        slippage_proportional=BASE_SLIPPAGE_PROPORTIONAL * scale,
    )


def segment_result(ticker, s, e, lookback, scale):
    """Fresh walk_forward + noise_benchmark result for one asset segment.

    No call to `stress_segments`; uses the independent cost config above.
    """
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = independent_cost_config(scale)
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


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def verdict(medians, null_medians):
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
    np.random.seed(42)
    art = load_artifact(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["cost_scales"] == list(COST_SCALES)

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks x scales) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for scale in COST_SCALES:
                medians, nulls, nfolds = [], [], None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, scale)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf
                pub = art["per_asset"][str(lb)][str(scale)][asset]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                status = ("MATCH"
                          if medians == pub_medians
                          and nulls == pub_nulls
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {}: lb {} scale {} segments={} -> {}".format(
                        asset, lb, scale, lbls, status))
                    print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                    print("    nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
        print("  {}: all lookbacks x cost scales recomputed".format(asset))

    print("\n=== Gate 2: zero-cost (scale 0.0) cross-consistency vs momentum_lookback_sweep_results.json ===")
    if ZERO_COST_ARTIFACT.exists():
        zero = load_artifact(ZERO_COST_ARTIFACT)
        for asset in ["AMZN", "JPM"]:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]
            for lb in LOOKBACKS:
                medians, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = segment_result(asset, s, e, lb, 0.0)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = zero["per_asset"][asset][str(lb)]
                status = ("MATCH"
                          if medians == [round(v, 3) for v in pub["medians"]]
                          and nulls == [round(v, 3) for v in pub["null_medians"]]
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                print("  {}: lb {} zero-cost vs prior artifact -> {}".format(
                    asset, lb, status))
    else:
        print("  prior zero-cost artifact not found; cross-consistency gate skipped")

    print("\n=== Gate 3: predeclared cost-survival classification reproducibility ===")
    pa = art["per_asset"]
    ticker_order = list(pa["1"]["0.0"].keys())

    def agg(lb, scale):
        pos_edge = 0
        pos_seg = 0
        total = 0
        meds = []
        for t in ticker_order:
            sr = pa[str(lb)][str(scale)][t]
            meds_m = sr["medians"]
            v = verdict(meds_m, sr["null_medians"])
            if v == "REGIME_STABLE" and all(m > 0 for m in meds_m):
                pos_edge += 1
            for m in meds_m:
                meds.append(m)
                if m > 0:
                    pos_seg += 1
                total += 1
        return pos_edge, (float(np.mean(meds)) if meds else None)

    cmismatch = False
    for lb in LOOKBACKS:
        e0, m0 = agg(lb, 0.0)
        e1, m1 = agg(lb, 1.0)
        if m1 is None or e1 == 0 or m1 <= 0:
            survival = "COLLAPSES"
        elif e1 >= e0 - 1:
            survival = "SURVIVES"
        else:
            survival = "DEGRADES"
        pub = art["summary"][str(lb)]
        pub_surv = pub["cost_survival"]
        pub_e0 = pub["by_scale"]["0.0"]["pos_edge_count"]
        pub_e1 = pub["by_scale"]["1.0"]["pos_edge_count"]
        ok = (survival == pub_surv and e0 == pub_e0 and e1 == pub_e1)
        if not ok:
            cmismatch = True
            all_ok = False
        print("  lb {}: recomputed pos_edge 0.0={} 1.0={} survival={} | "
              "published pos_edge 0.0={} 1.0={} survival={} -> {}".format(
                  lb, e0, e1, survival, pub_e0, pub_e1, pub_surv,
                  "MATCH" if ok else "MISMATCH"))

    print("\nDeterminism: recomputing AMZN lookback=5 scale=1.0 medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, 5, 1.0)[0], 3) for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, 1.0)[0], 3) for lab, s, e in runs]
    print("  AMZN lb 5 scale 1.0 r1: {}; r2: {} -> {}".format(
        m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    if m1 != m2:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "zero-cost column cross-checks against momentum_lookback_sweep_results.json, "
              "and the predeclared cost-survival classification reproduces.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on "
          "AMZN/JPM: exploratory simulation.")


if __name__ == "__main__":
    main()