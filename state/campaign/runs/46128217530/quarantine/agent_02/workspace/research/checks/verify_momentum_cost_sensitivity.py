"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates, all using a fresh code path that does NOT call
`stress_segments` (the check's engine):

1. Fresh recomputation of the candidate medians and the matched coin-flip null
   for AMZN/JPM at every (cost level x lookback). Candidate medians are rebuilt
   from re-implemented `volatility_blocks`, segment reconstruction, momentum
   signals, and `walk_forward` with an explicit per-level `BacktestConfig`; the
   null is rebuilt through the framework's own `noise_benchmark` (a separate code
   path from `stress_segments`) with the SAME per-level `cfg`. Every published
   per-asset, per-level, per-lookback median and null median is compared to the
   artifact.

2. Zero-cost cross-consistency: the artifact's zero-cost column must reproduce
   `momentum_lookback_sweep_results.json` exactly (medians, nulls, verdicts) for
   all 10 tickers x 3 lookbacks -- an independent wiring anchor.

3. Classification reproducibility: both the predeclared verdict-based class and
   the median-based class are re-derived from the artifact medians/verdicts via
   the documented rules, independently of the check's own computation; plus a
   determinism check (AMZN lookback-5 conservative recomputed twice).
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
REFERENCE_LOOKBACK = 5

COST_CONFIGS = {
    "zero": {},
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
}

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


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
    """Fresh candidate medians + matched null for one asset segment/lookback/cost.

    Uses walk_forward (explicit cfg) for the candidate and the framework
    noise_benchmark (same cfg) for the null. No call to stress_segments.
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
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252)
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def verdict(medians, null_medians):
    """Re-implementation of the framework's overall verdict rule."""
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def edge_state(medians, v):
    if v == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if v == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def main():
    np.random.seed(42)
    with open(ARTIFACT) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all cost x lookback) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for level, kw in COST_CONFIGS.items():
            cfg = bt.BacktestConfig(initial_capital=1e6, **kw)
            for lb in LOOKBACKS:
                medians, nulls, nfolds = [], [], None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, cfg)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf
                pub = art["per_asset"][asset]["levels"][level][str(lb)]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                ok = (medians == pub_medians and nulls == pub_nulls
                      and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                print("  {} {:12s} lb{:<2d} -> {}  (med {} vs {}, null {} vs {})".format(
                    asset, level, lb, "MATCH" if ok else "MISMATCH",
                    medians, pub_medians, nulls, pub_nulls))

    print("\n=== Gate 2: zero-cost column cross-consistency vs momentum_lookback_sweep_results.json ===")
    with open(SWEEP_ARTIFACT) as f:
        sweep = json.load(f)
    sw_lp = sweep["per_asset"]
    cmismatch = False
    for ticker in art["universe"]["ticker_order"]:
        for lb in LOOKBACKS:
            pub = sw_lp[ticker][str(lb)]
            mine = art["per_asset"][ticker]["levels"]["zero"][str(lb)]
            ok = (mine["medians"] == [round(v, 3) for v in pub["medians"]]
                  and mine["null_medians"] == [round(v, 3) for v in pub["null_medians"]]
                  and mine["verdict"] == pub["verdict"])
            if not ok:
                cmismatch = True
                all_ok = False
                print("  MISMATCH {} lb{}: mine med={} null={} v={} vs pub med={} null={} v={}"
                      .format(ticker, lb, mine["medians"], mine["null_medians"],
                              mine["verdict"], pub["medians"], pub["null_medians"],
                              pub["verdict"]))
    if not cmismatch:
        print("  all 10 tickers x 3 lookbacks zero-cost columns -> MATCH")

    print("\n=== Gate 3: classification reproducibility (verdict + median rules) ===")
    def verdict_class(states):
        z, c = states["zero"], states["conservative"]
        if z == "EDGE" and c == "EDGE":
            return "COST_ROBUST"
        if z == "EDGE" and c != "EDGE":
            return "COST_ERODED"
        return "NO_EDGE"

    def median_class(meds):
        z, c = meds["zero"], meds["conservative"]
        if all(m > 0 for m in z) and all(m > 0 for m in c):
            return "COST_ROBUST"
        if all(m > 0 for m in z) and not all(m > 0 for m in c):
            return "COST_ERODED"
        return "NO_EDGE"

    rmm = True
    for ticker in art["universe"]["ticker_order"]:
        e = art["per_asset"][ticker]
        # Re-derive edge states from medians+verdicts (not trusting stored edge_state).
        states = {}
        meds5 = {}
        for level in COST_CONFIGS:
            lv = e["levels"][level][str(REFERENCE_LOOKBACK)]
            v = verdict(lv["medians"], lv["null_medians"])
            states[level] = edge_state(lv["medians"], v)
            meds5[level] = lv["medians"]
        vc = verdict_class(states)
        mc = median_class(meds5)
        ok = (vc == e["cost_robustness"] and mc == e["cost_robustness_median_based"])
        if not ok:
            rmm = False
            all_ok = False
        print("  {}: verdict-class {} vs {} | median-class {} vs {} -> {}".format(
            ticker, vc, e["cost_robustness"], mc, e["cost_robustness_median_based"],
            "MATCH" if ok else "MISMATCH"))

    print("\nDeterminism: recomputing AMZN lookback-5 conservative medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    cfg = bt.BacktestConfig(initial_capital=1e6, **COST_CONFIGS["conservative"])
    m1 = [round(segment_result("AMZN", s, e, 5, cfg)[0], 3) for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, cfg)[0], 3) for lab, s, e in runs]
    det_ok = m1 == m2
    print("  AMZN lb5 conservative r1: {}; r2: {} -> {}".format(
        m1, m2, "identical" if det_ok else "DIFFERENT"))
    if not det_ok:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact (candidate "
              "medians + matched null across all cost levels and lookbacks), the "
              "zero-cost column cross-checks against momentum_lookback_sweep_results.json, "
              "and both cost-robustness classes reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()