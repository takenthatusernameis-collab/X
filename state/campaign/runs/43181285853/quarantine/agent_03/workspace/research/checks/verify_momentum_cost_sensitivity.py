"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates:
1. Fresh recomputation of the AMZN/JPM candidate and matched-null segment
   medians for every lookback and cost level via a separate code path
   (re-implemented `volatility_blocks`, segment reconstruction, momentum
   signals, `walk_forward`, and the framework `noise_benchmark` directly, with
   no call to `stress_segments`). Every value is compared to the artifact.
2. Cross-consistency: the artifact's zero-cost lookback-5 candidate/null
   medians reproduce `momentum_lookback_sweep_results.json` exactly (rounded to
   the sweep artifact's 3 decimals), so the cost run's reference column is an
   independent rerun of the earlier momentum computation.
3. Reproducibility: the survival flags and the universe classification are
   re-derived from the artifact medians via the documented a-priori rule and
   compared to the published values.
4. Determinism: AMZN lookback-5 at the realistic cost level is recomputed twice
   and asserted identical.
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
SURVIVE_THRESHOLD = 6
TOL = 1e-6

COST_MODELS = {
    "zero": {},
    "realistic": {
        "commission_per_trade": 2.0,
        "commission_per_share": 0.003,
        "slippage_cents": 2.0,
        "slippage_proportional": 0.0005,
    },
    "conservative": {
        "commission_per_trade": 4.0,
        "commission_per_share": 0.006,
        "slippage_cents": 4.0,
        "slippage_proportional": 0.0010,
    },
}

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def make_cfg(cost_name):
    return bt.BacktestConfig(**COST_MODELS[cost_name])


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window: i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0]) if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        runs.append((cur, start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1: i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(bars, s, e, lookback, cfg):
    seg_bars = list(bars)[s:e]
    closes = np.array([b.close for b in seg_bars], dtype=np.float64)
    sig = momentum_signals(closes, lookback)
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(seg_bars, param_grid=[{"lookback": lookback}],
                            train_window=TRAIN, test_window=TEST, warmup=0,
                            overlap_window=0, cfg=cfg, periods_per_year=252)
    return float(np.median(cand)), float(noise.baseline_median_log_return), len(res.folds)


def close(a, b):
    return abs(float(a) - float(b)) <= TOL


def main():
    with open(ARTIFACT) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks/costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for cost_name in COST_MODELS:
            cfg = make_cfg(cost_name)
            for lb in LOOKBACKS:
                pub = art["per_asset"][cost_name][str(lb)][asset]
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = segment_result(bars, s, e, lb, cfg)
                    meds.append(m)
                    nulls.append(n)
                ok = (lbls == pub["segments"]
                      and len(meds) == len(pub["medians"])
                      and all(close(a, b) for a, b in zip(meds, pub["medians"]))
                      and all(close(a, b) for a, b in zip(nulls, pub["null_medians"])))
                if not ok:
                    all_ok = False
                    print("  MISMATCH {} cost={} lb={}".format(asset, cost_name, lb))
                    print("    segments {} vs {}".format(lbls, pub["segments"]))
                    print("    medians  {} vs {}".format(meds, pub["medians"]))
                    print("    nulls    {} vs {}".format(nulls, pub["null_medians"]))
                else:
                    print("  MATCH {} cost={:12s} lookback={:2d} segments={}".format(
                        asset, cost_name, lb, lbls))
        if all_ok:
            pass

    print("\n=== Gate 2: zero-cost lookback-5 cross-consistency vs momentum_lookback_sweep.json ===")
    with open(SWEEP_ARTIFACT) as f:
        sweep = json.load(f)
    sweep_lp = sweep["universe"]["per_asset"]
    cmismatch = False
    for ticker in sweep_lp:
        pub = sweep_lp[ticker]
        art_sr = art["per_asset"]["zero"]["5"][ticker]
        pub_medians = pub["lookback_medians"]["5"]
        pub_nulls = pub["lookback_null_medians"]["5"]
        cand_ok = all(close(round(v, 3), round(p, 3))
                      for v, p in zip(art_sr["medians"], pub_medians))
        null_ok = all(close(round(v, 3), round(p, 3))
                      for v, p in zip(art_sr["null_medians"], pub_nulls))
        ok = cand_ok and null_ok
        if not ok:
            cmismatch = True
            all_ok = False
        print("  {}: cand {} vs {} | null {} vs {} -> {}".format(
            ticker, [round(v, 3) for v in art_sr["medians"]], pub_medians,
            [round(v, 3) for v in art_sr["null_medians"]], pub_nulls,
            "MATCH" if ok else "MISMATCH"))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: survival flags and universe classification reproducibility ===")
    def survives(medians):
        return all(m > 0 for m in medians)

    def classify(n):
        if n >= SURVIVE_THRESHOLD:
            return "COST_ROBUST"
        if n == 0:
            return "COST_FRAGILE"
        return "PARTIAL"

    rmismatch = False
    for cost_name in COST_MODELS:
        for lb in LOOKBACKS:
            pub = art["universe"][cost_name][str(lb)]
            surv = []
            for ticker, sr in art["per_asset"][cost_name][str(lb)].items():
                if survives(sr["medians"]) != sr["survives"]:
                    rmismatch = True
                    all_ok = False
                if survives(sr["medians"]):
                    surv.append(ticker)
            derived = classify(len(surv))
            ok = (len(surv) == pub["survival_count"]
                  and sorted(surv) == sorted(pub["survival_tickers"])
                  and derived == pub["classification"])
            if not ok:
                rmismatch = True
                all_ok = False
            print("  cost={:12s} lookback={:2d} -> survive {}/10 {} class {} vs published {} -> {}".format(
                cost_name, lb, len(surv), sorted(surv), derived, pub["classification"],
                "MATCH" if ok else "MISMATCH"))
    if not rmismatch:
        print("  all survival flags and classifications reproduce")

    print("\nDeterminism: recomputing AMZN lookback-5 realistic medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    cfg = make_cfg("realistic")
    m1 = [round(segment_result(bars, s, e, 5, cfg)[0], 6) for _, s, e in runs]
    m2 = [round(segment_result(bars, s, e, 5, cfg)[0], 6) for _, s, e in runs]
    print("  r1: {}; r2: {} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    if m1 != m2:
        all_ok = False
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the zero-cost "
              "lookback-5 column cross-checks against momentum_lookback_sweep_results.json, "
              "and the survival/classification rules reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()