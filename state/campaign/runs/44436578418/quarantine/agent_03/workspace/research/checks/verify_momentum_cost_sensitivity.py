"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates:
1. Fresh recomputation of AMZN/JPM medians and matched nulls for every
   cost/lookback cell using a separate implementation of volatility blocks,
   segment reconstruction, momentum signals, walk-forward, and the framework's
   noise benchmark.
2. Zero-cost cross-consistency: the zero-cost column must match the corrected
   `momentum_lookback_sweep_results.json` medians, nulls, and verdicts.
3. Classification reproducibility: asset and universe cost-robustness labels
   are re-derived from the artifact medians/verdicts using the predeclared rule.
4. Determinism: AMZN lookback-5 cells are recomputed twice.
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
COST_ORDER = ("zero", "realistic", "conservative")
COST_CONFIGS = {
    "zero": dict(
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    ),
    "realistic": dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "conservative": dict(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    ),
}

COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


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
        ret = float(np.log(closes[i] / closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, s, e, lookback, cost_label):
    bars, dates = bt.load_ticker(ticker)
    seg_bars = list(bars)[s:e]
    closes = np.array([float(b.close) for b in seg_bars], dtype=np.float64)
    sig = momentum_signals(closes, lookback)
    cfg = bt.BacktestConfig(**COST_CONFIGS[cost_label])
    res = bt.walk_forward(
        seg_bars, sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array(
        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0,
        cfg=cfg, periods_per_year=252)
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def lookback_verdict(result):
    if result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in result["medians"]):
        return "EDGE"
    if result["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def classify_asset(cost_results):
    edge_counts = {}
    for cost_label in COST_ORDER:
        edge_counts[cost_label] = sum(
            1 for lb in LOOKBACKS
            if lookback_verdict(cost_results[cost_label][str(lb)]) == "EDGE"
        )
    if edge_counts["zero"] < 2:
        label = "NO_BASELINE_EDGE"
    elif edge_counts["realistic"] >= 2 and edge_counts["conservative"] >= 2:
        label = "COST_ROBUST"
    else:
        label = "COST_FRAGILE"
    return label, edge_counts


def main():
    np.random.seed(42)
    artifact = load_artifact(COST_ARTIFACT)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["signal"]["lookbacks"] == list(LOOKBACKS)
    assert artifact["signal"]["formula"] == "sign(log(close[i] / close[i-lookback]))"
    assert set(artifact["cost_grid"].keys()) == set(COST_ORDER)

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all costs/lookbacks) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for cost_label in COST_ORDER:
            for lb in LOOKBACKS:
                medians = []
                nulls = []
                first_nfolds = None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, cost_label)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    if first_nfolds is None:
                        first_nfolds = nf
                pub = artifact["per_asset"][asset]["costs"][cost_label][str(lb)]
                status = ("MATCH"
                          if medians == pub["medians"]
                          and nulls == pub["null_medians"]
                          and lbls == pub["segments"]
                          and first_nfolds == pub["n_folds"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {} {} lookback {} -> {}".format(
                        asset, cost_label, lb, status))
                    print("    medians recomputed {} vs published {}".format(
                        medians, pub["medians"]))
                    print("    nulls recomputed {} vs published {}".format(
                        nulls, pub["null_medians"]))
                    print("    segments recomputed {} vs published {}".format(
                        lbls, pub["segments"]))
                else:
                    print("  {} {} lookback {} -> MATCH".format(
                        asset, cost_label, lb))

    print("\n=== Gate 2: zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    sweep = load_artifact(SWEEP_ARTIFACT)
    cmismatch = False
    for ticker in sweep["universe"]["per_asset"]:
        for lb in LOOKBACKS:
            pub = artifact["per_asset"][ticker]["costs"]["zero"][str(lb)]
            sw = sweep["per_asset"][ticker][str(lb)]
            status = ("MATCH"
                      if pub["medians"] == sw["medians"]
                      and pub["null_medians"] == sw["null_medians"]
                      and pub["verdict"] == sw["verdict"]
                      else "MISMATCH")
            if status != "MATCH":
                cmismatch = True
                all_ok = False
                print("  {} lookback {} -> {}".format(ticker, lb, status))
                print("    medians {} vs {}".format(
                    pub["medians"], sw["medians"]))
                print("    nulls {} vs {}".format(
                    pub["null_medians"], sw["null_medians"]))
                print("    verdict {} vs {}".format(
                    pub["verdict"], sw["verdict"]))
    if not cmismatch:
        print("  all 10 tickers and all 3 lookbacks cross-checked -> MATCH")

    print("\n=== Gate 3: cost-robustness classification reproducibility ===")
    rmismatch = False
    recomputed_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0,
                         "NO_BASELINE_EDGE": 0}
    for ticker in artifact["universe"]["per_asset"]:
        cost_results = artifact["per_asset"][ticker]["costs"]
        label, edge_counts = classify_asset(cost_results)
        recomputed_counts[label] += 1
        pub_label = artifact["universe"]["per_asset"][ticker]["cost_robustness"]
        pub_counts = artifact["universe"]["per_asset"][ticker]["edge_counts"]
        status = "MATCH" if (label == pub_label and edge_counts == pub_counts) else "MISMATCH"
        if status != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: edges={} -> {} vs published {} edges={} -> {}".format(
            ticker, edge_counts, label, pub_label, pub_counts, status))
    zero_robust = sum(
        1 for ticker in artifact["universe"]["per_asset"]
        if artifact["universe"]["per_asset"][ticker]["edge_counts"]["zero"] >= 2
    )
    if recomputed_counts["COST_ROBUST"] >= 6:
        universe_label = "COST_ROBUST"
    elif zero_robust >= 6:
        universe_label = "COST_FRAGILE"
    elif zero_robust == 0:
        universe_label = "NO_BASELINE_EDGE"
    else:
        universe_label = "MIXED"
    pub_universe = artifact["universe"]["classification"]
    status = "MATCH" if universe_label == pub_universe else "MISMATCH"
    if status != "MATCH":
        rmismatch = True
        all_ok = False
    print("  universe: {} vs published {} -> {}".format(
        universe_label, pub_universe, status))

    print("\n=== Gate 4: determinism (AMZN lookback 5, zero and realistic) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    for cost_label in ("zero", "realistic"):
        m1 = [round(segment_result("AMZN", s, e, 5, cost_label)[0], 3)
              for lab, s, e in runs]
        m2 = [round(segment_result("AMZN", s, e, 5, cost_label)[0], 3)
              for lab, s, e in runs]
        status = "identical" if m1 == m2 else "DIFFERENT"
        if m1 != m2:
            all_ok = False
        print("  {}: r1={} r2={} -> {}".format(cost_label, m1, m2, status))

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost artifact, the "
              "zero-cost column cross-checks against the corrected lookback "
              "sweep, and the cost-robustness labels reproduce.")
    else:
        print("MISMATCH FOUND: the cost artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on "
          "AMZN/JPM + zero-cost cross-consistency: exploratory simulation.")


if __name__ == "__main__":
    main()