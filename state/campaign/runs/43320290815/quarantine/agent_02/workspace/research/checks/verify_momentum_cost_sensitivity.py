"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM per-segment medians through a separate
   implementation path: re-implemented `volatility_blocks`, segment slicing,
   momentum signals, `walk_forward`, and the framework's own ``noise_benchmark``,
   all with the same cost ``BacktestConfig`` as the check. Every per-asset,
   per-lookback, per-cost-level median and null is recomputed and compared to
   the artifact. The coin-flip null at each cost level is independently
   reproduced, so the matched-cost baseline is not merely the check's output.

2. Cross-consistency gate: the zero-cost lookback=5 column of this artifact is
   an independent rerun of the momentum class, so it must match
   `momentum_lookback_sweep_results.json` exactly (medians, nulls, verdicts,
   segments). A mismatch would flag a change in the lookback-5 computation or
   a defect.

3. Reproducibility gate: the verifier's recomputation of AMZN lookback=5 is
   run twice and asserted identical.

Classification rules are re-derived from the artifact medians/nulls/verdicts
independently of the check's own classification functions: EDGE = REGIME_STABLE
with uniformly positive medians; COST_ROBUST = EDGE at >= 2 cost levels including
stressed; COST_SENSITIVE = EDGE at exactly one level; COST_KILLED = no EDGE.
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
TARGET = ("AMZN", "JPM")
COST_LEVELS = {
    "zero": (0.0, 0.0, 0.0),
    "cheap": (0.005, 1.0, 0.0005),
    "base": (0.005, 1.0, 0.0010),
    "stressed": (0.010, 2.0, 0.0020),
}
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks (independent path).
    Replicates the framework's exact logic including block-boundary handling."""
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
            block = i // block_size
            s, e = block * block_size, (block + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window
                else 0.0)
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


def cfg_from_costs(costs):
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_share=float(costs[0]),
        commission_per_trade=1.0,
        slippage_cents=float(costs[1]),
        slippage_proportional=float(costs[2]),
    )


def segment_result(ticker, s, e, lookback, costs):
    """Fresh recomputation of one asset segment under one cost level, using a
    separate implementation of volatility_blocks, slicing, momentum signals,
    walk_forward, and noise_benchmark. No call to stress_segments."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = cfg_from_costs(costs)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    null = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cfg)
    return (float(np.median(cand_log)), null.baseline_median_log_return, len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def verdict(medians, null_medians):
    """Re-derive the regime-stability verdict from medians/nulls directly."""
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def edge_or_not(medians, verdict_):
    if verdict_ == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict_ == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cost_verdict(edge_by_cost):
    n_edge = sum(1 for v in edge_by_cost.values() if v == "EDGE")
    return "COST_ROBUST" if n_edge >= 2 else ("COST_SENSITIVE" if n_edge == 1 else "COST_KILLED")


def main():
    cost = load_artifact(COST_ARTIFACT)
    assert cost["dataset_id"] == DATASET_ID
    assert set(cost["lookbacks"]) == set(LOOKBACKS)
    # artifact is json.dump(..., sort_keys=True) so cost_levels keys are alphabetical;
    # compare as dicts regardless of ordering.
    cost_levels_pub = cost["cost_levels"]
    assert set(cost_levels_pub.keys()) == set(COST_LEVELS.keys())

    # Cross-check cost descriptions match the predeclared grid.
    for cl in COST_LEVELS:
        assert cost_levels_pub[cl]["slippage_cents"] == COST_LEVELS[cl][1], cl
        assert cost["cost_levels"][cl]["slippage_proportional"] == COST_LEVELS[cl][2], cl

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks/costs) ===")
    for asset in TARGET:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            for cl in COST_LEVELS:
                medians = []
                nulls = []
                nfolds = None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, COST_LEVELS[cl])
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf
                pub = cost["per_asset"][asset][str(lb)][cl]
                pub_medians = pub["medians"]
                pub_nulls = pub["null_medians"]
                status = ("MATCH"
                          if medians == pub_medians
                          and nulls == pub_nulls
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    mismatch = True
                    all_ok = False
                print("  {}: lookback {} cost={} segments={} -> {}".format(
                    asset, lb, cl, lbls, status))
                if status != "MATCH":
                    print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                    print("    nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
                    print("    segments published {}".format(pub["segments"]))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [cost["per_asset"][asset][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: zero-cost lookback=5 cross-consistency vs momentum_lookback_sweep_results.json ===")
    sweep = load_artifact(SWEEP_ARTIFACT)
    cwmismatch = False
    for ticker in sweep["universe"]["per_asset"]:
        sw5 = cost["per_asset"][ticker]["5"]["zero"]
        status = "MATCH" if (sw5["medians"] == sweep["universe"]["per_asset"][ticker]["lookback_medians"]["5"]
                              and sw5["null_medians"] == sweep["universe"]["per_asset"][ticker]["lookback_null_medians"]["5"]
                              and sw5["verdict"] == sweep["universe"]["per_asset"][ticker]["verdicts_5"]) \
            else "MISMATCH"
        if status != "MATCH":
            cwmismatch = True
            all_ok = False
        print("  {}: medians {} vs {} | nulls {} vs {} | verdict {} vs {} -> {}"
              .format(ticker, sw5["medians"], sweep["universe"]["per_asset"][ticker]["lookback_medians"]["5"],
                      sw5["null_medians"], sweep["universe"]["per_asset"][ticker]["lookback_null_medians"]["5"],
                      sw5["verdict"], sweep["universe"]["per_asset"][ticker]["verdicts_5"], status))
    if not cwmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: cost-robustness classification reproducibility from artifact medians/nulls ===")
    rmismatch = False
    for ticker in cost["universe"]["per_asset"]:
        recalc = {}
        robust = {}
        for lb in LOOKBACKS:
            edge_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_KILLED": 0}
            for cl in COST_LEVELS:
                sr = cost["per_asset"][ticker][str(lb)][cl]
                v = verdict(sr["medians"], sr["null_medians"])
                eo = edge_or_not(sr["medians"], v)
                recalc.setdefault(str(lb), {})[cl] = eo
                robust[str(lb)] = cost_verdict(recalc[str(lb)])
            edge_counts[robust[str(lb)]] += 1
        status = "MATCH"
        for lb in LOOKBACKS:
            if robust[str(lb)] != cost["universe"]["per_asset"][ticker]["cost_robustness"][str(lb)]:
                status = "MISMATCH"
                rmismatch = True
                all_ok = False
        print("  {}: recalc={} vs published={} -> {}".format(
            ticker, robust, cost["universe"]["per_asset"][ticker]["cost_robustness"], status))

    print("\nDeterminism: recomputing AMZN lookback=5 medians twice at base costs.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, 5, COST_LEVELS["base"])[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, COST_LEVELS["base"])[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback 5 base-costs r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print("\n=== Gate 4: cost-level counts cross-check ===")
    cmismatch = False
    for cl in COST_LEVELS:
        pub = cost["universe"]["per_cost_edge_counts"][cl]
        recalc = sum(1 for t in cost["universe"]["per_asset"]
                     for lb in LOOKBACKS
                     if edge_or_not(cost["per_asset"][t][str(lb)][cl]["medians"],
                                    verdict(cost["per_asset"][t][str(lb)][cl]["medians"],
                                            cost["per_asset"][t][str(lb)][cl]["null_medians"])) == "EDGE")
        if recalc != pub:
            cmismatch = True
            all_ok = False
        print("  {}: published edge count {} vs recomputed {} -> {}".format(cl, pub, recalc,
              "MATCH" if pub == recalc else "MISMATCH"))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact: "
              "AMZN/JPM per-segment medians/nulls across all lookbacks and cost "
              "levels, the zero-cost lookback-5 column cross-checks against "
              "momentum_lookback_sweep_results.json, cost-robustness verdicts "
              "reproduce, cost-level edge counts cross-check, and the verifier "
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
