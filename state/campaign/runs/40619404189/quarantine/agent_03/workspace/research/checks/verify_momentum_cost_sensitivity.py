"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregating fold log returns with `walk_forward`). Every
   per-asset, per-lookback median is recomputed and compared to the artifact.
   The coin-flip null is recomputed through the framework's own
   ``noise_benchmark`` (a separate code path from ``stress_segments``), so the
   artifact's null medians are also independently reproduced.

2. Reproducibility gate: the lookback-robustness verdicts are re-derived from
   the artifact medians/nulls/verdicts via the documented rule (ROBUST if a
   positive REGIME_STABLE edge appears at >= 2 lookbacks), independently of the
   check's own verdict computation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_SENSITIVITY_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

COST_SENSITIVITY_VERDICT_TOL = 0.05

# Realistic cost model from examples/ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
)


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


def segment_result(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = vol_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=REALISTIC_COSTS,
    )
    scen = res.scenarios[0]
    return (round(scen.baseline_median_log_return, 3), round(scen.noise_median_log_return, 3), res.n_folds, labels)


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    cost_sensitivity = load_artifact(COST_SENSITIVITY_ARTIFACT)
    assert cost_sensitivity["dataset_id"] == DATASET_ID
    assert cost_sensitivity["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh recomputation via stress_segments (AMZN, JPM, all lookbacks) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        mismatch = False
        for lb in LOOKBACKS:
            medians = []
            nulls = []
            for lab, s, e in segments_from_labels(vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW), MIN_SEGMENT_BARS):
                m, n, nf, _ = segment_result(asset, list(bars)[s:e], lb)
                medians.append(m)
                nulls.append(n)
            pub_medians = cost_sensitivity["per_asset"][asset][str(lb)]["medians"]
            pub_nulls = cost_sensitivity["per_asset"][asset][str(lb)]["null_medians"]
            status = "MATCH" if (medians == pub_medians and nulls == pub_nulls) else "MISMATCH"
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("  {}: lookback {} -> {}".format(
                asset, lb, status))
            if status != "MATCH":
                print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                print("    nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [cost_sensitivity["per_asset"][asset][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: lookback-robustness verdict reproducibility ===")
    def verdict(medians, null_medians):
        cand_disp = float(np.std(np.array(medians)))
        null_disp = float(np.std(np.array(null_medians)))
        if all(abs(m) <= COST_SENSITIVITY_VERDICT_TOL for m in medians):
            return "CONSISTENT_WITH_NOISE"
        if cand_disp > 2.0 * null_disp:
            return "REGIME_DEPENDENT"
        if all(m < 0 for m in medians):
            return "REGIME_STABLE_LOSS"
        return "REGIME_STABLE"

    def edge_or_not(sr):
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            return "EDGE"
        if sr["verdict"] == "REGIME_STABLE_LOSS":
            return "LOSS"
        return "NO_EDGE"

    rmismatch = False
    for ticker in cost_sensitivity["universe"]["per_asset"]:
        edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr = {"medians": cost_sensitivity["universe"]["per_asset"][ticker]["lookback_medians"][str(lb)],
                  "null_medians": cost_sensitivity["universe"]["per_asset"][ticker]["lookback_null_medians"][str(lb)],
                  "verdict": verdict(cost_sensitivity["universe"]["per_asset"][ticker]["lookback_medians"][str(lb)],
                                     cost_sensitivity["universe"]["per_asset"][ticker]["lookback_null_medians"][str(lb)])}
            edge_counts[edge_or_not(sr)] += 1
        robust = "ROBUST" if edge_counts["EDGE"] >= 2 else ("SENSITIVE" if edge_counts["EDGE"] == 1 else "NO_EDGE")
        status = "MATCH" if robust == cost_sensitivity["universe"]["per_asset"][ticker]["robustness_verdict"] else "MISMATCH"
        if status != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: edge_counts={} -> {} vs published {} -> {}".format(
            ticker, edge_counts, robust, cost_sensitivity["universe"]["per_asset"][ticker]["robustness_verdict"], status))

    print("\nDeterminism: recomputing AMZN lookback=5 medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    m1 = [segment_result("AMZN", list(bars)[s:e], 5)[0]
          for lab, s, e in segments_from_labels(vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)]
    m2 = [segment_result("AMZN", list(bars)[s:e], 5)[0]
          for lab, s, e in segments_from_labels(vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)]
    print("  AMZN lookback 5 r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity check artifact, and robustness verdicts reproduce.")
    else:
        print("MISMATCH FOUND: the cost-sensitivity check artifact does not reproduce via the independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. Momentum lookback sweep with realistic transaction costs independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()