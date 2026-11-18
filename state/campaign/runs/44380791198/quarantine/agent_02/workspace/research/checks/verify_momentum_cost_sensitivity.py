"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates, all using a fresh recomputation path that does NOT
call `stress_segments`:

1. Fresh walk_forward + noise_benchmark recomputation of AMZN/JPM medians and
   null medians at every (lookback, cost) grid point, compared to the artifact.
   Uses re-implemented `volatility_blocks`, segment reconstruction, momentum
   signals, and `walk_forward` with the cost config threaded through. The null
   is recomputed via the framework's own `noise_benchmark` (a separate code
   path from `stress_segments`).

2. Reproducibility gate: the asset-level cost verdicts are re-derived from the
   artifact medians/verdicts via the documented rule (COST_ROBUST if a
   surviving REGIME_STABLE positive edge appears at >= 2 lookbacks at the
   conservative `comm_500` level), independently of the check's own verdict
   computation.

3. Determinism gate: recompute the AMZN lookback-5 harsh-cost medians twice and
   assert identical.

A mismatch would flag a defect in the check.
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
COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


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


def segment_result(ticker, s, e, lookback, commission_per_trade):
    """Fresh walk_forward + noise_benchmark recomputation for one segment.

    No call to `stress_segments`. The cost config is threaded through both the
    candidate walk_forward and the null noise_benchmark.
    """
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
                            commission_per_trade=float(commission_per_trade),
                            slippage_cents=0.0)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252, cfg=cfg)
    null_log = noise.baseline_median_log_return
    return (float(np.median(cand_log)), null_log, len(res.folds))


# Module-level constants used by segment_result (set in main).
train = test = warm = overlap = 0


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    global train, test, warm, overlap
    train, test, warm, overlap = TRAIN, TEST, WARM, OVERLAP
    np.random.seed(42)
    artifact = load_artifact(COST_ARTIFACT)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)

    # Build a label -> commission map from the artifact's declared grid.
    comm_by_label = {g["label"]: g["commission_per_trade"] for g in artifact["cost_grid"]}
    labels = [g["label"] for g in artifact["cost_grid"]]

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks x costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        lbls_runs = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(lbls_runs, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for label in labels:
            comm = comm_by_label[label]
            for lb in LOOKBACKS:
                medians = []
                nulls = []
                nfolds = None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, comm)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf
                pub = artifact["per_asset"][asset]["cost_profiles"][label][str(lb)]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                status = ("MATCH"
                          if medians == pub_medians
                          and nulls == pub_nulls
                          and lbls == artifact["per_asset"][asset]["cost_profiles"][label][str(lb)]["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    mismatch = True
                    all_ok = False
                    print("  {} lookback {} cost {}: medians {} vs {} | nulls {} vs {} -> {}".format(
                        asset, lb, label, medians, pub_medians, nulls, pub_nulls, status))
        if not mismatch:
            print("  {}: all lookbacks x costs -> MATCH".format(asset))

    print("\n=== Gate 2: asset-level cost verdict reproducibility ===")
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

    def survives(medians):
        return all(m > 0 for m in medians)

    vmismatch = False
    for ticker in artifact["universe"]["per_asset"]:
        tp = artifact["universe"]["per_asset"][ticker]
        # Re-derive per-(lookback,cost) edge status from medians.
        ref_lb = []
        realistic_lb = []
        harsh_lb = []
        for lb in LOOKBACKS:
            for label in labels:
                med = tp["cost_profiles"][label][str(lb)]["medians"]
                v = verdict(med, tp["cost_profiles"][label][str(lb)]["null_medians"])
                edge = (v == "REGIME_STABLE" and survives(med))
                if not edge:
                    continue
                if label == "zero":
                    ref_lb.append(lb)
                if label == "comm_50":
                    realistic_lb.append(lb)
                if label == "comm_500":
                    harsh_lb.append(lb)
        if len(harsh_lb) >= 2:
            regen = "COST_ROBUST"
        elif len(realistic_lb) >= 2:
            regen = "COST_SENSITIVE"
        else:
            regen = "COST_ERASED"
        status = "MATCH" if regen == tp["cost_verdict"] else "MISMATCH"
        if status != "MATCH":
            vmismatch = True
            all_ok = False
        print("  {}: ref={} realistic={} harsh={} -> {} vs published {} -> {}".format(
            ticker, ref_lb, realistic_lb, harsh_lb, regen, tp["cost_verdict"], status))
    if not vmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\nDeterminism: recomputing AMZN lookback=5 comm_500 medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    lbls_runs = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(lbls_runs, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, 5, comm_by_label["comm_500"])[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, comm_by_label["comm_500"])[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback 5 comm_500 r1: {}; r2: {} -> {}".format(
        m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, and the "
              "asset-level cost verdicts reproduce from the artifact medians.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()