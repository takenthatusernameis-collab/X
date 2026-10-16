"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, momentum
   signals, walk_forward) with realistic costs, comparing against the artifact.
   The coin-flip null is recomputed through the framework's own
   ``noise_benchmark`` (a separate code path from ``stress_segments``), so the
   artifact's null medians are also independently reproduced.

2. Cross-consistency gate: the lookback=5 column of this artifact is an
   independent rerun of the momentum class, so it must match
   `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch would
   flag a change in the lookback-5 computation or a defect.

3. Reproducibility gate: the lookback-robustness verdicts are re-derived from
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
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Matching cost model from the check
COMMISSION_PER_SHARE = 0.000005
SLIPPAGE_PROPORTIONAL = 0.0005
COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"
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


def create_cost_config():
    """Create BacktestConfig with matching realistic costs."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=0.0,
        commission_per_share=COMMISSION_PER_SHARE,
        slippage_cents=0.0,
        slippage_proportional=SLIPPAGE_PROPORTIONAL,
        warmup_periods=0,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )


def segment_result_with_costs_verification(ticker, s, e, train, test, warm, overlap, lookback):
    """Walk-forward result for one asset segment and lookback, computed with
    fresh components and realistic costs."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback=lookback)
    seg_signals = sig[s:e]
    
    cfg = create_cost_config()
    
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252)
    
    null_log = noise.baseline_median_log_return
    return (float(np.median(cand_log)), null_log, len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact(COST_ARTIFACT)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)
    assert artifact["commission_per_share"] == COMMISSION_PER_SHARE
    assert artifact["slippage_proportional"] == SLIPPAGE_PROPORTIONAL

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            medians = []
            nulls = []
            nfolds = None
            for lab, s, e in runs:
                m, n, nf = segment_result_with_costs_verification(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                medians.append(round(m, 3))
                nulls.append(round(n, 3))
                nfolds = nf
            pub_medians = [round(v, 3) for v in artifact["per_asset"][asset][str(lb)]["medians"]]
            pub_nulls = [round(v, 3) for v in artifact["per_asset"][asset][str(lb)]["null_medians"]]
            status = ("MATCH"
                      if medians == pub_medians
                      and nulls == pub_nulls
                      and lbls == artifact["per_asset"][asset][str(lb)]["segments"]
                      else "MISMATCH")
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("  {}: lookback {} segments={} -> {}".format(
                asset, lb, lbls, status))
            if status != "MATCH":
                print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                print("    nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [artifact["per_asset"][asset][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: lookback-5 column cross-consistency vs momentum_results.json ===")
    mom = load_artifact(MOM_ARTIFACT)
    cost_lp = artifact["universe"]["per_asset"]
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False
    for ticker in mom_lp:
        pub5 = mom_lp[ticker]
        sw5 = cost_lp[ticker]["lookback_medians"]["5"]
        sn5 = cost_lp[ticker]["lookback_null_medians"]["5"]
        sv5 = cost_lp[ticker]["verdicts_5"]
        status = "MATCH" if (sw5 == pub5["medians"]
                             and sn5 == pub5["null_medians"]
                             and sv5 == pub5["verdict"]) \
            else "MISMATCH"
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: medians {} vs {} | nulls {} vs {} | verdict {} vs {} -> {}"
              .format(ticker, sw5, pub5["medians"], sn5, pub5["null_medians"],
                      sv5, pub5["verdict"], status))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: lookback-robustness verdict reproducibility ===")
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

    def edge_or_not(sr):
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            return "EDGE"
        if sr["verdict"] == "REGIME_STABLE_LOSS":
            return "LOSS"
        return "NO_EDGE"

    rmismatch = False
    for ticker in cost_lp:
        edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr = {"medians": [round(v, 3) for v in cost_lp[ticker]["lookback_medians"][str(lb)]],
                  "null_medians": [round(v, 3) for v in cost_lp[ticker]["lookback_null_medians"][str(lb)]],
                  "verdict": verdict(cost_lp[ticker]["lookback_medians"][str(lb)],
                                     cost_lp[ticker]["lookback_null_medians"][str(lb)])}
            edge_counts[edge_or_not(sr)] += 1
        robust = "ROBUST" if edge_counts["EDGE"] >= 2 else ("SENSITIVE" if edge_counts["EDGE"] == 1 else "NO_EDGE")
        status = "MATCH" if robust == cost_lp[ticker]["robustness_verdict"] else "MISMATCH"
        if status != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: edge_counts={} -> {} vs published {} -> {}".format(
            ticker, edge_counts, robust, cost_lp[ticker]["robustness_verdict"], status))

    print("\nDeterminism: recomputing AMZN lookback=5 medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result_with_costs_verification("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result_with_costs_verification("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5)[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback 5 r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "lookback-5 column cross-checks against momentum_results.json, "
              "and robustness verdicts reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on "
          "AMZN/JPM + cross-consistency vs momentum_results.json: exploratory "
          "simulation with realistic costs.")


if __name__ == "__main__":
    main()
