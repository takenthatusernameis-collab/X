"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, momentum signals,
   and walk_forward with realistic cost). Every per-asset, per-lookback median is
   recomputed and compared to the artifact. The cost model is the same as the
   check's realistic cost configuration, so the relative impact should match.

2. Cross-consistency gate: the lookback=5 column of this artifact must match
   `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch would
   flag a change in the lookback-5 computation or a defect.

3. Reproducibility gate: the cost-impacts (realistic cost - zero cost) and cost ratios
   are re-derived from the artifact medians via the documented rule
   (cost_impact = realistic_mean_ret - zero_mean_ret), independently of the
   check's own walk_forward computation.
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
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

COST_IMPACT_TOL = 0.005

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


def segment_result(ticker, s, e, train, test, warm, overlap, lookback, cfg_zero, cfg_realistic):
    """Walk-forward regime-stability result for one asset segment and lookback,
    computed with fresh components: volatility_blocks, segment slicing,
    momentum signals, walk_forward, and the framework noise_benchmark. No call
    to stress_segments."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    
    res_zero = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_zero)
    res_realistic = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_realistic)
    
    zero_mean = np.mean([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res_zero.folds])
    realistic_mean = np.mean([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                            for f in res_realistic.folds])
    
    cost_impact = realistic_mean - zero_mean
    
    return (float(zero_mean), float(realistic_mean), cost_impact, len(res_zero.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    cost = load_artifact(COST_ARTIFACT)
    assert cost["dataset_id"] == DATASET_ID
    assert cost["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    # Define cost configs (same as in the cost check)
    cfg_zero = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    print("=== Gate 1: fresh walk_forward with cost configs (AMZN, JPM, all lookbacks) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            zero_means = []
            realistic_means = []
            cost_impacts = []
            nfolds = None
            for lab, s, e in runs:
                z, r, ci, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb, cfg_zero, cfg_realistic)
                zero_means.append(round(z, 3))
                realistic_means.append(round(r, 3))
                cost_impacts.append(round(ci, 6))
                nfolds = nf
            
            # Get published values from cost artifact
            pub_zero = [round(v, 3) for v in cost["lookback_results"][str(lb)][asset][0]["zero_cost"]["mean_log_return"]
                       for i, (lab, _, _) in enumerate(runs)]
            pub_realistic = [round(v, 3) for v in cost["lookback_results"][str(lb)][asset][0]["realistic_cost"]["mean_log_return"]
                           for i, (lab, _, _) in enumerate(runs)]
            pub_cost_impact = [round(v, 6) for v in cost["lookback_results"][str(lb)][asset][0]["cost_impact"]["absolute_impact"]
                              for i, (lab, _, _) in enumerate(runs)]
            
            zero_match = zero_means == pub_zero
            realistic_match = realistic_means == pub_realistic
            cost_match = cost_impacts == pub_cost_impact
            
            status = "MATCH" if zero_match and realistic_match and cost_match else "MISMATCH"
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("  {}: lookback {} segments={} -> {}".format(
                asset, lb, lbls, status))
            if status != "MATCH":
                print("    zero_means recomputed {} vs published {}".format(zero_means, pub_zero))
                print("    realistic_means recomputed {} vs published {}".format(realistic_means, pub_realistic))
                print("    cost_impacts recomputed {} vs published {}".format(cost_impacts, pub_cost_impact))
        if mismatch:
            print("  [DEBUG] {} runs {}".format(asset, [runs]))

    print("\n=== Gate 2: lookback-5 column cross-consistency vs momentum_results.json ===")
    mom = load_artifact(MOM_ARTIFACT)
    cost_lp = cost["lookback_results"]["5"]
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False
    for ticker in mom_lp:
        pub5 = mom_lp[ticker]
        c5 = cost_lp[ticker][0]  # Use first segment's values for lookback-5
        status = "MATCH" if (c5["zero_cost"]["mean_log_return"] == pub5["medians"][0]
                             and c5["realistic_cost"]["mean_log_return"] == pub5["null_medians"][0]
                             and ("REGIME_STABLE" if c5["zero_cost"]["mean_log_return"] > 0 else "CONSISTENT_WITH_NOISE") == pub5["verdict"]) \
            else "MISMATCH"
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: zero_cost {} vs {} | realistic_cost {} vs {} | verdict ? vs {} -> {}"
              .format(ticker, c5["zero_cost"]["mean_log_return"], pub5["medians"][0],
                      c5["realistic_cost"]["mean_log_return"], pub5["null_medians"][0],
                      pub5["verdict"], status))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: cost-impacts and cost ratios reproducibility ===")
    rmismatch = False
    for ticker in cost_lp:
        for lb in LOOKBACKS:
            # Get published values
            pub = cost_lp[ticker][0][lb - 3]  # Index by lookback
            published_cost_impact = pub["cost_impact"]["absolute_impact"]
            published_cost_ratio = pub["cost_impact"]["cost_ratio"]
            
            # Recompute from first segment
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            if not runs:
                continue
            
            lab, s, e = runs[0]  # Use first segment
            z, r, ci, _ = segment_result(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb, cfg_zero, cfg_realistic)
            recomputed_cost_impact = ci
            recomputed_cost_ratio = r / z if z != 0 else float('inf')
            
            cost_impact_match = abs(recomputed_cost_impact - published_cost_impact) < COST_IMPACT_TOL
            cost_ratio_match = abs(recomputed_cost_ratio - published_cost_ratio) < 0.001 if published_cost_ratio != float('inf') else recomputed_cost_ratio == float('inf')
            
            status = "MATCH" if cost_impact_match and cost_ratio_match else "MISMATCH"
            if status != "MATCH":
                rmismatch = True
                all_ok = False
            print("  {}: lookback {} cost_impact {} vs {} -> {}".format(
                ticker, lb, recomputed_cost_impact, published_cost_impact, status))
            if status != "MATCH":
                print("    cost_ratio {} vs {}".format(recomputed_cost_ratio, published_cost_ratio))

    print("\nDeterminism: recomputing AMZN lookback=5 cost twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    ci1 = None
    ci2 = None
    if runs:
        lab, s, e = runs[0]
        ci1 = segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, cfg_zero, cfg_realistic)[2]
        ci2 = segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, cfg_zero, cfg_realistic)[2]
    print("  AMZN lookback 5 r1: {}; r2: {} -> {}".format(ci1, ci2,
          "identical" if ci1 == ci2 else "DIFFERENT"))
    assert ci1 == ci2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity check artifact, the "
              "lookback-5 column cross-checks against momentum_results.json, "
              "and cost-impacts/reproducibility reproduce.")
    else:
        print("MISMATCH FOUND: the cost-sensitivity check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity independent verification on "
          "AMZN/JPM + cross-consistency vs momentum_results.json: exploratory "
          "simulation.")


if __name__ == "__main__":
    main()
