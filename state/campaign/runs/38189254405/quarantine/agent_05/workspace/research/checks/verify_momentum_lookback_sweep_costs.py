"""Independent cost-sensitivity verification of momentum lookback sweep (3/5/10).

This verification computes the momentum edge under realistic transaction costs
and compares it against the matched null benchmark, answering the primary question:
"Does the short-horizon momentum edge survive conservative transaction-cost stress?"

Design:
- Loads the existing lookback sweep artifact (momentum_lookback_sweep_results.json)
- Recomputes per-asset, per-lookback medians via walk_forward with realistic costs
- Compares candidate medians against coin-flip null benchmarks (using noise_benchmark)
- Applies the same robustness criteria as the lookback sweep to classify cost robustness
- Performs three verification gates similar to other checks:
  1. Fresh recomputation with realistic costs matches artifact expectations
  2. Cross-consistency with lookback-5 column of the artifact
  3. Reproducibility of cost-robustness verdicts
- Reports whether the momentum edge survives realistic costs (ROBUST/SENSITIVE/NO_EDGE)

The realistic cost model is the BacktestConfig from research/backtest/engine.py:
commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005
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

# Realistic cost configuration (matching BacktestConfig from engine.py)
REALISTIC_COSTS = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    initial_capital=1e6,
    warmup_periods=WARM,
)

SWEEP_ARTIFACT = Path("/home/runner/work/X/X/state/check_artifacts/momentum_lookback_sweep_results.json")
MOM_ARTIFACT = Path("/home/runner/work/X/X/state/check_artifacts/momentum_results.json")

COST_VERDICT_TOL = 0.05


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


def segment_cost_result(ticker, s, e, train, test, warm, overlap, lookback):
    """Walk-forward regime-stability result with realistic costs for one asset segment and lookback.
    
    Computed with fresh components: volatility_blocks, segment slicing,
    momentum signals, walk_forward (with realistic costs), and the framework noise_benchmark
    (using default costs). No call to stress_segments.
    """
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=REALISTIC_COSTS)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    # Coin-flip null benchmark uses default costs (zero cost) for fair comparison
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
    sweep = load_artifact(SWEEP_ARTIFACT)
    assert sweep["dataset_id"] == DATASET_ID
    assert sweep["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: Fresh recomputation with realistic costs (AMZN, JPM, all lookbacks) ===")
    print(f"  Realistic cost config: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    
    cost_medians_by_asset = {}
    cost_nulls_by_asset = {}
    
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
                m, n, nf = segment_cost_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                medians.append(round(m, 3))
                nulls.append(round(n, 3))
                nfolds = nf
            
            # Get published (zero-cost) medians and nulls from the lookback sweep artifact
            pub_medians = [round(v, 3) for v in sweep["per_asset"][asset][str(lb)]["medians"]]
            pub_nulls = [round(v, 3) for v in sweep["per_asset"][asset][str(lb)]["null_medians"]]
            
            status = ("MATCH"
                      if medians == pub_medians
                      and nulls == pub_nulls
                      and lbls == sweep["per_asset"][asset][str(lb)]["segments"]
                      else "MISMATCH")
            
            if status != "MATCH":
                mismatch = True
                all_ok = False
                
            print("  {}: lookback {} with realistic costs segments={} -> {}".format(
                asset, lb, lbls, status))
            
            if status != "MATCH":
                print("    cost medians recomputed {} vs published {}".format(medians, pub_medians))
                print("    cost nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
                
            # Store for later analysis
            if asset not in cost_medians_by_asset:
                cost_medians_by_asset[asset] = {}
                cost_nulls_by_asset[asset] = {}
            cost_medians_by_asset[asset][str(lb)] = medians
            cost_nulls_by_asset[asset][str(lb)] = nulls
            
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [sweep["per_asset"][asset][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: Cross-consistency with lookback-5 column of lookback sweep artifact ===")
    mom = load_artifact(MOM_ARTIFACT)
    sweep_lp = sweep["universe"]["per_asset"]
    mom_lp = mom["universe"]["per_asset"]
    
    cmismatch = False
    for ticker in mom_lp:
        pub5 = mom_lp[ticker]
        sw5 = sweep_lp[ticker]["lookback_medians"]["5"]
        sn5 = sweep_lp[ticker]["lookback_null_medians"]["5"]
        sv5 = sweep_lp[ticker]["verdicts_5"]
        
        status = "MATCH" if (sw5 == pub5["medians"]
                             and sn5 == pub5["null_medians"]
                             and sv5 == pub5["verdict"]) \
            else "MISMATCH"
        
        if status != "MATCH":
            cmismatch = True
            all_ok = False
            
        print("  {}: lookback 5 medians {} vs {} | nulls {} vs {} | verdict {} vs {} -> {}"
              .format(ticker, sw5, pub5["medians"], sn5, pub5["null_medians"],
                      sv5, pub5["verdict"], status))
    
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: Cost-robustness verdict reproducibility ===")
    def verdict(medians, null_medians):
        cand_disp = float(np.std(np.array(medians)))
        null_disp = float(np.std(np.array(null_medians)))
        if all(abs(m) <= COST_VERDICT_TOL for m in medians):
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

    def cost_robustness_verdict(asset_results):
        """Assess cost robustness using the same rule as lookback sweep."""
        counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            lb_str = str(lb)
            if lb_str not in asset_results:
                continue
            sr = {"medians": asset_results[lb_str]["medians"],
                  "null_medians": asset_results[lb_str]["null_medians"],
                  "verdict": verdict(asset_results[lb_str]["medians"], asset_results[lb_str]["null_medians"])}
            counts[edge_or_not(sr)] += 1
        
        if counts["EDGE"] >= 2:
            return "ROBUST", counts
        if counts["EDGE"] == 1:
            return "SENSITIVE", counts
        return "NO_EDGE", counts

    cost_robustness_mismatch = False
    
    print("\n=== Cost-robustness analysis ===")
    for ticker in sweep_lp:
        # Combine zero-cost and cost results for this ticker
        asset_results = {
            "cost": {"medians": cost_medians_by_asset.get(ticker, {}),
                    "null_medians": cost_nulls_by_asset.get(ticker, {})},
            "zero": {str(lb): {"medians": sweep_lp[ticker]["lookback_medians"][str(lb)],
                              "null_medians": sweep_lp[ticker]["lookback_null_medians"][str(lb)]}
                     for lb in LOOKBACKS}
        }
        
        # Get zero-cost robustness verdict
        zero_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr = {"medians": asset_results["zero"][str(lb)]["medians"],
                  "null_medians": asset_results["zero"][str(lb)]["null_medians"],
                  "verdict": verdict(asset_results["zero"][str(lb)]["medians"], asset_results["zero"][str(lb)]["null_medians"])}
            zero_counts[edge_or_not(sr)] += 1
        
        zero_robust = "ROBUST" if zero_counts["EDGE"] >= 2 else ("SENSITIVE" if zero_counts["EDGE"] == 1 else "NO_EDGE")
        
        # Get cost robustness verdict (using only cost results if available)
        if ticker in cost_medians_by_asset:
            cost_robust, cost_counts = cost_robustness_verdict({
                str(lb): {"medians": cost_medians_by_asset[ticker][str(lb)],
                         "null_medians": cost_nulls_by_asset[ticker][str(lb)]}
                for lb in LOOKBACKS})
            cost_mismatch = cost_robust != sweep_lp[ticker]["robustness_verdict"]
            if cost_mismatch:
                cost_robustness_mismatch = True
                all_ok = False
            
            print("  {}: zero-cost robustness {} | cost robustness {} -> {} vs {} -> {}"
                  .format(ticker, zero_robust, cost_robust, cost_counts, sweep_lp[ticker]["robustness_verdict"], "MATCH" if not cost_mismatch else "MISMATCH"))
        else:
            print("  {}: zero-cost robustness {} | no cost data available".format(
                ticker, zero_robust))

    print("\n=== Determinism: recomputing AMZN cost lookback=5 medians twice ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = []
    for lab, s, e in runs:
        m, _, _ = segment_cost_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5)
        m1.append(round(m, 3))
    
    m2 = []
    for lab, s, e in runs:
        m, _, _ = segment_cost_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5)
        m2.append(round(m, 3))
    
    print("  AMZN cost lookback 5 r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "cost verification path not deterministic"

    print("\n=== Summary ===")
    if all_ok:
        print("SUCCESS: All independent recomputations with realistic costs MATCH the")
        print("lookback sweep artifact expectations, the lookback-5 column cross-checks")
        print("against momentum_results.json, and cost-robustness verdicts reproduce.")
        print("\nCONCLUSION: The momentum lookback sweep results are REPRODUCIBLE under")
        print("realistic transaction costs.")
        
        if cost_robustness_mismatch:
            print("\nNOTE: Some assets show different cost-robustness verdicts compared to")
            print("the zero-cost baseline, indicating transaction costs affect the")
            print("robustness classification for those assets.")
        else:
            print("\nNOTE: Cost-robustness classifications match the zero-cost baseline,")
            print("indicating transaction costs do not change the robustness verdicts.")
    else:
        print("FAILURE: MISMATCH FOUND - the lookback sweep artifact does not reproduce")
        print("via the independent cost-sensitivity path. Investigate before admitting")
        print("the momentum lookback sweep results with realistic costs.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum lookback sweep independent cost-verification on")
    print("      AMZN/JPM + cross-consistency vs momentum_results.json: exploratory")
    print("      simulation.")

if __name__ == "__main__":
    main()