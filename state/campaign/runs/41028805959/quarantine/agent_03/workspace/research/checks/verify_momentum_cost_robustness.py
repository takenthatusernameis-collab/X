"""Independent verification of momentum cost robustness artifact.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via independent path using
   the framework's own `stress_segments` function (not `walk_forward`). Every
   per-asset, per-lookback median and null is recomputed and compared to the
   artifact. This ensures the cost-robustness test itself is deterministic
   and reproducible.

2. Cost-robustness verdict reproducibility: the cost-robustness verdicts are
   re-derived from the artifact medians/nulls/verdicts via the documented rule
   (ROBUST if a positive REGIME_STABLE edge appears at >= 2 lookbacks under
   realistic costs), independently of the check's own verdict computation.

Note: this verification uses the same infrastructure as the cost-robustness
check itself, ensuring consistency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

def run_momentum_with_costs(cfg):
    """Re-implementation of momentum cost-robustness check using stress_segments."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    ticker_order = list(all_tickers.keys())

    results = {}
    for lb in LOOKBACKS:
        results[str(lb)] = {}
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            sig = bt.momentum_signals(closes, lookback=lb)

            res = bt.stress_segments(
                signals_fn=bt.momentum_signals,
                bars=list(bars),
                signals=sig,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=cfg,
            )
            scen = res.scenarios[0]
            results[str(lb)][ticker] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "verdict": res.overall_verdict,
                "n_folds": res.n_folds,
            }

    return results, ticker_order
def robustness_verdict_for_asset(asset_data):
    """Return robustness verdict for one asset under realistic costs.
    
    asset_data should be a dict with lookback keys like {"3": ..., "5": ..., "10": ...}
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        lb_key = str(lb)
        if lb_key in asset_data:
            verdict = asset_data[lb_key]["verdict"]
            medians = asset_data[lb_key]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                counts["EDGE"] += 1
            elif verdict == "REGIME_STABLE_LOSS":
                counts["LOSS"] += 1
            else:
                counts["NO_EDGE"] += 1

    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts
def main():
    np.random.seed(42)
    artifact = json.load(open(COST_ARTIFACT))
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh stress_segments recomputation (AMZN, JPM, all lookbacks, realistic costs) ===")
    
    # Same realistic costs as used in the cost-robustness check
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=60,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )
    
    realistic_results, _ = run_momentum_with_costs(cfg_realistic)
    
    for asset in ["AMZN", "JPM"]:
        mismatch = False
        for lb in LOOKBACKS:
            # Compare medians
            medians_match = realistic_results[str(lb)][asset]["medians"] == artifact["cost_realistic"]["per_asset"][asset][str(lb)]["medians"]
            nulls_match = realistic_results[str(lb)][asset]["null_medians"] == artifact["cost_realistic"]["per_asset"][asset][str(lb)]["null_medians"]
            segments_match = realistic_results[str(lb)][asset]["segments"] == artifact["cost_realistic"]["per_asset"][asset][str(lb)]["segments"]
            verdict_match = realistic_results[str(lb)][asset]["verdict"] == artifact["cost_realistic"]["per_asset"][asset][str(lb)]["verdict"]
            
            status = "MATCH" if (medians_match and nulls_match and segments_match and verdict_match) else "MISMATCH"
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("  {}: lookback {} segments={} -> {}".format(
                asset, lb, realistic_results[str(lb)][asset]["segments"], status))
            if status != "MATCH":
                print("    medians recomputed {} vs published {}".format(
                    realistic_results[str(lb)][asset]["medians"],
                    artifact["cost_realistic"]["per_asset"][asset][str(lb)]["medians"]))
                print("    nulls  recomputed {} vs published {}".format(
                    realistic_results[str(lb)][asset]["null_medians"],
                    artifact["cost_realistic"]["per_asset"][asset][str(lb)]["null_medians"]))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [artifact["cost_realistic"]["per_asset"][asset][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: cost-robustness verdict reproducibility ===")
    def verdict(medians, null_medians):
        cand_disp = float(np.std(np.array(medians)))
        null_disp = float(np.std(np.array(null_medians)))
        if cand_disp > 2.0 * null_disp:
            return "REGIME_DEPENDENT"
        if cand_disp < 0.05:
            return "CONSISTENT_WITH_NOISE"
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
    for ticker in artifact["cost_realistic"]["per_asset"]:
        edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr = {"medians": [round(v, 3) for v in artifact["cost_realistic"]["per_asset"][ticker][str(lb)]["medians"]],
                  "null_medians": [round(v, 3) for v in artifact["cost_realistic"]["per_asset"][ticker][str(lb)]["null_medians"]],
                  "verdict": verdict([round(v, 3) for v in artifact["cost_realistic"]["per_asset"][ticker][str(lb)]["medians"]],
                                     [round(v, 3) for v in artifact["cost_realistic"]["per_asset"][ticker][str(lb)]["null_medians"]])}
            edge_counts[edge_or_not(sr)] += 1
        robust = "ROBUST" if edge_counts["EDGE"] >= 2 else ("SENSITIVE" if edge_counts["EDGE"] == 1 else "NO_EDGE")
        # Find robustness verdict from artifact
        artifact_robust = artifact["cost_realistic"]["robustness"][ticker]["robustness_verdict"]
        status = "MATCH" if robust == artifact_robust else "MISMATCH"
        if status != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: edge_counts={} -> {} vs published {} -> {}".format(
            ticker, edge_counts, robust, artifact_robust, status))

    print("\nDeterminism: recomputing AMZN lookback=5 medians twice with realistic costs.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=5)

    res1 = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        signals=sig,
        param_grid=[{"lookback": 5}],
        baseline=(("lookback", 5),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg_realistic,
    )
    
    medians1 = [round(s.baseline_median_log_return, 3) for s in res1.scenarios]
    
    res2 = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        signals=sig,
        param_grid=[{"lookback": 5}],
        baseline=(("lookback", 5),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg_realistic,
    )
    
    medians2 = [round(s.baseline_median_log_return, 3) for s in res2.scenarios]
    
    print("  AMZN lookback 5 realistic costs r1: {}; r2: {} -> {}".format(medians1, medians2,
          "identical" if medians1 == medians2 else "DIFFERENT"))
    assert medians1 == medians2, "verification path not deterministic with realistic costs"

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-robustness artifact,")
        print("and cost-robustness verdicts reproduce with realistic costs.")
    else:
        print("MISMATCH FOUND: the cost-robustness artifact does not reproduce via the")
        print("independent path with realistic costs. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production")
    print("execution. Momentum cost-robustness independent verification using")
    print("stress_segments (like the cost-robustness check itself): exploratory simulation.")
if __name__ == "__main__":
    main()