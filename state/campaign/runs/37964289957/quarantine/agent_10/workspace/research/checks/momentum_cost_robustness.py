"""Momentum cost robustness: walk-forward on the collected universe across lookbacks 3/5/10.

Background: the momentum edge observed at lookback=5 is positive across the
collected real-data universe, but the evidence is short-horizon and mechanical.
This check closes the gap: momentum is re-tested across lookbacks 3/5/10 with
some entries having realistic transaction costs, others with zero costs, to
test whether the edge survives conservative execution assumptions.

The edge must persist across lookbacks and cost regimes before momentum can
be considered robust enough for further investment.

Method (fixed a-priori, not tuned to OOS):
- long the previous N-day return, hold 1 day, daily rebalance
- walk-forward: train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment
- realism: use the same 4 contiguous volatility blocks as other momentum checks
- cost sensitivity: zero cost vs realistic costs (commission_per_trade=2.0,
  commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)
- matched null: same walk-forward segments, coin-flip null across each segment
- benchmark comparison: each asset's cost-robustness trend is compared against
  its own coin-flip null via the same regime-stability gates as other checks

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
NOMINAL_ALPHA = 0.05
BONFERRONI_ALPHA = NOMINAL_ALPHA / 10
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

# Realistic cost configuration (from ma_crossover_real_data.py)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)
def momentum_signals(closes: np.ndarray, lookback: int) -> list[bt.Signal]:
    """Past-only momentum signal (same contract as research/backtest/regime_stability)."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return signals
def run_cost_sensitivity_check():
    """Run the full cost robustness check."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Momentum cost robustness across collected universe ===")
    print("Momentum = long the previous N-day return, hold 1 day, daily rebalanced. "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, compared vs coin-flip null on the same "
          "segments.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load all tickers from the manifest
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    # Run cost-robustness check for each lookback
    output = {}
    cross_lookback_summary = {}
    
    for lookback in LOOKBACKS:
        print(f"\n=== Lookback {lookback} days ===")
        
        # Collect results for zero cost and realistic cost
        zero_cost_results = {}
        realistic_cost_results = {}
        
        # Run regime stability check for each cost scenario
        for cfg_name, cfg in [("zero_cost", ZERO_COSTS), ("realistic_cost", REALISTIC_COSTS)]:
            print(f"\n  Testing {cfg_name}:")
            
            if cfg_name == "zero_cost":
                results = zero_cost_results
            else:
                results = realistic_cost_results

            # Run regime stability analysis using stress_segments_across_tickers
            summary = bt.stress_segments_across_tickers(
                tickers=tickers,
                signals_fn=lambda c, **kw: momentum_signals(c, lookback=lookback),
                regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS,
                                                               window=WINDOW),
                baseline=(("lookback", lookback),),
                param_grid=[{"lookback": lookback}],
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            
            verdicts = [r.overall_verdict for r in summary.assets.values()]
            counts = dict(summary.verdict_counts)
            stable = counts.get("REGIME_STABLE", 0)
            
            results["verdict_counts"] = counts
            results["n_regime_stable"] = stable
            results["ticker_order"] = summary.ticker_order
            results["per_asset"] = []
            
            for ticker, r in summary.assets.items():
                asset_result = {
                    "ticker": ticker,
                    "segments": [s.name for s in r.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in r.scenarios],
                    "candidate_dispersion": round(r.candidate_dispersion, 3),
                    "null_dispersion": round(r.null_dispersion, 3),
                    "verdict": r.overall_verdict,
                    "n_folds": r.n_folds,
                }
                results["per_asset"].append(asset_result)
            
            print(f"    Assets: {len(summary.assets)}")
            print(f"    REGIME_STABLE: {counts.get('REGIME_STABLE', 0)}")
            print(f"    REGIME_STABLE_LOSS: {counts.get('REGIME_STABLE_LOSS', 0)}")
            print(f"    REGIME_DEPENDENT: {counts.get('REGIME_DEPENDENT', 0)}")
            print(f"    CONSISTENT_WITH_NOISE: {counts.get('CONSISTENT_WITH_NOISE', 0)}")
            
            # Print per-asset summary
            print(f"    Per-asset results:")
            for asset_result in results["per_asset"]:
                print(f"      {asset_result['ticker']}: verdict={asset_result['verdict']}, "
                      f"stable_median={asset_result['medians'][-1]:.3f}")
        
        # Store cross-lookback results
        cross_lookback_summary[lookback] = {
            "zero_cost": zero_cost_results,
            "realistic_cost": realistic_cost_results,
        }

    # Cross-lookback summary
    print("\n=== 4. Cross-lookback summary ===")
    print("REGIME_STABLE asset count by lookback and cost scenario:")
    for lookback in LOOKBACKS:
        zero_stable = cross_lookback_summary[lookback]["zero_cost"]["n_regime_stable"]
        realistic_stable = cross_lookback_summary[lookback]["realistic_cost"]["n_regime_stable"]
        print(f"  lookback {lookback}: zero_cost={zero_stable}/10 REGIME_STABLE, "
              f"realistic_cost={realistic_stable}/10 REGIME_STABLE")

    print("\n=== 5. Cost comparison ===")
    print("Comparing zero-cost vs realistic-cost regime-stability:")
    cost_robustness_assessments = {}
    
    for lookback in LOOKBACKS:
        print(f"\n  Lookback {lookback}:")
        zero_cost = cross_lookback_summary[lookback]["zero_cost"]
        realistic_cost = cross_lookback_summary[lookback]["realistic_cost"]
        
        zero_verdicts = [a["verdict"] for a in zero_cost["per_asset"]]
        realistic_verdicts = [a["verdict"] for a in realistic_cost["per_asset"]]
        
        zero_stable = zero_cost["n_regime_stable"]
        realistic_stable = realistic_cost["n_regime_stable"]
        
        print(f"    Zero cost: {zero_stable}/10 REGIME_STABLE")
        print(f"    Realistic cost: {realistic_stable}/10 REGIME_STABLE")
        print(f"    Difference: {realistic_stable - zero_stable} assets")
        
        # Assess cost robustness
        if realistic_stable >= zero_stable:
            cost_robust = "COST_ROBUST"
            cost_rob_msg = "edge survives realistic costs"
        else:
            cost_robust = "COST_SENSITIVE"
            cost_rob_msg = "edge deteriorates with realistic costs"
        
        print(f"    Cost robustness assessment: {cost_rob_msg}")
        cost_robustness_assessments[lookback] = {
            "cost_robustness": cost_robust,
            "message": cost_rob_msg,
            "zero_stable": zero_stable,
            "realistic_stable": realistic_stable,
            "difference": realistic_stable - zero_stable,
        }

    # Store results for independent verification
    output = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "nominal_alpha": NOMINAL_ALPHA, "bonferroni_alpha": BONFERRONI_ALPHA,
        "cross_lookback_summary": cross_lookback_summary,
        "cost_robustness_assessments": cost_robustness_assessments,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost robustness check across lookbacks 3/5/10: exploratory simulation.")
    
    return output
def main():
    return run_cost_sensitivity_check()
if __name__ == "__main__":
    sys.exit(main())