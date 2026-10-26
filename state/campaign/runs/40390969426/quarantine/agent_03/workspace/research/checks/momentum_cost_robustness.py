"""Cost-sensitivity check for lookback 3/5/10 momentum.

Runs bounded cost-sensitivity for lookback 3/5/10 momentum under the
repository's existing realistic cost model and matched null.

Objective: Answer whether the short-horizon momentum edge survives
conservative transaction-cost stress on the collected real-data universe.

Scope: Use the existing 10-asset collected universe and existing momentum
implementation/check framework.

Note: research/simulation only. No live trading or production execution.
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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

# The existing realistic cost model from the examples
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero cost baseline for comparison
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
)


def main() -> int:
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

    # Load all tickers from the manifest
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("\n=== 2. Cost-sensitivity check for lookback 3/5/10 momentum ===")
    print("Momentum: long the previous lookback-day return, hold 1 day, daily "
          "rebalanced. Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, compared vs coin-flip null on the same segments\n".format(
              LOOKBACKS[1], TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (turbulent = block median trailing-{}-bar realized vol "
          "> series-wide median)\n".format(N_BLOCKS, WINDOW))

    results = {}

    for lookback in LOOKBACKS:
        print(f"\n--- Lookback = {lookback} ---")
        
        # Get momentum signals for this lookback
        signals_fn = lambda c, **kw: bt.momentum_signals(c, lookback=lookback)
        
        # Run stress segments with realistic costs
        print(f"  Realistic costs (commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
              f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
              f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
              f"slippage_proportional={REALISTIC_COSTS.slippage_proportional})")
        
        realistic_res = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=signals_fn,
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lookback),),
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=REALISTIC_COSTS,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        
        # Run stress segments with zero costs
        print(f"  Zero costs")
        zero_res = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=signals_fn,
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lookback),),
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=ZERO_COSTS,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        
        # Collect results
        realistic_data = []
        zero_cost_data = []
        
        for ticker in realistic_res.ticker_order:
            realistic_asset = realistic_res.assets[ticker]
            zero_cost_asset = zero_res.assets[ticker]
            
            realistic_data.append({
                "ticker": ticker,
                "segments": [s.name for s in realistic_asset.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in realistic_asset.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in realistic_asset.scenarios],
                "candidate_dispersion": round(realistic_asset.candidate_dispersion, 3),
                "null_dispersion": round(realistic_asset.null_dispersion, 3),
                "verdict": realistic_asset.overall_verdict,
                "n_folds": realistic_asset.n_folds,
            })
            
            zero_cost_data.append({
                "ticker": ticker,
                "segments": [s.name for s in zero_cost_asset.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in zero_cost_asset.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in zero_cost_asset.scenarios],
                "candidate_dispersion": round(zero_cost_asset.candidate_dispersion, 3),
                "null_dispersion": round(zero_cost_asset.null_dispersion, 3),
                "verdict": zero_cost_asset.overall_verdict,
                "n_folds": zero_cost_asset.n_folds,
            })
        
        # Compute baseline_median_log_return as the average of segment medians
        realistic_baseline_median = sum(s["medians"][-1] for s in realistic_data) / len(realistic_data) if realistic_data else 0
        zero_baseline_median = sum(s["medians"][-1] for s in zero_cost_data) / len(zero_cost_data) if zero_cost_data else 0
        
        results[f"lookback_{lookback}"] = {
            "realistic_costs": {
                "assets": realistic_data,
                "universe_summary": {
                    "segments": realistic_res.ticker_order,
                    "verdict_counts": realistic_res.verdict_counts,
                    "ticker_order": realistic_res.ticker_order,
                },
                "cost_impact": {
                    "median_reduction": round(
                        zero_baseline_median - realistic_baseline_median,
                        4
                    ),
                    "median_reduction_pct": round(
                        (zero_baseline_median - realistic_baseline_median) /
                        zero_baseline_median * 100.0,
                        2
                    ) if zero_baseline_median != 0 else None,
                }
            },
            "zero_costs": {
                "assets": zero_cost_data,
                "universe_summary": {
                    "segments": zero_res.ticker_order,
                    "verdict_counts": zero_res.verdict_counts,
                    "ticker_order": zero_res.ticker_order,
                },
            },
        }
        
        # Determine overall verdict from verdict counts
        def get_overall_verdict(verdict_counts):
            if verdict_counts.get("CONSISTENT_WITH_NOISE", 0) > 0:
                return "CONSISTENT_WITH_NOISE"
            elif verdict_counts.get("REGIME_DEPENDENT", 0) > 0:
                return "REGIME_DEPENDENT"
            elif verdict_counts.get("REGIME_STABLE_LOSS", 0) == len(realistic_res.ticker_order):
                return "REGIME_STABLE_LOSS"
            else:
                return "REGIME_STABLE"
        
        realistic_verdict = get_overall_verdict(realistic_res.verdict_counts)
        zero_verdict = get_overall_verdict(zero_res.verdict_counts)
        
        print(f"  Realistic verdict: {realistic_verdict}")
        print(f"  Zero cost verdict: {zero_verdict}")
        print(f"  Median reduction: {results[f'lookback_{lookback}']['realistic_costs']['cost_impact']['median_reduction']:.4f} "
              f"({results[f'lookback_{lookback}']['realistic_costs']['cost_impact']['median_reduction_pct']:.2f}%)")
        
        # Print segment results
        print(f"  Real costs segments: {len(realistic_res.ticker_order)} assets")
        print(f"  Zero cost segments: {len(zero_res.ticker_order)} assets")

    # Check determinism: rerun the check for lookback=5 and verify identical results
    print("\n=== Determinism check ===")
    lookback = 5
    signals_fn = lambda c, **kw: bt.momentum_signals(c, lookback=lookback)
    
    res1 = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=signals_fn,
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    res2 = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=signals_fn,
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    deterministic = True
    # Compare results
    if len(res1.ticker_order) != len(res2.ticker_order):
        deterministic = False
        print(f"  MISMATCH: different number of tickers: {len(res1.ticker_order)} vs {len(res2.ticker_order)}")
    
    for ticker in res1.ticker_order:
        if ticker not in res2.ticker_order:
            deterministic = False
            print(f"  MISMATCH: missing ticker in res2: {ticker}")
            continue
            
        if res1.ticker_order != res2.ticker_order:
            deterministic = False
            print(f"  MISMATCH: different ticker order")
            break
            
        s1 = res1.assets[ticker]
        s2 = res2.assets[ticker]
        
        if len(s1.scenarios) != len(s2.scenarios):
            deterministic = False
            print(f"  MISMATCH in asset {ticker}: different number of scenarios: {len(s1.scenarios)} vs {len(s2.scenarios)}")
            continue
            
        for i, (sc1, sc2) in enumerate(zip(s1.scenarios, s2.scenarios)):
            if sc1.name != sc2.name or \
               abs(sc1.baseline_median_log_return - sc2.baseline_median_log_return) > 1e-6 or \
               abs(sc1.noise_median_log_return - sc2.noise_median_log_return) > 1e-6:
                deterministic = False
                print(f"  MISMATCH in scenario {i} of asset {ticker}: {sc1.name}")
                print(f"    s1 baseline: {sc1.baseline_median_log_return}, s2 baseline: {sc2.baseline_median_log_return}")
                print(f"    s1 null: {sc1.noise_median_log_return}, s2 null: {sc2.noise_median_log_return}")
                print(f"    s1 edge_status: {sc1.edge_status}, s2 edge_status: {sc2.edge_status}")
                print(f"    s1 n_folds: {sc1.n_folds}, s2 n_folds: {sc2.n_folds}")
                print(f"    s1 verdict: {s1.overall_verdict}, s2 verdict: {s2.overall_verdict}")
    
    if deterministic:
        print("  Determinism: PASS - results are identical across runs")
    else:
        print("  Determinism: FAIL - results differ between runs")
        sys.exit(1)

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=LOOKBACKS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        costs={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        results=results,
        deterministic=deterministic,
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Artifact written to: {ARTIFACT_PATH} ===")
    print("NOTE: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())