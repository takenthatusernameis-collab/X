"""Real-data cost-sensitivity check for momentum lookback 3/5/10.

Objective: test whether the momentum edge at lookback=5 survives conservative
transaction costs on the collected real-data universe. We test lookbacks 3, 5, 10
to assess parameter sensitivity.

Method (fixed a-priori, not tuned to OOS):
- Long the previous N-day return; hold 1 day; daily rebalance
- Same walk-forward: train=252d/test=84d/warmup=60d/overlap=60d
- Realistic costs: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005
- Regime-stability stress test across 4 contiguous volatility blocks
- Matched null benchmark: coin-flip signals on same segments
- Determinism asserted via independent recomputation

Research/simulation only. No live trading or production execution.
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
REALISTIC_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
)
ZERO_COST_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"


def run_cost_sensitivity():
    """Run cost-sensitivity check for momentum lookback 3/5/10."""
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

    print("\n=== 3. Cost-sensitivity sweep: momentum lookback 3/5/10 ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, realistic costs vs matched null "
          "(coin-flip on same segments)".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load full universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(tickers.keys())

    # Run per lookback
    cost_sensitivity = {}
    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} ---")
        
        # Realistic cost run
        print(f"  Realistic costs (commission_per_trade=2.0, commission_per_share=0.003, "
              f"slippage_cents=2.0, slippage_proportional=0.0005)...")
        
        # Collect per-asset results
        cost_sensitivity[lb] = {}
        for ticker in ticker_order:
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            signals = bt.momentum_signals(closes, lookback=lb)
            
            # Realistic cost results
            cfg_cost = bt.BacktestConfig(
                initial_capital=1e6,
                commission_per_trade=2.0,
                commission_per_share=0.003,
                slippage_cents=2.0,
                slippage_proportional=0.0005,
                warmup_periods=WARM,
            )
            
            res_cost = bt.stress_segments(
                signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=cfg_cost,
            )
            
            # Zero-cost baseline for comparison
            res_zero = bt.stress_segments(
                signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=ZERO_COST_CFG,
            )
            
# Store results
            cost_sensitivity[lb][ticker] = {
                "realistic": {
                    "segments": [s.name for s in res_cost.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res_cost.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res_cost.scenarios],
                    "candidate_dispersion": round(res_cost.candidate_dispersion, 3),
                    "null_dispersion": round(res_cost.null_dispersion, 3),
                    "n_folds": res_cost.n_folds,
                    "verdict": res_cost.overall_verdict,
                },
                "zero_cost": {
                    "segments": [s.name for s in res_zero.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res_zero.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res_zero.scenarios],
                    "candidate_dispersion": round(res_zero.candidate_dispersion, 3),
                    "null_dispersion": round(res_zero.null_dispersion, 3),
                    "n_folds": res_zero.n_folds,
                    "verdict": res_zero.overall_verdict,
                },
                "cost_impact": {
                    "median_reduction": round(
                        np.mean([r.realistic["medians"][i] - r.zero_cost["medians"][i]
                                for i in range(len(r.realistic["medians"]))
                                for r in [cost_sensitivity[lb][ticker]]]), 3
                    ),
                    "edge_preservation": "EDGE_PRESERVED" if any(
                        m > 0.05 for m in cost_sensitivity[lb][ticker]["realistic"]["medians"]
                    ) else "EDGE_LOST",
                }
            }
            }
            
            print(f"    {ticker}: realistic={cost_sensitivity[lb][ticker]['cost_impact']['edge_preservation']}, "
                  f"median reduction={cost_sensitivity[lb][ticker]['cost_impact']['median_reduction']:.3f}")
        
        # Summary for this lookback
        robust_at_realistic = sum(
            1 for ticker in ticker_order
            if cost_sensitivity[lb][ticker]["cost_impact"]["edge_preservation"] == "EDGE_PRESERVED"
        )
        print(f"    Robust at realistic costs: {robust_at_realistic}/{len(ticker_order)} assets")

    # Determinism check for lookback=5
    print("\\n=== 4. Determinism check (lookback=5) ===")
    target = ["AMZN", "JPM"]
    res1 = {}
    res2 = {}
    
    for lb in LOOKBACKS:
        res1[lb] = {}
        res2[lb] = {}
        for ticker in target:
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            signals = bt.momentum_signals(closes, lookback=lb)
            
            res1[lb][ticker] = bt.stress_segments(
                signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=REALISTIC_CFG,
            )
            
            res2[lb][ticker] = bt.stress_segments(
                signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=REALISTIC_CFG,
            )
    
    # Verify determinism
    deterministic = True
    for lb in LOOKBACKS:
        for ticker in target:
            # Compare key metrics between res1 and res2
            for scenario_idx in range(len(res1[lb][ticker].scenarios)):
                s1 = res1[lb][ticker].scenarios[scenario_idx]
                s2 = res2[lb][ticker].scenarios[scenario_idx]
                if abs(s1.baseline_median_log_return - s2.baseline_median_log_return) > 1e-6:
                    deterministic = False
                    print(f"  DETERMINISM FAILURE: lookback={lb}, ticker={ticker}, "
                          f"segment={s1.name}, median mismatch: {s1.baseline_median_log_return} vs {s2.baseline_median_log_return}")
    
    if deterministic:
        print("  DETERMINISM VERIFIED: lookback=3/5/10 produces identical results across independent runs")
    else:
        print("  DETERMINISM FAILED: non-deterministic output detected")
        sys.exit(1)

    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "type": "momentum_cost_sensitivity",
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_model": {
            "commission_per_trade": 2.0,
            "commission_per_share": 0.003,
            "slippage_cents": 2.0,
            "slippage_proportional": 0.0005,
        },
        "cost_sensitivity": cost_sensitivity,
        "universe_robustness": {
            "ticker_order": ticker_order,
            "lookback_5_edge_counts": {
                lb: sum(
                    1 for ticker in ticker_order
                    if any(m > 0.05 for m in cost_sensitivity[lb][ticker]["realistic"]["medians"])
                )
                for lb in LOOKBACKS
            },
            "lookback_5_robust_at_realistic": {
                lb: sum(
                    1 for ticker in ticker_order
                    if cost_sensitivity[lb][ticker]["cost_impact"]["edge_preservation"] == "EDGE_PRESERVED"
                )
                for lb in LOOKBACKS
            },
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print("\n=== 5. Artifact written ===")
    print(f"  artifact: {ARTIFACT_PATH}")
    print("\n=== Summary ===")
    for lb in LOOKBACKS:
        robust = cost_sensitivity[lb]["lookback_5_robust_at_realistic"]
        print(f"  Lookback {lb}: {robust}/{len(ticker_order)} assets preserve edge at realistic costs")
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-sensitivity check on the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(run_cost_sensitivity())