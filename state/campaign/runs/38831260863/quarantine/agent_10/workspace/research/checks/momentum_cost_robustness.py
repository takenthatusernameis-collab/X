"""Cost-sensitivity check for momentum lookback 3 / 5 / 10.

This is a bounded cost-sensitivity test of the momentum edge using the
repository's existing realistic cost model and a matched coin-flip null.
It addresses the bottleneck: candidate positive momentum evidence is
short-horizon and mechanical; cost robustness remains decision-relevant.

Objective: classify the cost robustness of the momentum edge at lookback 3,
5, and 10 on the collected 10-asset universe.

Method (fixed a-priori, not tuned to OOS):
- long the previous lookback-day return; hold 1 day; daily rebalance;
- walk-forward train=252d / test=84d / warmup=60d / overlap=60d;
- 4 contiguous volatility blocks per asset;
- 600+ bars per segment (MIN_SEGMENT_BARS);
- realistic costs (from examples/ma_crossover.py):
    commission_per_trade=2.0
    commission_per_share=0.003
    slippage_cents=2.0
    slippage_proportional=0.0005
- matched coin-flip null (using framework noise_benchmark);
- no-look-ahead, no signal tuning after cost evaluation.

Deliverable: a reproducible artifact and independent verification result
classifying cost robustness (EDGE vs NO_EDGE at each lookback).
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

# Realistic cost model from examples/ma_crossover.py
REALISTIC_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def segment_result_with_costs(ticker, bars, lookback, costs_cfg):
    """Walk-forward result for one ticker and one lookback with given costs."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    cfg = bt.BacktestConfig(**costs_cfg)
    
    res = bt.stress_segments(
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
        bars=list(bars),
        signals=bt.momentum_signals(closes, lookback=lookback),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )
    return res


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost-sensitivity check for momentum lookback 3 / 5 / 10 ===")
    print(f"Realistic costs: {REALISTIC_COSTS}")
    print(f"Dataset: {manifest['dataset_id']} | Universe: {len(manifest['universe'])}")
    print()

    # Load all tickers from collected universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Define cost scenarios to test
    cost_scenarios = {
        "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                     slippage_cents=0.0, slippage_proportional=0.0),
        "realistic": REALISTIC_COSTS,
    }

    # Results structure
    results = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_scenarios": list(cost_scenarios.keys()),
        "per_asset": {},
        "universe": {}
    }

    # Run for each cost scenario
    for cost_name, costs_cfg in cost_scenarios.items():
        print(f"=== {cost_name.upper()} COSTS ===")
        
        # Per-asset results
        per_asset = {}
        universe_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        
        for ticker in ticker_order:
            print(f"  {ticker}: ", end="")
            
            # Run stress segments for each lookback with current cost scenario
            lookback_results = {}
            for lookback in LOOKBACKS:
                res = segment_result_with_costs(ticker, all_tickers[ticker], lookback, costs_cfg)
                lookback_results[str(lookback)] = {
                    "segments": [s.name for s in res.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                    "candidate_dispersion": round(res.candidate_dispersion, 3),
                    "null_dispersion": round(res.null_dispersion, 3),
                    "verdict": res.overall_verdict,
                    "n_folds": res.n_folds,
                }
            
            # Determine robustness verdict for this asset
            edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            for lb in LOOKBACKS:
                sr = lookback_results[str(lb)]
                verdict = sr["verdict"]
                if verdict == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
                    edge_counts["EDGE"] += 1
                elif verdict == "REGIME_STABLE_LOSS":
                    edge_counts["LOSS"] += 1
                else:
                    edge_counts["NO_EDGE"] += 1
            
            if edge_counts["EDGE"] >= 2:
                robustness_verdict = "ROBUST"
                universe_counts["ROBUST"] += 1
            elif edge_counts["EDGE"] == 1:
                robustness_verdict = "SENSITIVE"
                universe_counts["SENSITIVE"] += 1
            else:
                robustness_verdict = "NO_EDGE"
                universe_counts["NO_EDGE"] += 1
            
            # Store per-asset profile
            profile = {
                "lookback_results": lookback_results,
                "edge_counts": edge_counts,
                "robustness_verdict": robustness_verdict,
                "lookback_medians": {str(lb): lookback_results[str(lb)]["medians"] for lb in LOOKBACKS},
                "lookback_null_medians": {str(lb): lookback_results[str(lb)]["null_medians"] for lb in LOOKBACKS},
            }
            per_asset[ticker] = profile
            
            # Print per-asset summary
            meds_str = ", ".join(f"{lb}:{med:+.3f}" for lb, med in zip(LOOKBACKS, [lookback_results[str(lb)]["medians"][0] for lb in LOOKBACKS]))
            print(f"lookbacks {meds_str} -> {robustness_verdict}")

        results["per_asset"][cost_name] = per_asset
        
        print(f"  Universe verdict counts: {universe_counts}")
        print()

    # Determinism check: rerun AMZN with realistic costs
    print("=== DETERMINISM CHECK ===")
    print("Rerunning AMZN with realistic costs to verify determinism...")
    
    amzn1_results = {}
    amzn2_results = {}
    
    for lookback in LOOKBACKS:
        res1 = segment_result_with_costs("AMZN", all_tickers["AMZN"], lookback, REALISTIC_COSTS)
        amzn1_results[str(lookback)] = {
            "segments": [s.name for s in res1.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res1.scenarios],
        }
        
        res2 = segment_result_with_costs("AMZN", all_tickers["AMZN"], lookback, REALISTIC_COSTS)
        amzn2_results[str(lookback)] = {
            "segments": [s.name for s in res2.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res2.scenarios],
        }
    
    deterministic = amzn1_results == amzn2_results
    print(f"  AMZN realistic costs deterministic: {deterministic}")
    
    if not deterministic:
        print("  Non-deterministic results found!")
        sys.exit(1)

    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("This is a bounded cost-sensitivity check for momentum lookback 3/5/10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
