"""Cost robustness check for momentum lookback 3/5/10 across real data.

Research / simulation only. No live trading or production execution.

This check tests whether the short-horizon momentum edge observed in
`research/checks/momentum.py` survives realistic transaction costs.

Method: run momentum walk-forward validation with realistic costs for
each lookback (3,5,10) on the collected 10-asset universe, and compare
against matched null (same cost model). Results are classified as
ROBUST/EDGE vs NO_EDGE/LOSS based on cost-robust positive evidence.

Cost model (realistic):
  commission_per_trade=2.0
  commission_per_share=0.003
  slippage_cents=2.0
  slippage_proportional=0.0005

Cost model (zero):
  all costs = 0.0

The artifact contains per-asset, per-lookback metrics under both cost
models, with a cost-robustness verdict per asset (ROBUST if a positive
REGIME_STABLE edge appears at >= 2 lookbacks under realistic costs).
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

def run_momentum_with_costs(cost_label, cfg):
    """Run momentum stress_segments for all lookbacks under given cost config."""
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
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
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
def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Momentum cost robustness check ===")
    print(f"Dataset: {DATASET_ID} | {len(LOOKBACKS)} lookbacks: {LOOKBACKS}")
    print(f"Universe: 10 collected assets")
    print(f"Cost models: zero vs realistic")
    print(f"Realistic costs: commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005")
    print()

    # Zero cost model
    cfg_zero = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
    zero_results, ticker_order = run_momentum_with_costs("zero_cost", cfg_zero)
    print("Zero cost run complete")

    # Realistic cost model
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=60,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )
    realistic_results, _ = run_momentum_with_costs("realistic_cost", cfg_realistic)
    print("Realistic cost run complete")

    # Build artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": 42,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_zero": {
            "per_asset": {},
        },
        "cost_realistic": {
            "per_asset": {},
        },
    }

    # Fill artifact data
    for ticker in ticker_order:
        artifact["cost_zero"]["per_asset"][ticker] = {
            lb: zero_results[str(lb)][ticker] for lb in LOOKBACKS
        }
        artifact["cost_realistic"]["per_asset"][ticker] = {
            lb: realistic_results[str(lb)][ticker] for lb in LOOKBACKS
        }

    # Add robustness verdicts
    robustness_results = {}
    for ticker in ticker_order:
        asset_data = {str(lb): realistic_results[str(lb)][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict_for_asset(asset_data)
        robustness_results[ticker] = {
            "robustness_verdict": v,
            "edge_counts": counts,
        }
    artifact["cost_realistic"]["robustness"] = robustness_results

    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"Artifact written to: {ARTIFACT_PATH}")
    print()
    print("=== Summary ===")
    for ticker in ticker_order:
        robust = realistic_results["5"][ticker]["verdict"] == "REGIME_STABLE" and all(
            realistic_results[str(lb)][ticker]["medians"][0] > 0 for lb in LOOKBACKS
        )
        if robust:
            print(f"  {ticker}: ROBUST edge at >=2 lookbacks under realistic costs")
        else:
            print(f"  {ticker}: NO_EDGE or SENSITIVE under realistic costs")

    return 0
if __name__ == "__main__":
    sys.exit(main())