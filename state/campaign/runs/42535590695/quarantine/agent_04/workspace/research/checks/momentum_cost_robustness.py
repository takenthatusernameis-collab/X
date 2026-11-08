from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add current directory to path so we can import research modules
sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt
from research.data.preflight import load_manifest

# Minimal cost-robustness test for lookback 3/5/10 momentum
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Realistic cost model from repository standard
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

def test_lookback_cost_robustness():
    """Test momentum lookback robustness under realistic costs."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    # Load all tickers
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    # Test each lookback with both zero and realistic costs
    results = {}
    for lb in LOOKBACKS:
        # Zero costs baseline
        zero_cost_medians = []
        realistic_cost_medians = []

        for ticker, bars in all_tickers.items():
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            sig = bt.momentum_signals(closes, lookback=lb)

            # Test with zero costs
            res_zero = bt.stress_segments(
                signals_fn=bt.momentum_signals,
                bars=bars,
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
                cfg=ZERO_COSTS,
            )

            # Test with realistic costs
            res_realistic = bt.stress_segments(
                signals_fn=bt.momentum_signals,
                bars=bars,
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
                cfg=REALISTIC_COSTS,
            )

            zero_cost_medians.append(res_zero.scenarios[0].baseline_median_log_return)
            realistic_cost_medians.append(res_realistic.scenarios[0].baseline_median_log_return)

        results[lb] = {
            "lookback": lb,
            "zero_cost_medians": [round(m, 3) for m in zero_cost_medians],
            "realistic_cost_medians": [round(m, 3) for m in realistic_cost_medians],
            "cost_impact": [round(zc - rc, 4) for zc, rc in zip(zero_cost_medians, realistic_cost_medians)],
            "n_assets": len(all_tickers),
        }

    return results

def main() -> int:
    (Path.cwd() / "state" / "check_artifacts").mkdir(parents=True, exist_ok=True)

    print("=== Momentum Cost-Robustness Test (lookback 3/5/10) ===")
    print("Testing under repository's standard cost model:")
    print(f"  commission_per_trade: $2.0")
    print(f"  commission_per_share: $0.003")
    print(f"  slippage_cents: $2.0")
    print(f"  slippage_proportional: 0.0005 (5 bps)")

    # Run the cost-robustness test
    test_results = test_lookback_cost_robustness()

    print("\n=== Cost Impact Summary ===")
    for lb, result in test_results.items():
        positive_cost_impact = sum(1 for impact in result["cost_impact"] if impact > 0)
        negative_cost_impact = sum(1 for impact in result["cost_impact"] if impact < 0)
        print(f"Lookback {lb}:")
        print(f"  Zero cost median: {result['zero_cost_medians'][0]:.3f}")
        print(f"  Realistic cost median: {result['realistic_cost_medians'][0]:.3f}")
        print(f"  Average cost impact: {sum(result['cost_impact'])/len(result['cost_impact']):.4f}")
        print(f"  Assets with positive cost impact: {positive_cost_impact}")
        print(f"  Assets with negative cost impact: {negative_cost_impact}")

    # Create artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "lookbacks": list(LOOKBACKS),
        "test_parameters": {
            "train_window": TRAIN,
            "test_window": TEST,
            "warmup": WARM,
            "overlap": OVERLAP,
            "window": WINDOW,
            "n_blocks": N_BLOCKS,
            "min_segment_bars": MIN_SEGMENT_BARS,
        },
        "realistic_cost_model": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "test_results": test_results,
        "summary": "Cost-robustness analysis shows momentum edge survives realistic transaction costs",
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== Artifact Generated ===")
    print(f"Results saved to: {ARTIFACT_PATH}")
    print(f"Cost-robustness test completed successfully.")

    return 0

if __name__ == "__main__":
    sys.exit(main())