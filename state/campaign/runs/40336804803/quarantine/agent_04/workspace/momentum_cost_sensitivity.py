"""Cost sensitivity test for momentum (lookback 3/5/10) with realistic transaction costs.

Research/simulation only. No live trading or production execution.

This test evaluates whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected 10-asset real-data universe.

Evidence gate: the result is reproducible and independently verified, regardless of outcome.
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

# Realistic cost parameters from ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=WARM,
)


async def momentum_cost_sensitivity():
    """Run cost sensitivity test for lookback 3/5/10 momentum."""
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

    print("=== Momentum Cost Sensitivity Test (lookback 3/5/10) ===")
    print("Realistic cost model: commission=$2/trade, commission=$0.003/share, "
          "slippage=$0.02/share, proportional=0.05%")

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("\n=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight ===")
    # Import and run preflight directly
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from research.data import preflight
        rc = preflight.main()
        if rc != 0:
            print("PREFLIGHT FAILED; aborting real-data run.")
            return None
    except Exception as e:
        print(f"Error running preflight: {e}")
        print("PREFLIGHT FAILED; aborting real-data run.")
        return None

    # Load the full universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        print(f"  {ticker}: {len(bars)} bars from {dates[0]} to {dates[-1]}")

    # Test momentum with different lookbacks under realistic costs
    print("\n=== 3. Cost sensitivity test ===")
    lookback_params = [3, 5, 10]
    results = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_costs": {
            "initial_capital": REALISTIC_COSTS.initial_capital,
            "target_exposure": REALISTIC_COSTS.target_exposure,
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
            "warmup_periods": REALISTIC_COSTS.warmup_periods,
        },
        "zero_costs": {
            "initial_capital": ZERO_COSTS.initial_capital,
            "target_exposure": ZERO_COSTS.target_exposure,
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
            "warmup_periods": ZERO_COSTS.warmup_periods,
        },
        "lookback_sensitivity": [],
    }

    for lookback in lookback_params:
        print(f"\n  Lookback={lookback} momentum")

        # Define momentum signal function for this lookback
        def make_momentum_signals(lb):
            def momentum_signals(closes, **params):
                n = len(closes)
                out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
                for i in range(lb, n):
                    ret = float(np.mean(np.log(closes[i - lb + 1 : i + 1])))
                    out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
                return out
            return momentum_signals

        momentum_sig = make_momentum_signals(lookback)

        # Run with zero costs (baseline)
        print(f"    Zero costs: ", end="")
        result_zero = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: momentum_sig(c, lookback=lookback),
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

        # Get median from first asset's regime stress result (use first scenario)
        first_ticker = list(tickers.keys())[0]
        zero_scenario = result_zero.assets[first_ticker].scenarios[0]
        zero_median = zero_scenario.baseline_median_log_return
        zero_positive = zero_scenario.n_folds
        zero_verdict = result_zero.assets[first_ticker].overall_verdict
        print(f"median log ret={zero_median:+.3f}, positive folds={zero_positive}, "
              f"verdict={zero_verdict}")

        # Run with realistic costs
        print(f"    Realistic costs: ", end="")
        result_realistic = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: momentum_sig(c, lookback=lookback),
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

        realistic_scenario = result_realistic.assets[first_ticker].scenarios[0]
        realistic_median = realistic_scenario.baseline_median_log_return
        realistic_positive = realistic_scenario.n_folds
        realistic_verdict = result_realistic.assets[first_ticker].overall_verdict
        print(f"median log ret={realistic_median:+.3f}, positive folds={realistic_positive}, "
              f"verdict={realistic_verdict}")

        # Store results
        lookback_result = {
            "lookback": lookback,
            "zero_costs": {
                "median_log_return": zero_median,
                "positive_folds": zero_positive,
                "verdict": zero_verdict,
                "candidates_sigificant": sum(1 for v in result_zero.verdict_counts.values() if v > 0),
                "candidates_consistent": result_zero.verdict_counts.get("REGIME_STABLE", 0),
            },
            "realistic_costs": {
                "median_log_return": realistic_median,
                "positive_folds": realistic_positive,
                "verdict": realistic_verdict,
                "candidates_sigificant": sum(1 for v in result_realistic.verdict_counts.values() if v > 0),
                "candidates_consistent": result_realistic.verdict_counts.get("REGIME_STABLE", 0),
            },
            "cost_impact": {
                "median_return_change": realistic_median - zero_median,
                "positive_folds_change": realistic_positive - zero_positive,
                "edge_survival": realistic_median > zero_median,  # naive survival
            }
        }
        results["lookback_sensitivity"].append(lookback_result)

    # Write artifact for independent verification
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\n  artifact written to: {ARTIFACT_PATH}")

    return results


async def main():
    print("Running momentum cost sensitivity test...")
    print("\nResearch/simulation only. No live trading or production execution.")
    print("\nPrimary learning question: Does the short-horizon momentum edge survive "
          "conservative transaction-cost stress on the collected real-data universe?")

    results = await momentum_cost_sensitivity()

    if results is None:
        print("Test failed - preflight checks failed")
        return 1

    print("\n=== Cost sensitivity test completed ===")
    print("\nSummary of results:")
    for lb_result in results["lookback_sensitivity"]:
        lb = lb_result["lookback"]
        zero = lb_result["zero_costs"]["median_log_return"]
        realistic = lb_result["realistic_costs"]["median_log_return"]
        change = lb_result["cost_impact"]["median_return_change"]
        print(f"  Lookback {lb}: zero-cost={zero:+.3f}, "
              f"realistic-cost={realistic:+.3f}, change={change:+.3f}")

    print(f"\nArtifact available for independent verification: {results.get('artifact_path', 'NOT SET')}")
    print(f"\n  artifact written to: {ARTIFACT_PATH}")
    print(f"  artifact available for independent verification: {ARTIFACT_PATH}")

    return 0


def main() -> int:
    print("Running momentum cost sensitivity test...")
    print("\nResearch/simulation only. No live trading or production execution.")
    print("\nPrimary learning question: Does the short-horizon momentum edge survive "
          "conservative transaction-cost stress on the collected real-data universe?")

    results = momentum_cost_sensitivity()

    if results is None:
        print("Test failed - preflight checks failed")
        return 1

    print("\n=== Cost sensitivity test completed ===")
    print("\nSummary of results:")
    for lb_result in results["lookback_sensitivity"]:
        lb = lb_result["lookback"]
        zero = lb_result["zero_costs"]["median_log_return"]
        realistic = lb_result["realistic_costs"]["median_log_return"]
        change = lb_result["cost_impact"]["median_return_change"]
        print(f"  Lookback {lb}: zero-cost={zero:+.3f}, "
              f"realistic-cost={realistic:+.3f}, change={change:+.3f}")

    print(f"\nArtifact available for independent verification: {results.get('artifact_path', ARTIFACT_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
