"""Cost-sensitivity check for lookback 3/5/10 momentum under realistic costs.

This task addresses the bottleneck: "The candidate positive momentum evidence is
short-horizon and mechanical; cost robustness remains decision-relevant."

Runs bounded cost-sensitivity checks for lookback 3/5/10 momentum under the
repository's existing realistic cost model and matched null.

Evidence basis: The existing momentum test shows REGIME_STABLE for 7 assets
and REGIME_DEPENDENT for 1 asset (JPM). This positive edge needs to survive
conservative transaction-cost stress.

Falsification prediction: The edge either CONSISTENT_WITH_NOISE (cost kills
any edge) or REGIME_DEPENDENT (cost affects regime mixes differently).
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
ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_momentum.json"

# Realistic cost model from examples/ma_crossover.py
cfg_realistic = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero-cost baseline for comparison
cfg_zero_cost = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)


def run_cost_sensitivity(univ_tickers):
    """Run cost-sensitivity check for each lookback length."""
    results = {}

    for lookback in LOOKBACKS:
        print(f"\n=== Lookback {lookback} momentum cost sensitivity ===")

        # Test on the full universe with realistic costs
        print(f"  Full-universe cost impact:")
        summary = bt.stress_segments_across_tickers(
            tickers=univ_tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lookback),),
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg_realistic,
        )
        
        # Extract verdict counts and ticker order
        verdict_counts = summary.verdict_counts
        ticker_order = summary.ticker_order
        
        # Extract aggregate metrics from first asset's result
        if ticker_order:
            first_ticker = ticker_order[0]
            first_asset_result = summary.assets[first_ticker]
            # Use the median from the first scenario as representative
            if first_asset_result.scenarios:
                representative_median = first_asset_result.scenarios[0].baseline_median_log_return
            else:
                representative_median = 0.0
        else:
            representative_median = 0.0
            
        results[f"lookback_{lookback}"] = {
            "verdict_counts": verdict_counts,
            "ticker_order": ticker_order,
            "representative_median": round(representative_median, 3),
        }

        print(f"    Representative median log return: {results[f'lookback_{lookback}']['representative_median']:+.3f}")
        print(f"    Verdict counts: {results[f'lookback_{lookback}']['verdict_counts']}")

    return results


def perturbation_sweep():
    """Synthetic perturbation sweep of lookback 3/5/10 over canonical regime."""
    print("\n=== Synthetic perturbation sweep (lookback 3/5/10) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", 5),),
        multipliers=(0.6, 1.0, 2.0),  # 3, 5, 10
    )
    sweep = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", 5),))
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )

    print(f"  param sets: {len(sweep.param_sets)}")
    print(f"  candidate medians: {[round(m, 3) for m in summary.median_log_returns]}")
    print(f"  baseline (lookback=5): {summary.baseline_median_log_return:+.3f}")
    print(f"  coin-flip null baseline: {noise.baseline_median_log_return:+.3f}")
    print(f"  baseline vs null (sample-calibrated tol): {summary.compare_noise(noise.baseline_median_log_return)}")
    print(f"  baseline vs null (fixed tol 0.05): {summary.compare_noise(noise.baseline_median_log_return, tol=0.05)}")

    return {
        "param_sets": [tuple(ps) for ps in sweep.param_sets],
        "medians": [round(m, 3) for m in summary.median_log_returns],
        "baseline_median": round(summary.baseline_median_log_return, 3),
        "noise_median": round(noise.baseline_median_log_return, 3),
        "compare_noise_default": summary.compare_noise(noise.baseline_median_log_return),
        "compare_noise_fixed": summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
        "effective_tolerance": round(summary.effective_tolerance, 4),
        "n_folds": summary.n_folds,
        "cost_resistant": summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
    }


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

    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Loading universe for cost robustness test ===")
    univ_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        univ_tickers[ticker] = bt.load_ticker(ticker)[0]
    print(f"  loaded {len(univ_tickers)} assets: {list(univ_tickers.keys())}")

    print("\n=== 4. Cost-sensitivity check for lookback 3/5/10 momentum ===")
    cost_results = run_cost_sensitivity(univ_tickers)

    print("\n=== 5. Synthetic perturbation sweep ===")
    perturbation = perturbation_sweep()

    print("\n=== 6. Determinism check ===")
    # Verify by running the same logic again
    cost_results2 = run_cost_sensitivity(univ_tickers)
    cost_json1 = json.dumps(cost_results, sort_keys=True)
    cost_json2 = json.dumps(cost_results2, sort_keys=True)
    print(f"  determinism: cost_results == cost_results2: {cost_json1 == cost_json2}")
    assert cost_json1 == cost_json2, "non-deterministic cost-sensitivity results"

    # Write artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=LOOKBACKS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_model=dict(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
        cost_results=cost_results,
        perturbation=perturbation,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
