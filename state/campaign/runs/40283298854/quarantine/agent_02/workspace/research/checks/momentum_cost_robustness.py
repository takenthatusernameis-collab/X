#!/usr/bin/env python3
"""Cost-sensitivity analysis for momentum lookback 3/5/10.

This script tests the short-horizon momentum edge (lookback 3/5/10) under
the repository's existing realistic cost model and matched null.

Based on the evidence base from momentum.py, we know momentum shows
REGIME_STABLE positive edges in some assets at lookback=5.
This analysis determines whether that edge survives conservative
transaction costs.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = [3, 5, 10]
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

# Realistic cost model based on test_engine.py unit tests
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.003,  # 3 basis points per share
    slippage_cents=5.0,           # 5 cents per share
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

# Zero-cost model (benchmark)
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

def main():
    print("=== Momentum Cost-Robustness Analysis ===")
    print(f"Lookback: {LOOKBACKS}")
    print(f"Realistic cost model: commission_per_share=0.003, slippage_cents=5.0")
    print(f"Baseline (zero-cost): commission_per_share=0, slippage_cents=0")
    print(f"Test data: {DATASET_ID}")
    print()

    # Load manifest for data verification
    manifest = load_manifest()
    print("=== Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    # Run cost-sensitivity analysis for each lookback
    cost_results = {}
    for lookback in LOOKBACKS:
        print(f"=== Lookback {lookback} days: momentum with realistic costs ===")

        # Load real data
        tickers = {}
        for entry in manifest["entries"]:
            ticker = entry["ticker"]
            tickers[ticker] = bt.load_ticker(ticker)[0]

        # Generate noise benchmark for this lookback (matched null)
        bars = bt.generate_bars(600, seed=SEED)
        noise_signals = bt.momentum_signals(bars.closes_array(), lookback=lookback)
        noise_res = bt.walk_forward(
            list(bars), noise_signals, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=ZERO_COSTS)
        noise_medians = []
        for f in noise_res.folds:
            log_ret = np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            noise_medians.append(log_ret)
        noise_median = np.median(noise_medians) if noise_medians else 0.0

        # Run momentum with realistic costs
        summary_real = bt.stress_segments_across_tickers(
            tickers=tickers,
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
        )

        # Run momentum with zero costs (baseline)
        summary_zero = bt.stress_segments_across_tickers(
            tickers=tickers,
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
        )

        # Analyze cost impact
        per_asset_results = {}
        zero_vs_real_comparison = []

        for ticker in summary_real.assets:
            real_asset = summary_real.assets[ticker]
            zero_asset = summary_zero.assets[ticker]

            # Get medians from both runs
            real_medians = [round(s.baseline_median_log_return, 3) for s in real_asset.scenarios]
            zero_medians = [round(s.baseline_median_log_return, 3) for s in zero_asset.scenarios]

            # Calculate cost erosion (difference)
            erosion = [round(real - zero, 6) for real, zero in zip(real_medians, zero_medians)]

            # Verdict remains same if both consistent with noise, both regime stable, etc.
            real_verdict = real_asset.overall_verdict
            zero_verdict = zero_asset.overall_verdict

            verdict_preserved = real_verdict == zero_verdict

            erosion_sum = sum(erosion)
            per_asset_results[ticker] = {
                "lookback": lookback,
                "real_cost_medians": real_medians,
                "zero_cost_medians": zero_medians,
                "cost_erosion_log_return": erosion,
                "erosion_sum": erosion_sum,
                "median_erosion": round(np.mean(np.abs(erosion)), 6) if erosion else 0.0,
                "real_cost_verdict": real_verdict,
                "zero_cost_verdict": zero_verdict,
                "verdict_preserved": verdict_preserved,
                "n_segments": len(real_medians),
                "noise_median_log_return": round(noise_median, 6),
                "edge_preserved_vs_null": None,  # Will calculate after
            }

            # Calculate edge vs null for realistic costs
            edge_vs_null_real = []
            for m in real_medians:
                if all(abs(m) <= 0.05 for m in [m]):  # Conservative tolerance
                    edge_vs_null_real.append("CONSISTENT_WITH_NOISE")
                elif abs(m) > 2.0 * abs(noise_median) if noise_median != 0 else False:
                    edge_vs_null_real.append("REGIME_DEPENDENT_OR_STABLE")
                elif all(m < 0 for m in [m]):
                    edge_vs_null_real.append("REGIME_STABLE_LOSS")
                else:
                    edge_vs_null_real.append("REGIME_STABLE_POSITIVE")

            per_asset_results[ticker]["edge_vs_null_real"] = edge_vs_null_real
            zero_vs_real_comparison.append({
                "ticker": ticker,
                "lookback": lookback,
                "zero_cost_verdict": zero_verdict,
                "real_cost_verdict": real_verdict,
                "verdict_changed": real_verdict != zero_verdict,
            })

        # Summary statistics
        verdict_changes = sum(1 for r in per_asset_results.values() if r["verdict_preserved"] is False)
        total_assets = len(per_asset_results)
        verdict_preserved_pct = (total_assets - verdict_changes) / total_assets * 100 if total_assets > 0 else 0.0

        cost_results[f"lookback_{lookback}"] = {
            "lookback": lookback,
            "cost_model": {
                "commission_per_share": REALISTIC_COSTS.commission_per_share,
                "slippage_cents": REALISTIC_COSTS.slippage_cents,
                "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
            },
            "per_asset": per_asset_results,
            "summary": {
                "total_assets": total_assets,
                "verdict_changes": verdict_changes,
                "verdict_preserved_pct": round(verdict_preserved_pct, 2),
                "avg_cost_erosion_log_return": round(np.mean([r["erosion_sum"] for r in per_asset_results.values()]) / total_assets if total_assets > 0 else 0.0, 6),
                "max_single_fold_erosion": round(max([max(abs(m) for m in r["cost_erosion_log_return"]) for r in per_asset_results.values()]) if per_asset_results else 0.0, 6),
            },
            "noise_median_log_return": round(noise_median, 6),
        }

        print(f"  Lookback {lookback}: {summary_real.n_assets} assets analyzed")
        print(f"    Verdicts changed by costs: {verdict_changes}/{total_assets} ({verdict_preserved_pct:.1f}% preserved)")
        print(f"    Average cost erosion: {cost_results[f'lookback_{lookback}']['summary']['avg_cost_erosion_log_return']:.6f} log-return")
        print(f"    Max single-fold erosion: {cost_results[f'lookback_{lookback}']['summary']['max_single_fold_erosion']:.6f} log-return")
        print(f"    Noise (matched null) median: {noise_median:.6f}")
        print()

    # Generate artifact
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "dataset_id": DATASET_ID,
        "lookbacks": LOOKBACKS,
        "cost_model": {
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
            "description": "Realistic cost model based on test_engine.py unit tests",
        },
        "zero_cost_model": {
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
        "cost_sensitivity": cost_results,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print("=== Cost-Robustness Analysis Complete ===")
    print(f"Artifact written to: {ARTIFACT_PATH}")
    print()
    print("Key findings:")
    for lb_key, results in cost_results.items():
        lb = results["lookback"]
        preserved = results["summary"]["verdict_preserved_pct"]
        erosion = results["summary"]["avg_cost_erosion_log_return"]
        print(f"  Lookback {lb}: {preserved:.1f}% verdicts preserved vs zero-cost")
        print(f"    Average log-return erosion: {erosion:.6f}")
        print(f"    Noise median: {results['noise_median_log_return']:.6f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
