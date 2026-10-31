"""Momentum cost robustness: walk-forward on the collected universe with realistic transaction costs.

This check complements the momentum_sweep (research/checks/momentum_sweep.py) by testing
cost robustness of the positive edge identified in the canonical momentum frontier test.
The momentum_sweep shows that 7/10 assets have REGIME_STABLE positive edges at lookback=5.
This check tests whether those edges survive realistic transaction costs (commission + slippage).

Method (fixed a-priori, not tuned to OOS): long the previous 5-day return; hold 1 day; daily rebalance.
Base lookback 5 is the reference. N_blocks = 4 contiguous volatility blocks.
Train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment.

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

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

    # Zero-cost configuration (baseline)
    cfg0 = bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=0.0, commission_per_share=0.0,
        slippage_cents=0.0, slippage_proportional=0.0,
    )

    # Realistic costs configuration
    cfg1 = bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )

    print("\n=== 3. Momentum across collected universe (lookback=5) ===")
    print("Momentum = long the previous 5-day return, hold 1 day, daily rebalanced. "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, compared vs coin-flip null on the same "
          "segments.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load all tickers once
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    # Run momentum with both cost configurations
    results_zero = {}
    results_realistic = {}

    for ticker, bars in tickers.items():
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)

        # Zero cost results
        res0 = bt.stress_segments(
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=5),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": 5}],
            baseline=tuple([("lookback", 5)]),
            segment_fn=bt.segment_fn_from_labels(bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)),
            train_window=TRAIN, test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
            periods_per_year=252, min_segment_bars=MIN_SEGMENT_BARS,
        )
        results_zero[ticker] = res0

        # Realistic cost results
        res1 = bt.stress_segments(
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=5),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": 5}],
            baseline=tuple([("lookback", 5)]),
            segment_fn=bt.segment_fn_from_labels(bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)),
            train_window=TRAIN, test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
            periods_per_year=252, min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg1,
        )
        results_realistic[ticker] = res1

    # Print comparison
    print("  {:10s} {:14s} {:14s} {:10s} {:12s} {:12s} {:12s}".format(
        "asset", "zero_cost", "realistic_cost", "diff", "edge_loss", "still_pos", "cost_sensitive"))
    print("  " + "-" * 85)

    cost_robust_assets = []
    cost_unrobust_assets = []

    for ticker in sorted(tickers.keys()):
        zero_verdict = results_zero[ticker].overall_verdict
        realistic_verdict = results_realistic[ticker].overall_verdict

        zero_median = sum(s.baseline_median_log_return for s in results_zero[ticker].scenarios) / len(results_zero[ticker].scenarios)
        realistic_median = sum(s.baseline_median_log_return for s in results_realistic[ticker].scenarios) / len(results_realistic[ticker].scenarios)

        verdict_match = "YES" if zero_verdict == realistic_verdict else "NO"
        edge_loss = "YES" if realistic_median <= 0.0 else "NO"
        still_pos = "YES" if realistic_median > 0.0 else "NO"
        cost_sensitive = "YES" if edge_loss and verdict_match else "NO"

        if cost_sensitive == "YES":
            cost_robust_assets.append(ticker)
        else:
            cost_unrobust_assets.append(ticker)

        print("  {:10s} {:14s} {:14s} {:10s} {:12s} {:12s} {:12s}".format(
            ticker, zero_verdict, realistic_verdict, verdict_match,
            edge_loss, still_pos, cost_sensitive))

    print()

    # Synthetic perturbation sweep for parameter robustness
    print("\n=== 4. Synthetic perturbation sweep (lookback 3 / 5 / 10) ===")
    bars = bt.generate_bars(600, seed=42)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]

    # Zero cost perturbation
    sweep0 = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60, test_window=20, warmup=10, overlap_window=10,
        periods_per_year=252, cfg=cfg0,
    )
    summary0 = bt.sweep_summary(sweep0, baseline=tuple([("lookback", 5)]))
    noise0 = bt.noise_benchmark(
        bars=bars, param_grid=grid, train_window=60, test_window=20,
        warmup=0, overlap_window=0, periods_per_year=252, cfg=cfg0, seed=42,
    )

    # Realistic cost perturbation
    sweep1 = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60, test_window=20, warmup=10, overlap_window=10,
        periods_per_year=252, cfg=cfg1,
    )
    summary1 = bt.sweep_summary(sweep1, baseline=tuple([("lookback", 5)]))
    noise1 = bt.noise_benchmark(
        bars=bars, param_grid=grid, train_window=60, test_window=20,
        warmup=0, overlap_window=0, periods_per_year=252, cfg=cfg1, seed=42,
    )

    print("  zero cost:")
    print("    baseline (lookback=5): {:+.3f}".format(summary0.baseline_median_log_return))
    print("    coin-flip null: {:+.3f}".format(noise0.baseline_median_log_return))
    print("    baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(summary0.effective_tolerance, summary0.compare_noise(noise0.baseline_median_log_return)))

    print("  realistic cost:")
    print("    baseline (lookback=5): {:+.3f}".format(summary1.baseline_median_log_return))
    print("    coin-flip null: {:+.3f}".format(noise1.baseline_median_log_return))
    print("    baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(summary1.effective_tolerance, summary1.compare_noise(noise1.baseline_median_log_return)))

    print()

    # Universe-level summary
    zero_verdict_counts = {}
    realistic_verdict_counts = {}

    for ticker in sorted(tickers.keys()):
        zero_verdict_counts[results_zero[ticker].overall_verdict] = \
            zero_verdict_counts.get(results_zero[ticker].overall_verdict, 0) + 1
        realistic_verdict_counts[results_realistic[ticker].overall_verdict] = \
            realistic_verdict_counts.get(results_realistic[ticker].overall_verdict, 0) + 1

    print("=== 5. Summary ===")
    print("  zero cost verdict counts:", zero_verdict_counts)
    print("  realistic cost verdict counts:", realistic_verdict_counts)
    print("  cost-robust assets (edge survives realistic costs):", cost_robust_assets)
    print("  cost-unrobust assets (edge lost realistic costs):", cost_unrobust_assets)
    print("  cost-robust ratio: {:.1%}".format(
        len(cost_robust_assets) / len(tickers.keys())))

    # Determinism check
    print("\n=== 6. Determinism check ===")
    bars2 = bt.generate_bars(600, seed=42)
    sweep2 = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars2,
        param_grid=[{"lookback": 5}],
        train_window=60, test_window=20, warmup=10, overlap_window=10,
        periods_per_year=252, cfg=cfg0,
    )
    summary2 = bt.sweep_summary(sweep2, baseline=tuple([("lookback", 5)]))

    deterministic = abs(summary0.baseline_median_log_return - summary2.baseline_median_log_return) < 1e-6
    print("  deterministic: {}".format("YES" if deterministic else "NO"))
    if not deterministic:
        print("    warning: sweep not deterministic")

    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "lookback": 5,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "zero_cost": {
            "verdict_counts": zero_verdict_counts,
            "per_asset": {
                ticker: {
                    "segments": [s.name for s in res.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                    "candidate_dispersion": round(res.candidate_dispersion, 3),
                    "null_dispersion": round(res.null_dispersion, 3),
                    "verdict": res.overall_verdict,
                }
                for ticker, res in results_zero.items()
            },
            "baseline_median": round(summary0.baseline_median_log_return, 3),
            "noise_median": round(noise0.baseline_median_log_return, 3),
            "compare_noise": summary0.compare_noise(noise0.baseline_median_log_return),
        },
        "realistic_cost": {
            "verdict_counts": realistic_verdict_counts,
            "per_asset": {
                ticker: {
                    "segments": [s.name for s in res.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                    "candidate_dispersion": round(res.candidate_dispersion, 3),
                    "null_dispersion": round(res.null_dispersion, 3),
                    "verdict": res.overall_verdict,
                }
                for ticker, res in results_realistic.items()
            },
            "baseline_median": round(summary1.baseline_median_log_return, 3),
            "noise_median": round(noise1.baseline_median_log_return, 3),
            "compare_noise": summary1.compare_noise(noise1.baseline_median_log_return),
        },
        "cost_robust_assets": cost_robust_assets,
        "cost_unrobust_assets": cost_unrobust_assets,
        "cost_robust_ratio": len(cost_robust_assets) / len(tickers.keys()),
        "seed": 42,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost robustness on the collected universe: exploratory simulation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())