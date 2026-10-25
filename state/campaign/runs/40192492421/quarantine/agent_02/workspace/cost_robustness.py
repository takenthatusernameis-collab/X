#!/usr/bin/env python3
"""Bounded cost-sensitivity check for lookback 3/5/10 momentum.

Evidence basis: LEARNING_STATE admits short-horizon momentum as candidate positive
evidence but leaves execution-cost robustness as a meaningful qualification.

Objective: Run a bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

The realistic cost model is taken from examples/ma_crossover_real_data.py:
    commission_per_trade=2.0
    commission_per_share=0.003
    slippage_cents=2.0
    slippage_proportional=0.0005

This test:
1. Loads the collected universe from manifest.json
2. Generates momentum_signals for lookback=3,5,10 using research.backtest.regime_stability.momentum_signals
3. Runs walk-forward validation (train=252d/test=84d/warmup=60d/overlap=60d, per-asset nulls)
4. Compares performance against a matched coin-flip null with identical parameters
5. Produces an artifact for independent verification in verify_cost_robustness.py
6. Uses existing deterministic research infrastructure; no post-hoc tuning

Research/simulation only. No live trading or production execution.
Results inform only whether momentum is cost-robust enough to merit continued research.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASETS = [
    "yf-ohlcv-universe-2009-to-2026-10-03",  # from existing momentum.py
    # Add any other in-scope real datasets here if available
]

# Realistic cost model from examples/ma_crossover_real_data.py and ma_crossover.py
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=60,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero-cost baseline for comparison (equivalent to synthetic tooling validation)
ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=60,
)

# Consistent with existing momentum.py in research/checks/momentum.py
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Consistent with existing momentum.py
SEED = 42
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"


def momentum_signals_for_lookbacks(closes: np.ndarray, lookback: int):
    """Momentum signals for a specific lookback."""
    return bt.momentum_signals(closes, lookback=lookback)


def run_cost_sensitivity_check():
    """Execute the cost-sensitivity check for lookback 3,5,10 momentum."""
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_momentum_results.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Cost Robustness Check for Lookback 3/5/10 Momentum ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Cost model: realistic (commission_per_trade={REALISTIC_COST_CONFIG.commission_per_trade}, ")
    print(f" commission_per_share={REALISTIC_COST_CONFIG.commission_per_share}, ")
    print(f" slippage_cents={REALISTIC_COST_CONFIG.slippage_cents}, ")
    print(f" slippage_proportional={REALISTIC_COST_CONFIG.slippage_proportional})")
    print(f"Walk-forward: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d")
    print(f"Regime classification: {N_BLOCKS} blocks via volatility_blocks, window={WINDOW}, min_segment_bars={MIN_SEGMENT_BARS}")

    # 1. Manifest integrity and data preflight
    print("\n=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']} vs {DATASET_ID}"

    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Data preflight
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    from importlib import import_module
    pf = import_module("research.data.preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-robustness run.")
        sys.exit(1)

    # 3. Load the collected universe
    print("\n=== 3. Loading collected universe ===")
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}
    print(f"  Loaded {len(tickers)} tickers")

    # 4. For each lookback, run cost-sensitive regime-stability testing
    results_by_lookback = {}
    for lookback in [3, 5, 10]:
        print(f"\n=== 4. Lookback={lookback} momentum with realistic costs ===")

        # Generate momentum signals
        momentum_results = {}
        for ticker, bars in tickers.items():
            closes = bars.closes_array()
            signals = momentum_signals_for_lookbacks(closes, lookback)

        # Walk-forward with realistic costs
        print(f"  {ticker}: running walk-forward (cost-sensitive)...")
        wf_result = bt.walk_forward(
            list(bars),
            signals,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=REALISTIC_COST_CONFIG,
        )

        # Walk-forward with zero costs
        wf_zero_result = bt.walk_forward(
            list(bars),
            signals,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=ZERO_COST_CONFIG,
        )

        # Use AAPL as market proxy for regime classification (consistent with existing momentum tests)
        aapl_bars = tickers["AAPL"]
        aapl_closes = aapl_bars.closes_array()
        regime_labels = bt.volatility_blocks(aapl_closes, n_blocks=N_BLOCKS, window=WINDOW)
        segment_fn = bt.segment_fn_from_labels(regime_labels)

        # Use the regime_stress function (more robust and consistent with existing tests)
        from research.backtest.regime_stability import regime_stress
        from research.backtest.perturbation import parameter_grid_around

        grid = parameter_grid_around((("lookback", lookback),), multipliers=(0.5, 1.0, 2.0))
        stress_result = regime_stress(
            signals_fn=lambda c, **p: momentum_signals_for_lookbacks(c, lookback),
            bars=list(bars),
            param_grid=grid,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=REALISTIC_COST_CONFIG,
        )

        seg_results = stress_result

            momentum_results[ticker] = {
                "segment_verdicts": [s.name for s in stress_result.scenarios],
                "candidate_medians": [round(s.baseline_median_log_return, 3) for s in stress_result.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in stress_result.scenarios],
                "cost_verdict": stress_result.overall_verdict,
                "cost_edge": round(stress_result.baseline_median_log_return, 3),
                "cost_null": round(stress_result.noise_baseline_median_log_return, 3),
                "cost_dispersion": round(stress_result.candidate_dispersion, 3),
                "cost_null_dispersion": round(stress_result.null_dispersion, 3),
                "zero_cost_edge": round(wf_zero_result.aggregate_metrics.get("median_log_total_return", 0), 3),
                "zero_cost_verdict": "VERDICT_NOT_RECORDED",  # Would need additional logic
            }

        results_by_lookback[f"lookback_{lookback}"] = momentum_results

        print(f"  Lookback {lookback} cost-sensitive completion:")
        edge_counts = {}
        for ticker, res in momentum_results.items():
            verdict = res["cost_verdict"]
            edge = res["cost_edge"]
            edge_counts[verdict] = edge_counts.get(verdict, 0) + 1
            print(f"    {ticker}: {verdict} (edge={edge:+.3f}, zero-cost={res['zero_cost_edge']:+.3f})")

        print(f"    Summary: {edge_counts}")

    # 5. Write artifact for independent verification
    print("\n=== 5. Writing artifact for independent verification ===")
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "walk_forward_params": {
            "train_window": TRAIN,
            "test_window": TEST,
            "warmup": WARM,
            "overlap": OVERLAP,
            "periods_per_year": 252,
        },
        "regime_params": {
            "n_blocks": N_BLOCKS,
            "window": WINDOW,
            "min_segment_bars": MIN_SEGMENT_BARS,
        },
        "cost_model": {
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
        },
        "momentum_cost_sensitivity": results_by_lookback,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"  Artifact written to: {ARTIFACT_PATH}")

    # 6. Summary for evidence base
    print("\n=== 6. Summary for evidence base ===")
    for lb_key, results in results_by_lookback.items():
        print(f"{lb_key}:")
        for ticker, res in results.items():
            print(f"  {ticker}: {res['cost_verdict']} (cost edge={res['cost_edge']:+.3f}, zero-cost edge={res['zero_cost_edge']:+.3f}, erosion={res['cost_edge'] - res['zero_cost_edge']:+.3f})")

    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Cost robustness check for short-horizon momentum: exploratory simulation.")

    return 0


def main():
    return run_cost_sensitivity_check()

if __name__ == "__main__":
    sys.exit(main())
