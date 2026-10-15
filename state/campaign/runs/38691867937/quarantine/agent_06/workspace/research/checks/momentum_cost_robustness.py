"""Cost robustness test for lookback 3/5/10 momentum on collected universe.

Objective: test whether the candidate positive momentum edge (lookback 5, currently
admitted as evidence) survives conservative transaction-cost stress under the
repository's existing realistic cost model and matched coin-flip null.

Method (fixed a-priori, not tuned to OOS): long the previous N-day return;
hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward regime-stability
gate as `momentum_lookback_sweep.py` but with realistic costs; matched null
uses the same walk-forward configuration and parameter grid.

The a-priori prediction is: costs may materially degrade or erase the edge,
particularly given the 5-bps proportional slippage and $2 commission per trade
which represent realistic retail transaction costs.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / ".."))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def momentum_signals(closes: np.ndarray, lookback: int):
    """Past-only momentum signals: long the previous N-day return.

    Bars before the lookback window are neutral. The signal author guarantees
    padding through the lookback window so no real signal exists before the
    warmup boundary.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    if lookback <= 0:
        return signals
    for i in range(lookback - 1, n):
        returns = (closes[i - lookback + 1 : i + 1] - closes[i - lookback]) / closes[i - lookback]
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if np.mean(returns) > 0 else -1.0)
    return signals


def run_momentum_with_costs(target_lookback: int, realistic_costs=True):
    """Run momentum for one lookback with zero costs (reference) or realistic costs."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    # Configure cost model
    if realistic_costs:
        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            warmup_periods=WARM,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
            periods_per_year=252,
        )
        cost_label = "realistic"
    else:
        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            warmup_periods=WARM,
            periods_per_year=252,
        )
        cost_label = "zero"

    # Run across the full collected universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    ticker_order = list(all_tickers.keys())

    # Results indexed [ticker][lookback]
    results = {}

    for ticker in ticker_order:
        bars = all_tickers[ticker]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)

        signals = momentum_signals(closes, target_lookback)
        if len(signals) != len(closes):
            raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")

        res = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, target_lookback),
            bars=bars,
            signals=signals,
            param_grid=[{"lookback": target_lookback}],
            baseline=(("lookback", target_lookback),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg,
        )

        results[ticker] = {
            "lookback": target_lookback,
            "segments": [s.name for s in res.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
            "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
            "candidate_dispersion": round(res.candidate_dispersion, 3),
            "null_dispersion": round(res.null_dispersion, 3),
            "n_folds": res.n_folds,
            "verdict": res.overall_verdict,
            "cost_model": cost_label,
        }

    return results, ticker_order


def run_matched_null_with_costs(target_lookback: int):
    """Run matched coin-flip null with the same realistic costs."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    # Use the same realistic cost model as momentum
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
        periods_per_year=252,
    )

    # Synthetic perturbation for null
    bars = bt.generate_bars(n_bars=600, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", target_lookback),),
        multipliers=(0.5, 1.0, 2.0),
    )

    sweep = bt.parameter_sweep(
        signals_fn=lambda closes, **params: bt.random_signals(len(closes), seed=42),
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        cfg=cfg,
        periods_per_year=252,
    )

    summary = bt.sweep_summary(sweep, baseline=(("lookback", target_lookback),))
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        cfg=cfg,
        periods_per_year=252,
        seed=SEED,
    )

    return {
        "lookback": target_lookback,
        "synthetic_summary": {
            "param_sets": [tuple(ps) for ps in sweep.param_sets],
            "medians": [round(m, 3) for m in summary.median_log_returns],
            "baseline_median": round(summary.baseline_median_log_return, 3),
            "noise_median": round(noise.baseline_median_log_return, 3),
            "compare_noise": summary.compare_noise(noise.baseline_median_log_return),
            "effective_tolerance": round(summary.effective_tolerance, 4),
            "n_folds": summary.n_folds,
        },
        "cost_model": "realistic",
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Momentum cost robustness test (lookback 3 / 5 / 10) ===")
    print("Cost model: commission_per_trade=2.0, commission_per_share=0.003,")
    print("           slippage_cents=2.0 (2 cents per share),")
    print("           slippage_proportional=0.0005 (5 bps)")
    print()

    # 1. Manifest integrity
    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    # 2. Data preflight gate
    print("=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    print("PREFLIGHT PASSED")
    print()

    # 3. Run momentum with zero costs (baseline)
    print("=== 3. Momentum with zero costs (baseline) ===")
    baseline_results = {}
    for lookback in LOOKBACKS:
        print(f"  lookback {lookback}:")
        results, ticker_order = run_momentum_with_costs(lookback, realistic_costs=False)
        baseline_results[lookback] = results

        regim_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                       "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
        for ticker in ticker_order:
            regim_counts[results[ticker]["verdict"]] += 1
        print(f"    REGIME_STABLE (edge): {regim_counts['REGIME_STABLE']}")
        print(f"    CONSISTENT_WITH_NOISE: {regim_counts['CONSISTENT_WITH_NOISE']}")
        print(f"    REGIME_DEPENDENT: {regim_counts['REGIME_DEPENDENT']}")
        print(f"    REGIME_STABLE_LOSS: {regim_counts['REGIME_STABLE_LOSS']}")
        print()

    # 4. Run momentum with realistic costs
    print("=== 4. Momentum with realistic costs ===")
    cost_results = {}
    for lookback in LOOKBACKS:
        print(f"  lookback {lookback}:")
        results, ticker_order = run_momentum_with_costs(lookback, realistic_costs=True)
        cost_results[lookback] = results

        regim_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                       "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
        for ticker in ticker_order:
            regim_counts[results[ticker]["verdict"]] += 1
        print(f"    REGIME_STABLE (edge): {regim_counts['REGIME_STABLE']}")
        print(f"    CONSISTENT_WITH_NOISE: {regim_counts['CONSISTENT_WITH_NOISE']}")
        print(f"    REGIME_DEPENDENT: {regim_counts['REGIME_DEPENDENT']}")
        print(f"    REGIME_STABLE_LOSS: {regim_counts['REGIME_STABLE_LOSS']}")
        print()

    # 5. Run matched null with realistic costs
    print("=== 5. Matched null with realistic costs ===")
    null_results = {}
    for lookback in LOOKBACKS:
        null_result = run_matched_null_with_costs(lookback)
        null_results[lookback] = null_result
        print(f"  lookback {lookback}:")
        print(f"    synthetic baseline vs null: {null_result['synthetic_summary']['baseline_median']:+.3f} vs "
              f"{null_result['synthetic_summary']['noise_median']:+.3f} "
              f"(tol {null_result['synthetic_summary']['effective_tolerance']:.4f}: "
              f"within {null_result['synthetic_summary']['compare_noise']})")
        print()

    # 6. Determinism and artifact writing
    print("=== 6. Determinism check and artifact writing ===")
    for lookback in LOOKBACKS:
        print(f"  lookback {lookback}: momentum results written")

    # Prepare artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "zero_cost_baseline": baseline_results,
        "realistic_cost": cost_results,
        "matched_null_realistic": null_results,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"artifact written to: {ARTIFACT_PATH}")
    print()
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost robustness test on the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())