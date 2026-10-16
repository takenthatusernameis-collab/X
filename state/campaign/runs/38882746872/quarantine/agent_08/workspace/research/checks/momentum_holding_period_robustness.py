"""Momentum holding-period robustness test.

This agent tests whether the qualified momentum effect survives a small
predeclared holding-period variation without becoming a single-point timing
artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLDING_PERIODS = [1, 2, 3, 5]
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_period_robustness.json"


def momentum_signals_with_holding(
    closes: np.ndarray,
    lookback: int,
    holding: int,
) -> list[bt.Signal]:
    """Past-only momentum signal with variable holding period.

    Long the previous ``lookback``-day return and hold for ``holding`` days:
    if the last ``lookback`` days were up, go long for ``holding`` days;
    if they were down, go short for ``holding`` days.
    Neutral before the ``lookback`` window. Daily rebalancing after holding.
    """
    n = len(closes)
    out: list[bt.Signal] = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    holding = int(holding)

    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        weight = 1.0 if ret > 0 else -1.0

        for h in range(holding):
            signal_idx = i + h
            if signal_idx < n:
                out[signal_idx] = bt.Signal(date=signal_idx + 1, weight=weight)

    return out


def run_holding_period_test(asset: str, tickers: dict) -> dict:
    """Run walk-forward tests for all holding periods on one asset."""
    bars = tickers[asset]
    closes = bars.closes_array()

    results = {}

    for hold_days in HOLDING_PERIODS:
        print(f"  holding_period {hold_days} days")

        signals = momentum_signals_with_holding(closes, LOOKBACK, hold_days)

        dates = [b.date for b in bars]
        bt.check_signal_integrity(signals, dates, warmup=0)

        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            warmup_periods=WARM,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        )

        wf_result = bt.walk_forward(
            list(bars),
            signals,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=cfg,
        )

        null_summary = bt.noise_benchmark(
            list(bars),
            param_grid=[{"holding": hold_days}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=cfg,
            seed=SEED,
        )

        n_folds = wf_result.aggregate_metrics["n_folds"]
        if n_folds > 1:
            null_std = null_summary.noise_fold_median_log_returns.std(ddof=1)
            effective_tolerance = 2.0 * null_std / np.sqrt(n_folds)
        else:
            effective_tolerance = 0.0

        median_log_return = wf_result.aggregate_metrics["median_log_total_return"]
        baseline_within_tolerance = abs(median_log_return) <= effective_tolerance
        verdict = "CONSISTENT_WITH_NOISE" if baseline_within_tolerance else "EDGE_PRESENT"

        results[str(hold_days)] = {
            "median_log_return": round(median_log_return, 4),
            "n_folds": n_folds,
            "effective_tolerance": round(effective_tolerance, 4),
            "baseline_within_tolerance": baseline_within_tolerance,
            "verdict": verdict,
            "null_baseline": round(null_summary.baseline_median_log_return, 4),
        }

    return results


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED")
        sys.exit(1)

    print("\n=== 3. Momentum holding-period robustness test ===")
    print(f"Holding periods: {HOLDING_PERIODS} days")
    print(f"Lookback: {LOOKBACK} days (fixed)")
    print(f"Assets: {len(manifest['universe'])} (full universe sweep)")

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    print("\n=== Running tests on REGIME_DEPENDENT assets (AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    all_results = {}
    for asset in target:
        print(f"\nAsset: {asset}")
        results = run_holding_period_test(asset, tickers)
        all_results[asset] = results

        print("  holding_period  | median_log_return | effective_tolerance | verdict")
        print("  ----------------|------------------|--------------------|--------")
        for hold_str, res in results.items():
            print(f"  {hold_str:14s} | {res['median_log_return']:16.4f} | {res['effective_tolerance']:17.4f} | "
                  f"{res['verdict']}")

    print("\n=== 4. Full universe robustness ===")
    universe_summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: momentum_signals_with_holding(
            c, LOOKBACK, kw.get("holding", 1)
        ),
        regime_labels_fn=lambda c: bt.volatility_blocks(
            c, n_blocks=N_BLOCKS, window=WINDOW
        ),
        baseline=("holding", 1),
        param_grid=[{"holding": 1}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("  " + universe_summary.inspect().replace("\n", "\n  "))

    print("\n=== 5. Determinism check ===")
    results_1 = run_holding_period_test("AMZN", tickers)
    results_5 = run_holding_period_test("JPM", tickers)

    out1 = json.dumps(results_1, sort_keys=True)
    out5 = json.dumps(results_5, sort_keys=True)
    print(f"determinism: AMZN holding=1 == JPM holding=5: {out1 == out5}")
    assert out1 == out5, "non-deterministic output"

    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookback": LOOKBACK,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "holding_periods": HOLDING_PERIODS,
        "per_asset": all_results,
        "universe": {
            "verdict_counts": universe_summary.verdict_counts,
            "ticker_order": universe_summary.ticker_order,
        },
    }
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== Artifact written to: {ARTIFACT_PATH} ===")
    print("\nNote: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
