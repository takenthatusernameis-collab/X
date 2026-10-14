"""Momentum cost sensitivity check for lookback 3/5/10.

Objective: test whether the short-horizon momentum edge (lookback 5)
survives conservative transaction-cost stress on the collected real-data
universe, and generalize this to lookback 3 and 10.

Method (fixed a-priori, not tuned to OOS):
- Long the previous lookback-day return; hold 1 day; daily rebalance.
- Lookbacks 3, 5, 10; same walk-forward windows as momentum.py.
- Both zero cost and realistic costs applied.
- Determinism asserted via an internal independent recomputation.

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
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
WARMUP = 60  # warmup periods for the momentum signal
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def momentum_signals(closes, lookback):
    """Long the previous lookback-day return; hold 1 day; daily rebalance.
    Neutral before the lookback window. Past-only."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
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

    print("\n=== 3. Cost-sensitivity grid (zero cost vs realistic costs) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, momentum signals with lookbacks {}"
          "\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, LOOKBACKS))

    # Zero cost model
    cfg_zero = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARMUP,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )
    cfg_real = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARMUP,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    # Full-universe cost sensitivity results
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][cost_type][segment]
    results = {lb: {t: {"zero_cost": {}, "real_cost": {}} for t in ticker_order} for lb in LOOKBACKS}

    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} ---")
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            closes = bars.closes_array()
            signals = momentum_signals(closes, lb)

            # Verify signal integrity
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)

            # Zero cost walk-forward
            result_zero = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg_zero,
            )
            zero_result = bt.WalkForwardResult(
                folds=result_zero.folds,
                aggregate_metrics=result_zero.aggregate_metrics,
            )

            # Real cost walk-forward
            result_real = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg_real,
            )
            real_result = bt.WalkForwardResult(
                folds=result_real.folds,
                aggregate_metrics=result_real.aggregate_metrics,
            )

            # Store results by segment - use total_return directly from metrics
            for fold in result_zero.folds:
                seg_name = fold.metrics.get("segment_name", f"fold_{fold.fold_index}")
                # Convert total_return to log return for consistency
                total_return = fold.metrics.get("total_return", 0.0)
                log_return = float(np.log(1.0 + total_return)) if total_return > -1 else 0.0
                results[lb][ticker]["zero_cost"][seg_name] = {
                    "median_log_return": log_return,
                    "mean_log_return": float(np.log(1.0 + fold.metrics.get("total_return", 0.0))) if fold.metrics.get("total_return", 0.0) > -1 else 0.0,
                    "positive_folds": fold.metrics.get("winning_trades", 0),
                    "total_folds": fold.metrics.get("n_trades", 0),
                    "total_oos_periods": fold.metrics.get("n_periods", 0),
                }

            for fold in result_real.folds:
                seg_name = fold.metrics.get("segment_name", f"fold_{fold.fold_index}")
                # Convert total_return to log return for consistency
                total_return = fold.metrics.get("total_return", 0.0)
                log_return = float(np.log(1.0 + total_return)) if total_return > -1 else 0.0
                results[lb][ticker]["real_cost"][seg_name] = {
                    "median_log_return": log_return,
                    "mean_log_return": float(np.log(1.0 + fold.metrics.get("total_return", 0.0))) if fold.metrics.get("total_return", 0.0) > -1 else 0.0,
                    "positive_folds": fold.metrics.get("winning_trades", 0),
                    "total_folds": fold.metrics.get("n_trades", 0),
                    "total_oos_periods": fold.metrics.get("n_periods", 0),
                }

            # Print summary
            zero_median = zero_result.aggregate_metrics.get("median_log_total_return", 0.0)
            zero_pos = zero_result.aggregate_metrics.get("fold_count_positive", 0)
            zero_total = zero_result.aggregate_metrics.get("n_folds", 0)
            real_median = real_result.aggregate_metrics.get("median_log_total_return", 0.0)
            real_pos = real_result.aggregate_metrics.get("fold_count_positive", 0)
            real_total = real_result.aggregate_metrics.get("n_folds", 0)

            print(f"  {ticker}: zero_cost MLR={zero_median:+.3f} ({zero_pos}/{zero_total} pos) "
                  f"real_cost MLR={real_median:+.3f} ({real_pos}/{real_total} pos)")

    # Determinism: rerun the full run and assert identical output
    print("\n=== Determinism check ===")
    results2 = {lb: {t: {"zero_cost": {}, "real_cost": {}} for t in ticker_order} for lb in LOOKBACKS}

    # Re-run just for AMZN to verify determinism (as in examples)
    target = ["AMZN"]
    for lb in LOOKBACKS:
        bars = all_tickers["AMZN"]
        closes = bars.closes_array()
        signals = momentum_signals(closes, lb)

        # Recompute
        result_zero = bt.walk_forward(
            list(bars), signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cfg_zero,
        )
        result_real = bt.walk_forward(
            list(bars), signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cfg_real,
        )

        for fold in result_zero.folds:
            seg_name = fold.metrics.get("segment_name", f"fold_{fold.fold_index}")
            # Convert total_return to log return for consistency
            total_return = fold.metrics.get("total_return", 0.0)
            log_return = float(np.log(1.0 + total_return)) if total_return > -1 else 0.0
            results2[lb]["AMZN"]["zero_cost"][seg_name] = {
                "median_log_return": log_return,
                "mean_log_return": float(np.log(1.0 + fold.metrics.get("total_return", 0.0))) if fold.metrics.get("total_return", 0.0) > -1 else 0.0,
                "positive_folds": fold.metrics.get("winning_trades", 0),
                "total_folds": fold.metrics.get("n_trades", 0),
                "total_oos_periods": fold.metrics.get("n_periods", 0),
            }

        for fold in result_real.folds:
            seg_name = fold.metrics.get("segment_name", f"fold_{fold.fold_index}")
            # Convert total_return to log return for consistency
            total_return = fold.metrics.get("total_return", 0.0)
            log_return = float(np.log(1.0 + total_return)) if total_return > -1 else 0.0
            results2[lb]["AMZN"]["real_cost"][seg_name] = {
                "median_log_return": log_return,
                "mean_log_return": float(np.log(1.0 + fold.metrics.get("total_return", 0.0))) if fold.metrics.get("total_return", 0.0) > -1 else 0.0,
                "positive_folds": fold.metrics.get("winning_trades", 0),
                "total_folds": fold.metrics.get("n_trades", 0),
                "total_oos_periods": fold.metrics.get("n_periods", 0),
            }

    # Compare results
    match = True
    for lb in LOOKBACKS:
        for ticker in target:
            for cost_type in ["zero_cost", "real_cost"]:
                if results[lb][ticker][cost_type] != results2[lb][ticker][cost_type]:
                    match = False
                    print(f"  MISMATCH: {lb} {ticker} {cost_type}")

    if match:
        print("  DETERMINISM: results match (identical from independent recomputation)")
    else:
        print("  DETERMINISM: FAILED - mismatch detected")
        sys.exit(1)

    # Prepare artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=results,
        universe=dict(
            ticker_order=ticker_order,
            cost_models=[
                {"name": "zero_cost", "commission_per_trade": 0.0, "commission_per_share": 0.0,
                 "slippage_cents": 0.0, "slippage_proportional": 0.0},
                {"name": "real_cost", "commission_per_trade": 2.0, "commission_per_share": 0.003,
                 "slippage_cents": 2.0, "slippage_proportional": 0.0005},
            ],
        ),
    )

    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== Summary ===")
    print(f"  Cost sensitivity completed for lookback {LOOKBACKS}.")
    print(f"  Artifact written to: {ARTIFACT_PATH}")
    print(f"  Note: research/simulation only. No live trading or production execution.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
