#!/usr/bin/env python3
"""
Simple cost-sensitivity check for momentum lookback 3/5/10 using existing framework.

This follows the same pattern as existing checks in the repository but focuses
on cost-robustness for the qualified momentum evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


# Predeclared cost grid (realistic transaction costs)
COSTS = [
    {
        "name": "baseline",
        "commission_per_trade": 0.0,
        "commission_per_share": 0.0,
        "slippage_cents": 0.0,
        "slippage_proportional": 0.0,
    },
    {
        "name": "low_cost",
        "commission_per_trade": 0.5,
        "commission_per_share": 0.00001,
        "slippage_cents": 0.5,
        "slippage_proportional": 0.0001,
    },
]


# Lookback grid for cost robustness testing
LOOKBACKS = [3, 5, 10]


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Manifest integrity
    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")

    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Data preflight
    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # 3. Load universe
    print("\n=== 3. Load collected universe ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars
        print(f"  {ticker}: {bars.n_bars} bars")

    # 4. Run cost-sensitivity test
    print("\n=== 4. Cost-sensitivity test for momentum lookback 3/5/10 ===")

    results = {}

    for cost_config in COSTS:
        print(f"\nCost scenario: {cost_config['name']}")

        cost_results = {}

        for lookback in LOOKBACKS:
            print(f"  Lookback {lookback}:")

            # Create config with this cost
            cfg = bt.BacktestConfig(
                commission_per_trade=cost_config["commission_per_trade"],
                commission_per_share=cost_config["commission_per_share"],
                slippage_cents=cost_config["slippage_cents"],
                slippage_proportional=cost_config["slippage_proportional"],
            )

            # Build per-asset results
            per_asset = {}

        for ticker, bars in tickers.items():
            closes = bars.closes_array()
            signals = bt.regime_stability.momentum_signals(closes, lookback)

            # Use existing regime_stability infrastructure
            labels = bt.regime_stability.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)

            seg_fn = bt.regime_stability.segment_fn_from_labels(labels)

            res = bt.regime_stability.stress_segments(
                signals_fn=lambda c, **kw: bt.regime_stability.momentum_signals(c, lookback),
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lookback}],
                baseline=(("lookback", lookback),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=cfg,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )

            per_asset[ticker] = {
                "verdict": res.overall_verdict,
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
                "verdict_counts": res.verdict_counts,
            }

            cost_results[lookback] = per_asset

        results[cost_config["name"]] = cost_results

        # Print summary for this cost scenario
        print(f"  Cost scenario summary:")
        for lookback, per_asset in cost_results.items():
            n_stable = sum(1 for data in per_asset.values() if data["verdict"] == "REGIME_STABLE")
            n_noise = sum(1 for data in per_asset.values() if data["verdict"] == "CONSISTENT_WITH_NOISE")
            n_dependent = sum(1 for data in per_asset.values() if data["verdict"] == "REGIME_DEPENDENT")
            n_loss = sum(1 for data in per_asset.values() if data["verdict"] == "REGIME_STABLE_LOSS")

            print(f"    Lookback {lookback}: REGIME_STABLE={n_stable}, CONSISTENT_WITH_NOISE={n_noise}, "
                  f"REGIME_DEPENDENT={n_dependent}, REGIME_STABLE_LOSS={n_loss}")

    # 5. Write artifact
    print("\n=== 5. Write artifact ===")

    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookback_grid": LOOKBACKS,
        "cost_scenarios": [
            {
                "cost_name": cost_name,
                "cost_config": COSTS[i],
                "results_by_lookback": {
                    str(lookback): {
                        "per_asset_verdicts": {
                            ticker: data["verdict"]
                            for ticker, data in cost_results[lookback].items()
                        },
                        "verdict_counts": {
                            ticker: cost_results[lookback][ticker]["verdict_counts"]
                            for ticker in cost_results[lookback]
                        },
                    }
                    for lookback, cost_results in results[cost_name].items()
                },
            }
            for i, cost_name in enumerate(results.keys())
        ],
        "manifest": manifest,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Cost-sensitivity check for momentum (lookback 3/5/10) with realistic transaction costs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
