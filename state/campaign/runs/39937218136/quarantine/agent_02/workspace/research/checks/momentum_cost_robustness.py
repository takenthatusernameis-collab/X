"""Momentum cost robustness check.

Objective: test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe.

Method: run the momentum signal (long previous N-day return, hold 1 day, daily
rebalanced) across lookbacks 3 / 5 / 10, comparing against realistic cost
models and matched nulls. Determinism is asserted via an internal
independent recomputation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def run_momentum_with_cost(
    asset: str,
    tickers: dict,
    lookback: int,
    commission_per_trade: float = 0.0,
    commission_per_share: float = 0.0,
    slippage_cents: float = 0.0,
    slippage_proportional: float = 0.0,
    seed: Optional[int] = SEED,
) -> dict:
    """Run momentum backtest for one asset with specific cost parameters."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    # Configure cost model
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=commission_per_trade,
        commission_per_share=commission_per_share,
        slippage_cents=slippage_cents,
        slippage_proportional=slippage_proportional,
        warmup_periods=WARM,
    )

    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")

    # Run with cost
    res = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=lookback),
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )

    # Run matched null (coin-flip random signals)
    random_signals_list = bt.random_signals(len(closes), seed=seed)
    null_res = bt.stress_segments(
        signals_fn=lambda c, **p: bt.random_signals(len(c), seed=seed),
        bars=list(bars),
        signals=random_signals_list,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )

    return {
        "asset": asset,
        "lookback": lookback,
        "cost_params": {
            "commission_per_trade": commission_per_trade,
            "commission_per_share": commission_per_share,
            "slippage_cents": slippage_cents,
            "slippage_proportional": slippage_proportional,
        },
        "candidate_median_log_return": round(res.scenarios[0].baseline_median_log_return, 3),
        "candidate_null_median_log_return": round(null_res.scenarios[0].baseline_median_log_return, 3),
        "candidate_verdict": res.overall_verdict,
        "null_verdict": null_res.overall_verdict,
        "vs_null": res.compare_noise("calm", tol=0.05),
        "n_folds": res.n_folds,
        "cost_impact": round(
            res.scenarios[0].baseline_median_log_return - null_res.scenarios[0].baseline_median_log_return,
            3,
        ),
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

    print("\n=== 3. Loading all tickers ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    print(f"  Loaded {len(tickers)} tickers")

    print("\n=== 4. Momentum cost robustness ===")
    print("Testing lookbacks: {} with cost scenarios".format(LOOKBACKS))

    # Run cost scenarios
    cost_scenarios = [
        {"name": "base", "commission_per_trade": 0.0, "commission_per_share": 0.0,
         "slippage_cents": 0.0, "slippage_proportional": 0.0},
        {"name": "moderate", "commission_per_trade": 5.0, "commission_per_share": 0.005,
         "slippage_cents": 2.0, "slippage_proportional": 0.001},
        {"name": "conservative", "commission_per_trade": 10.0, "commission_per_share": 0.01,
         "slippage_cents": 5.0, "slippage_proportional": 0.002},
    ]

    results_by_lookback = {}
    for lookback in LOOKBACKS:
        results_by_lookback[str(lookback)] = {}
        for scenario in cost_scenarios:
            print(f"  Processing {lookback}d with {scenario['name']} costs...")
            for asset in tickers.keys():
                result = run_momentum_with_cost(
                    asset, tickers, lookback,
                    commission_per_trade=scenario["commission_per_trade"],
                    commission_per_share=scenario["commission_per_share"],
                    slippage_cents=scenario["slippage_cents"],
                    slippage_proportional=scenario["slippage_proportional"],
                )
                if asset not in results_by_lookback[str(lookback)]:
                    results_by_lookback[str(lookback)][asset] = {}
                results_by_lookback[str(lookback)][asset][scenario["name"]] = result

    print("\n=== 5. Determinism verification ===")
    # Run one asset with each scenario to verify determinism
    test_asset = "AAPL"
    test_lookback = 5
    determinism_passed = True

    for scenario in cost_scenarios:
        print(f"  Verifying determinism for {test_lookback}d {scenario['name']} costs...")
        run1 = run_momentum_with_cost(
            test_asset, tickers, test_lookback,
            commission_per_trade=scenario["commission_per_trade"],
            commission_per_share=scenario["commission_per_share"],
            slippage_cents=scenario["slippage_cents"],
            slippage_proportional=scenario["slippage_proportional"],
            seed=SEED,
        )

        run2 = run_momentum_with_cost(
            test_asset, tickers, test_lookback,
            commission_per_trade=scenario["commission_per_trade"],
            commission_per_share=scenario["commission_per_share"],
            slippage_cents=scenario["slippage_cents"],
            slippage_proportional=scenario["slippage_proportional"],
            seed=SEED,
        )

        if run1 == run2:
            print(f"    PASS: deterministic output confirmed")
        else:
            print(f"    FAIL: non-deterministic output detected")
            determinism_passed = False

    if not determinism_passed:
        sys.exit(1)

    print("\n=== 6. Cost robustness summary ===")
    robustness_summary = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "cost_scenarios": [s["name"] for s in cost_scenarios],
        "per_asset_per_lookback": results_by_lookback,
        "determinism_verified": True,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(robustness_summary, f, indent=2, sort_keys=True)

    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost robustness check: exploratory simulation.")

    print("\n=== Cost robustness classification ===")
    for asset in sorted(tickers.keys()):
        print(f"\n  {asset}:")
        for lookback in LOOKBACKS:
            base_result = results_by_lookback[str(lookback)][asset]["base"]
            moderate_result = results_by_lookback[str(lookback)][asset]["moderate"]
            conservative_result = results_by_lookback[str(lookback)][asset]["conservative"]

            base_edge = base_result["vs_null"]
            moderate_edge = moderate_result["vs_null"]
            conservative_edge = conservative_result["vs_null"]

            print(f"    lookback {lookback}:")
            print(f"      base:     {base_result['candidate_median_log_return']:+.3f} vs null ({'EDGE' if base_edge else 'NO_EDGE'})")
            print(f"      moderate: {moderate_result['candidate_median_log_return']:+.3f} vs null ({'EDGE' if moderate_edge else 'NO_EDGE'})")
            print(f"      conservative: {conservative_result['candidate_median_log_return']:+.3f} vs null ({'EDGE' if conservative_edge else 'NO_EDGE'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
