#!/usr/bin/env python3
"""Cost-sensitivity check for momentum lookback 3/5/10 under realistic transaction costs.

Objective: Test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe using the repository's
existing realistic cost model and matched null.

Out of scope: Do not tune signal rules after seeing cost results, expand the universe, or introduce leverage.
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
COST_REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.003,      # 3 bps per share
    slippage_cents=5.0,               # 5 cents per share
    warmup_periods=0,
    periods_per_year=252,
)
COST_ZERO = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    warmup_periods=0,
    periods_per_year=252,
)
ARTIFACT_DIR = Path.cwd() / "state" / "cost_sensitivity_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def run_cost_sensitivity_test() -> dict:
    """Run cost-sensitive momentum tests for lookback 3, 5, 10."""
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

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # Load all tickers from the collected universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 3. Cost-sensitive momentum lookback sweep (3 / 5 / 10) ===")
    print(f"  Params: walk-forward train={TRAIN}d/test={TEST}d/warmup={WARM}d/"
          f"overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs coin-flip null\n")
    print(f"  Cost regimes: ZERO (commission_per_share=0, slippage_cents=0) vs "
          f"REALISTIC (commission_per_share=0.003, slippage_cents=5.0)")

    # Results indexed [lookback][ticker][cost_regime][segment_index]
    results = {}
    for lb in LOOKBACKS:
        results[lb] = {}
        for ticker in ticker_order:
            results[lb][ticker] = {}

    # Run test for each lookback and cost regime
    for cost_name, cfg in [("ZERO", COST_ZERO), ("REALISTIC", COST_REALISTIC)]:
        print(f"\n  Running {cost_name} cost regime:")
        for lb in LOOKBACKS:
            print(f"    Lookback {lb}:")
            for i, ticker in enumerate(ticker_order):
                if i % 5 == 0:
                    print(f"      {ticker}...", end="")

                bars = all_tickers[ticker]
                closes = bars.closes_array()
                labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
                seg_fn = bt.segment_fn_from_labels(labels)
                signals = bt.momentum_signals(closes, lookback=lb)

                if len(signals) != len(closes):
                    raise ValueError(f"{ticker} lookback {lb}: signals {len(signals)} != bars {len(closes)}")

                res = bt.stress_segments(
                    signals_fn=bt.momentum_signals,
                    bars=list(bars),
                    signals=signals,
                    param_grid=[{"lookback": lb}],
                    baseline=(("lookback", lb),),
                    segment_fn=seg_fn,
                    train_window=TRAIN,
                    test_window=TEST,
                    warmup=WARM,
                    overlap_window=OVERLAP,
                    periods_per_year=252,
                    min_segment_bars=MIN_SEGMENT_BARS,
                    cfg=cfg,
                )

                segment_results = []
                for s in res.scenarios:
                    segment_results.append({
                        "segment": s.name,
                        "baseline_median_log_return": round(s.baseline_median_log_return, 3),
                        "noise_median_log_return": round(s.noise_median_log_return, 3),
                    })

                results[lb][ticker][cost_name] = {
                    "segments": segment_results,
                    "cost_regime": cost_name,
                    "lookback": lb,
                    "ticker": ticker,
                    "candidate_dispersion": round(res.candidate_dispersion, 3),
                    "null_dispersion": round(res.null_dispersion, 3),
                    "n_folds": res.n_folds,
                    "overall_verdict": res.overall_verdict,
                }

                # Progress output
                if i % 5 == 4 or i == len(ticker_order) - 1:
                    print(f" [{i+1}/{len(ticker_order)}]")

    # Analyze cost impact
    print("\n=== 4. Cost impact analysis ===")
    cost_impact_summary = {}

    for lb in LOOKBACKS:
        print(f"\n  Lookback {lb}:")
        cost_impact_summary[lb] = {}

        for ticker in ticker_order:
            zero_median = np.mean([
                r["baseline_median_log_return"] 
                for r in results[lb][ticker]["ZERO"]["segments"]
            ])
            realistic_median = np.mean([
                r["baseline_median_log_return"] 
                for r in results[lb][ticker]["REALISTIC"]["segments"]
            ])

            cost_impact = realistic_median - zero_median
            relative_impact = cost_impact / abs(zero_median) if abs(zero_median) > 0 else 0.0

            cost_impact_summary[lb][ticker] = {
                "zero_cost_median": round(zero_median, 4),
                "realistic_cost_median": round(realistic_median, 4),
                "absolute_cost_impact": round(cost_impact, 4),
                "relative_cost_impact_pct": round(relative_impact * 100, 2),
                "edge_survives_realistic_costs": bool(realistic_median > 0),
                "edge_survives_zero_costs": bool(zero_median > 0),
            }

            print(f"    {ticker}: zero_cost={zero_median:+.3f}, realistic_cost={realistic_median:+.3f}, "
                  f"impact={cost_impact:+.3f} ({relative_impact*100:+.1f}%) "
                  f"survives={realistic_median > 0}")

    # Determine overall verdict
    print("\n=== 5. Overall cost-robustness verdict ===")
    universe_cost_robustness = {}

    for lb in LOOKBACKS:
        print(f"\n  Lookback {lb}:")
        robust_count = 0
        total_count = 0
        surviving_tickers = []

        for ticker in ticker_order:
            impact_data = cost_impact_summary[lb][ticker]
            total_count += 1
            if impact_data["edge_survives_realistic_costs"]:
                robust_count += 1
                surviving_tickers.append(ticker)

        survival_rate = (robust_count / total_count) * 100 if total_count > 0 else 0.0
        universe_cost_robustness[lb] = {
            "robust_assets_count": robust_count,
            "total_assets_count": total_count,
            "survival_rate_pct": round(survival_rate, 1),
            "surviving_tickers": surviving_tickers,
            "cost_robustness_verdict": "ROBUST" if survival_rate >= 50 else "FRAGILE" if survival_rate > 0 else "NO_EDGE"
        }

        print(f"    ROBUST assets: {robust_count}/{total_count} ({survival_rate:.1f}%)")
        print(f"    Surviving tickers: {', '.join(surviving_tickers)}")
        print(f"    Verdict: {universe_cost_robustness[lb]['cost_robustness_verdict']}")

    # Determinism test: run a subset to verify reproducibility
    print("\n=== 6. Determinism verification ===")
    target = ["AAPL", "AMZN", "MSFT", "TSLA"]
    determinism_passed = True

    for lb in LOOKBACKS:
        for ticker in target:
            # Run with realistic costs
            bars = all_tickers[ticker]
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            signals = bt.momentum_signals(closes, lookback=lb)

            res1 = bt.stress_segments(
                signals_fn=bt.momentum_signals,
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=COST_REALISTIC,
            )

            # Run same test again
            res2 = bt.stress_segments(
                signals_fn=bt.momentum_signals,
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=COST_REALISTIC,
            )

            # Compare results
            verdict1 = res1.overall_verdict
            verdict2 = res2.overall_verdict

            if verdict1 != verdict2:
                print(f"    DETERMINISM FAIL: {ticker} lookback {lb} verdict mismatch: {verdict1} vs {verdict2}")
                determinism_passed = False
            else:
                print(f"    {ticker} lookback {lb}: deterministic ({verdict1})")

    if determinism_passed:
        print("  All determinism tests passed.")
    else:
        print("  DETERMINISM TESTS FAILED - stopping execution.")
        sys.exit(1)

    # Prepare artifact for independent verification
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "lookbacks": list(LOOKBACKS),
        "cost_regimes": {
            "ZERO": {
                "commission_per_share": COST_ZERO.commission_per_share,
                "slippage_cents": COST_ZERO.slippage_cents,
            },
            "REALISTIC": {
                "commission_per_share": COST_REALISTIC.commission_per_share,
                "slippage_cents": COST_REALISTIC.slippage_cents,
            }
        },
        "universe": {
            "ticker_order": ticker_order,
            "cost_impact_summary": cost_impact_summary,
            "universe_cost_robustness": universe_cost_robustness,
        },
        "raw_results": results,
        "determinism_passed": determinism_passed,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-sensitivity test across the collected universe: exploratory simulation.")

    return artifact


if __name__ == "__main__":
    try:
        result = run_cost_sensitivity_test()
        print("\n=== EXECUTION COMPLETE ===")
        print("Cost-sensitivity check completed successfully.")
        print(f"Key finding: Lookback 3/5/10 momentum edge survives realistic costs "
              f"on {result['universe']['universe_cost_robustness'][3]['survival_rate_pct']}% of assets")
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)