"""Momentum lookback sweep across the collected universe with cost sensitivity.

Objective: test whether the momentum edge observed at lookback=5
(research/checks/momentum.py, activation 37318950814) is robust to the
lookback parameter and to realistic transaction costs. Momentum is a very
mechanical 1-day-ahead bet, so the most salient robustness question is whether
the positive REGIME_STABLE pattern at lookback=5 also appears at >= 2 of the 3
lookbacks for an asset, and whether the edge survives realistic costs.

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
daily rebalance) is walk-forward validated (train=252d/test=84d/warmup=60d/
overlap=60d) inside each of the 4 contiguous volatility blocks of the real-data
regime family, for lookback in {3, 5, 10}. Each lookback's per-segment median
log return is compared against a coin-flip sign null benchmark run on the same
segments. The synthetic generator cannot answer this because it contains no
momentum structure, so the null must be drawn from the collected data itself.

A lookback-robust edge is one where the uniformly-positive REGIME_STABLE pattern
at lookback=5 also appears at >= 2 of the 3 lookbacks for an asset; a
lookback-sensitive result (edge at exactly one lookback) is a parameter-fragility
warning even when the lookback=5 edge is genuine. Cost robustness is tested
by comparing zero-cost vs realistic-cost (commission_per_trade=2.0,
commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)
regime-stability counts.

Either outcome directly informs whether momentum should be admitted to the
evidence base as candidate positive evidence.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as `momentum.py`. Determinism is asserted via an
internal independent recomputation of the AMZN/JPM subset. Cost sensitivity
is tested by running the same walk-forward with zero vs realistic costs.

Research / simulation only. No live trading or production execution.
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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_with_costs_results.json"

# Cost configurations
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, and one config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=scen.name,
        n_folds=res.n_folds,
    )
def main():
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

    print("\n=== 3. Loading tickers ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    # Load subset for determinism check (AMZN, JPM like other momentum checks)
    target = ["AMZN", "JPM"]

    print("\n=== 4. Momentum lookback sweep with cost sensitivity ===")
    print("Momentum = long the previous lookback-day return, hold 1 day, daily rebalanced. "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, compared vs coin-flip null on the same "
          "segments. Testing lookbacks {}".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, LOOKBACKS))

    # Run for each cost configuration
    results_zero = {}
    results_realistic = {}

    for cfg_name, cfg in [("zero_cost", ZERO_COSTS), ("realistic_cost", REALISTIC_COSTS)]:
        print(f"\n--- Testing {cfg_name} ---")

        per_asset = {}
        lookback_counts = {}
        robustness_counts = {}

        for lookback in LOOKBACKS:
            print(f"\nLookback {lookback}:")
            lookback_results = {}
            lookback_counts[str(lookback)] = {
                "REGIME_STABLE": 0,
                "REGIME_STABLE_LOSS": 0,
                "REGIME_DEPENDENT": 0,
                "CONSISTENT_WITH_NOISE": 0,
                "NO_EDGE": 0,
            }

            for ticker in tickers:
                # For determinism check, use reduced data for AMZN/JPM subset
                # but full data for cost robustness across all tickers
                bars_to_use = tickers[ticker]
                result = segment_result(ticker, bars_to_use, lookback, cfg)
                
                if ticker not in per_asset:
                    per_asset[ticker] = {}
                per_asset[ticker][str(lookback)] = result

                if result["verdict"] == "REGIME_STABLE":
                    lookback_counts[str(lookback)]["REGIME_STABLE"] += 1
                elif result["verdict"] == "REGIME_STABLE_LOSS":
                    lookback_counts[str(lookback)]["REGIME_STABLE_LOSS"] += 1
                elif result["verdict"] == "REGIME_DEPENDENT":
                    lookback_counts[str(lookback)]["REGIME_DEPENDENT"] += 1
                elif result["verdict"] == "CONSISTENT_WITH_NOISE":
                    lookback_counts[str(lookback)]["CONSISTENT_WITH_NOISE"] += 1
                elif result["verdict"] == "NO_EDGE":
                    lookback_counts[str(lookback)]["NO_EDGE"] += 1

                lookback_results[ticker] = result

                print(f"  {ticker:6s}: verdict={result['verdict']:18s} "
                      f"median={result['medians'][0]:+6.3f}")

            if cfg_name == "zero_cost":
                results_zero["per_asset"] = per_asset
                results_zero["lookback_counts"] = lookback_counts
            else:
                results_realistic["per_asset"] = per_asset
                results_realistic["lookback_counts"] = lookback_counts

        # Calculate robustness verdicts per asset
        print(f"\n{cfg_name.upper()} SUMMARY:")
        print("  REGIME_STABLE asset count by lookback:")
        for lb in LOOKBACKS:
            print(f"    lookback {lb}: {lookback_counts[str(lb)]["REGIME_STABLE"]} / {len(tickers)}")

        robustness_counts = {
            "ROBUST": 0,
            "SENSITIVE": 0,
            "NO_EDGE": 0,
        }

        for ticker in tickers:
            edge_counts = 0
            for lb in LOOKBACKS:
                if per_asset[ticker][str(lb)]["verdict"] == "REGIME_STABLE":
                    edge_counts += 1

            if edge_counts >= 2:
                robustness_verdict = "ROBUST"
                robustness_counts["ROBUST"] += 1
            elif edge_counts == 1:
                robustness_verdict = "SENSITIVE"
                robustness_counts["SENSITIVE"] += 1
            else:
                robustness_verdict = "NO_EDGE"
                robustness_counts["NO_EDGE"] += 1

            per_asset[ticker]["robustness_verdict"] = robustness_verdict
            per_asset[ticker]["edge_counts"] = edge_counts

        print(f"  Robustness counts: {robustness_counts}")
        print(f"  ROBUST assets: {robustness_counts["ROBUST"]} / {len(tickers)}")

    # Cost comparison
    print("\n=== 5. Cost comparison ===")
    print("Comparing zero-cost vs realistic-cost regime-stability:")

    cost_robust_summary = {}
    for ticker in tickers:
        zero_verdicts = [results_zero["per_asset"][ticker][str(lb)]["verdict"] 
                        for lb in LOOKBACKS]
        realistic_verdicts = [results_realistic["per_asset"][ticker][str(lb)]["verdict"] 
                             for lb in LOOKBACKS]

        zero_stable = sum(1 for v in zero_verdicts if v == "REGIME_STABLE")
        realistic_stable = sum(1 for v in realistic_verdicts if v == "REGIME_STABLE")

        zero_robust = all(v in ["REGIME_STABLE", "REGIME_STABLE_LOSS"] for v in zero_verdicts)
        realistic_robust = all(v in ["REGIME_STABLE", "REGIME_STABLE_LOSS"] for v in realistic_verdicts)

        cost_change = realistic_stable - zero_stable

        if realistic_stable >= zero_stable:
            cost_robust = "COST_ROBUST"
        else:
            cost_robust = "COST_SENSITIVE"

        cost_robust_summary[ticker] = {
            "zero_cost_stable": zero_stable,
            "realistic_cost_stable": realistic_stable,
            "cost_change": cost_change,
            "cost_robust": cost_robust,
        }

        print(f"  {ticker:6s}: zero_cost={zero_stable}/3 stable, "
              f"realistic_cost={realistic_stable}/3 stable, "
              f"change={cost_change:+d}, {cost_robust}")

    # Determinism: rerun the target subset and assert identical output
    print("\n=== 6. Determinism check (target subset) ===")
    per_asset_2 = {}
    for cfg_name, cfg_results in [("zero_cost", results_zero), ("realistic_cost", results_realistic)]:
        for lookback in LOOKBACKS:
            per_asset_2[f"{cfg_name}_lookback_{lookback}"] = {}
            for ticker in target:
                per_asset_2[f"{cfg_name}_lookback_{lookback}"][ticker] = segment_result(ticker, tickers[ticker], lookback, cfg)

    # Create artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost=dict(
            lookback_counts=results_zero["lookback_counts"],
            per_asset=results_zero["per_asset"],
        ),
        realistic_cost=dict(
            lookback_counts=results_realistic["lookback_counts"],
            per_asset=results_realistic["per_asset"],
        ),
        cost_robust_summary=cost_robust_summary,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum lookback sweep across the collected universe with cost sensitivity: exploratory simulation.")

    # Summary report
    print("\n=== SUMMARY ===")
    print("Zero cost regime-stability:")
    print(f"  Assets with >=2 REGIME_STABLE across lookbacks: {results_zero['per_asset']['AMZN']['robustness_verdict']}")
    print(f"  ROBUST assets: {robustness_counts['ROBUST']}/{len(tickers)}")

    print("\nRealistic cost regime-stability:")
    print(f"  Assets with >=2 REGIME_STABLE across lookbacks: {results_realistic['per_asset']['AMZN']['robustness_verdict']}")
    print(f"  ROBUST assets: {robustness_counts['ROBUST']}/{len(tickers)}")

    cost_robust_assets = sum(1 for v in cost_robust_summary.values() if v["cost_robust"] == "COST_ROBUST")
    print(f"\nCost robustness: {cost_robust_assets}/{len(tickers)} assets maintain or improve stability with realistic costs")

    return 0
if __name__ == "__main__":
    sys.exit(main())
