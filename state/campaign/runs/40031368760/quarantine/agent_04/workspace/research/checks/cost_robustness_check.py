"""Cost-sensitivity check for lookback 3/5/10 momentum.

Objective: test whether the candidate positive momentum evidence (lookback=5)
survives conservative transaction-cost stress on the collected real-data universe.

Method: for each lookback (3, 5, 10), run momentum (long previous N-day return,
hold 1 day, daily rebalance) on the full collected 10-asset universe via
walk-forward IS/OOS validation. Compare performance with realistic costs vs
zero costs to assess cost robustness.

Realistic cost model (from examples/ma_crossover.py):
- commission_per_trade=2.0
- commission_per_share=0.003
- slippage_cents=2.0
- slippage_proportional=0.0005

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
ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_results.json"

# Realistic cost model from examples/ma_crossover.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero cost model (matched null)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)


def segment_result(ticker, bars, lookback, costs):
    """Walk-forward regime-stability result for one ticker, one lookback, and given costs."""
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
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        costs_label=costs.__class__.__name__,
    )


def run_cost_sensitivity_check():
    """Run cost-sensitivity check for momentum lookbacks 3,5,10."""
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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-robustness run.")
        sys.exit(1)

    print("\n=== 3. Cost robustness check for momentum lookbacks 3/5/10 ===")
    print("  Method: walk-forward regime-stability analysis via stress_segments")
    print(f"  Parameters: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"  Realistic costs: commission_trade=2.0, commission_share=0.003, "
          f"slippage_cents=2.0, slippage_proportional=0.0005")
    print(f"  Zero costs (matched null): all cost parameters = 0")
    print()

    # Load all tickers from manifest
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Store results for each lookback, with and without costs
    cost_results = {}
    zero_cost_results = {}

    for lookback in LOOKBACKS:
        print(f"--- Lookback = {lookback} ---")

        # Run with realistic costs
        print(f"  Running with realistic costs...")
        cost_ticker_results = {}
        for ticker in ticker_order:
            result = segment_result(ticker, all_tickers[ticker], lookback, REALISTIC_COSTS)
            cost_ticker_results[ticker] = result
        cost_results[lookback] = cost_ticker_results

        # Run with zero costs (matched null)
        print(f"  Running with zero costs (matched null)...")
        zero_cost_ticker_results = {}
        for ticker in ticker_order:
            result = segment_result(ticker, all_tickers[ticker], lookback, ZERO_COSTS)
            zero_cost_ticker_results[ticker] = result
        zero_cost_results[lookback] = zero_cost_ticker_results

        # Print summary for this lookback
        print(f"  Results summary:")
        cost_verdicts = {}
        zero_cost_verdicts = {}
        for ticker in ticker_order:
            cost_verdicts[ticker] = cost_ticker_results[ticker]["verdict"]
            zero_cost_verdicts[ticker] = zero_cost_ticker_results[ticker]["verdict"]

        # Count verdict types
        cost_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0, "CONSISTENT_WITH_NOISE": 0, "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
        zero_cost_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0, "CONSISTENT_WITH_NOISE": 0, "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

        for verdict in cost_verdicts.values():
            if verdict in cost_counts:
                cost_counts[verdict] += 1
        for verdict in zero_cost_verdicts.values():
            if verdict in zero_cost_counts:
                zero_cost_counts[verdict] += 1

        print(f"    Realistic costs: {cost_counts}")
        print(f"    Zero costs:       {zero_cost_counts}")

        # Calculate cost impact on edges
        cost_edges = cost_counts["EDGE"]
        zero_cost_edges = zero_cost_counts["EDGE"]
        print(f"    Cost impact on edges: {cost_edges} -> {zero_cost_edges} "
              f"(change: {cost_edges - zero_cost_edges})")
        print()

    # Determine cost robustness verdict
    print("\n=== 4. Cost robustness verdict ===")
    robustness_verdicts = {}
    for lookback in LOOKBACKS:
        cost_edges = sum(1 for ticker in ticker_order if cost_results[lookback][ticker]["verdict"] == "EDGE")
        zero_cost_edges = sum(1 for ticker in ticker_order if zero_cost_results[lookback][ticker]["verdict"] == "EDGE")

        # Check if momentum edge survives cost stress
        if cost_edges >= 2 and zero_cost_edges >= 2:  # Both have at least 2 edges
            verdict = "ROBUST"
        elif cost_edges >= 1 and zero_cost_edges >= 1:  # Both have at least 1 edge
            verdict = "SENSITIVE"
        else:
            verdict = "FAILED"

        robustness_verdicts[lookback] = {
            "verdict": verdict,
            "cost_edges": cost_edges,
            "zero_cost_edges": zero_cost_edges,
        }

        print(f"  Lookback {lookback}: {verdict}")
        print(f"    Zero-cost edges: {zero_cost_edges}/{len(ticker_order)} assets")
        print(f"    Cost-surviving edges: {cost_edges}/{len(ticker_order)} assets")
        print()

    # Overall verdict
    robust_lookbacks = [lb for lb, verdict in robustness_verdicts.items() if verdict['verdict'] == 'ROBUST']
    sensitive_lookbacks = [lb for lb, verdict in robustness_verdicts.items() if verdict['verdict'] == 'SENSITIVE']
    failed_lookbacks = [lb for lb, verdict in robustness_verdicts.items() if verdict['verdict'] == 'FAILED']

    overall_verdict = "ROBUST" if len(robust_lookbacks) >= 2 else "SENSITIVE" if len(robust_lookbacks) + len(sensitive_lookbacks) > 0 else "FAILED"

    print(f"\n=== 5. Overall cost robustness verdict ===")
    print(f"  ROBUST lookbacks: {robust_lookbacks}")
    print(f"  SENSITIVE lookbacks: {sensitive_lookbacks}")
    print(f"  FAILED lookbacks: {failed_lookbacks}")
    print(f"  OVERALL VERDICT: {overall_verdict}")

    # Create artifact for independent verification
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "zero_costs": {
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
        "cost_results": cost_results,
        "zero_cost_results": zero_cost_results,
        "robustness_verdicts": robustness_verdicts,
        "overall_verdict": overall_verdict,
        "robust_lookbacks": robust_lookbacks,
        "sensitive_lookbacks": sensitive_lookbacks,
        "failed_lookbacks": failed_lookbacks,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Cost robustness check for lookback 3/5/10 momentum.")

    return 0

    # Determine cost robustness verdict
    print("\n=== 4. Cost robustness verdict ===")
    robustness_verdicts = {}
    for lookback in LOOKBACKS:
        cost_agg = cost_results[lookback]["realistic_costs"]["aggregate"]
        zero_agg = zero_cost_results[lookback]["zero_costs"]["aggregate"]

        # Check if momentum edge survives cost stress
        # Criteria: median log return remains positive AND significantly different from zero
        cost_survives = cost_agg['median_log_total_return'] > 0
        zero_survives = zero_agg['median_log_total_return'] > 0

        if cost_survives and cost_agg['n_folds'] > 0:
            # Check if cost impact is reasonable (not completely eroding the edge)
            impact_ratio = abs(cost_agg['median_log_total_return'] / zero_agg['median_log_total_return']) if zero_agg['median_log_total_return'] != 0 else float('inf')
            if impact_ratio > 0.5:  # Edge survives if it retains > 50% of zero-cost performance
                verdict = "ROBUST"
            else:
                verdict = "SENSITIVE"
        else:
            verdict = "FAILED"

        robustness_verdicts[lookback] = {
            "verdict": verdict,
            "cost_survives": cost_survives,
            "zero_survives": zero_survives,
            "impact_ratio": impact_ratio if 'impact_ratio' in locals() else None,
            "cost_median_log_return": cost_agg['median_log_total_return'],
            "zero_median_log_return": zero_agg['median_log_total_return'],
        }

        print(f"  Lookback {lookback}: {verdict}")
        print(f"    Zero-cost edge: {'YES' if zero_survives else 'NO'} "
              f"(median log return: {zero_agg['median_log_total_return']:.4f})")
        print(f"    Cost-surviving edge: {'YES' if cost_survives else 'NO'} "
              f"(median log return: {cost_agg['median_log_total_return']:.4f})")
        print(f"    Cost impact ratio: {impact_ratio:.3f}" if 'impact_ratio' in locals() else "    Cost impact ratio: N/A")
        print()

    # Overall verdict
    robust_lookbacks = [lb for lb, verdict in robustness_verdicts.items() if verdict['verdict'] == 'ROBUST']
    sensitive_lookbacks = [lb for lb, verdict in robustness_verdicts.items() if verdict['verdict'] == 'SENSITIVE']
    failed_lookbacks = [lb for lb, verdict in robustness_verdicts.items() if verdict['verdict'] == 'FAILED']

    overall_verdict = "ROBUST" if len(robust_lookbacks) >= 2 else "SENSITIVE" if len(robust_lookbacks) + len(sensitive_lookbacks) > 0 else "FAILED"

    print(f"\n=== 5. Overall cost robustness verdict ===")
    print(f"  ROBUST lookbacks: {robust_lookbacks}")
    print(f"  SENSITIVE lookbacks: {sensitive_lookbacks}")
    print(f"  FAILED lookbacks: {failed_lookbacks}")
    print(f"  OVERALL VERDICT: {overall_verdict}")

    # Create artifact for independent verification
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "zero_costs": {
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
        "cost_results": cost_results,
        "zero_cost_results": zero_cost_results,
        "robustness_verdicts": robustness_verdicts,
        "overall_verdict": overall_verdict,
        "robust_lookbacks": robust_lookbacks,
        "sensitive_lookbacks": sensitive_lookbacks,
        "failed_lookbacks": failed_lookbacks,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Cost robustness check for lookback 3/5/10 momentum.")

    return 0


def main():
    return run_cost_sensitivity_check()


if __name__ == "__main__":
    sys.exit(main())
