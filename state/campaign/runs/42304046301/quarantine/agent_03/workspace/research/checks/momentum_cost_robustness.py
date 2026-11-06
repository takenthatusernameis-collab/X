"""Bounded cost-sensitivity check for lookback 3/5/10 momentum.

This implements the contract's requirement: "Run a bounded cost-sensitivity check 
for lookback 3/5/10 momentum under the repository's existing realistic cost model 
and matched null."

The objective is to evaluate whether the short-horizon momentum edge survives 
conservative transaction-cost stress on the collected real-data universe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import research.backtest as bt
from research.backtest.regime_stability import (
    momentum_signals,
    volatility_blocks,
    segment_fn_from_labels,
    OUTLIER_TOL,
)
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


def create_cost_model():
    """Create a realistic cost model matching the repository's existing cost model."""
    return bt.BacktestConfig(
        initial_capital=1_000_000,
        target_exposure=1.0,
        commission_per_trade=0.001,  # $0.001 per trade
        commission_per_share=0.0001,  # $0.0001 per share
        slippage_cents=0.5,  # $0.50 fixed slippage per share
        slippage_proportional=0.0002,  # 2 bps proportional slippage
        warmup_periods=0,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )


def segment_result(
    ticker: str,
    bars: bt.BarSequence,
    lookback: int,
    cfg: bt.BacktestConfig,
) -> Dict:
    """Walk-forward cost-robustness result for one ticker and one lookback."""
    closes = bars.closes_array()
    labels = volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = segment_fn_from_labels(labels)
    signals = momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")

    # Create parameter grid for the lookback (single point)
    param_grid = [{"lookback": lookback}]
    baseline = (("lookback", lookback),)

    # Run stress segments with the cost model
    res = bt.stress_segments(
        signals_fn=lambda c, **p: momentum_signals(c, lookback=p["lookback"]),
        bars=bars,
        signals=signals,
        param_grid=param_grid,
        baseline=baseline,
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )

    # Extract cost robustness information
    scenario = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        baseline_median_log_return=round(scenario.baseline_median_log_return, 3),
        noise_median_log_return=round(scenario.noise_median_log_return, 3),
    )


def robustness_verdict(
    lookback_results: Dict[str, Dict],
) -> tuple[str, Dict[str, int]]:
    """Verdict on whether the edge is cost-robust for one asset across lookbacks."""
    # Edge is defined as baseline_median_log_return > 0 and statistically significant
    # (outside the null distribution by a meaningful margin)
    lookback_edges = {}
    for lb in LOOKBACKS:
        result = lookback_results[str(lb)]
        edge = (
            result["verdict"] == "REGIME_STABLE"
            and result["baseline_median_log_return"] > 0
            and abs(result["baseline_median_log_return"] - result["noise_median_log_return"]) > 0.05
        )
        lookback_edges[str(lb)] = "EDGE" if edge else "NO_EDGE"

    # Cost-robustness verdict
    edge_count = sum(1 for lb in LOOKBACKS if lookback_edges[str(lb)] == "EDGE")
    if edge_count >= 2:
        return "COST_ROBUST", {"EDGE": edge_count, "NO_EDGE": 3 - edge_count}
    elif edge_count == 1:
        return "COST_SENSITIVE", {"EDGE": edge_count, "NO_EDGE": 3 - edge_count}
    else:
        return "NO_COST_EDGE", {"EDGE": edge_count, "NO_EDGE": 3 - edge_count}


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Bounded Cost-Sensitivity Check for Momentum Lookback 3/5/10 ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Cost model: commission=$0.001/trade + $0.0001/share, slippage=$0.50/share + 2bps")
    print(f"Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print()

    # Create cost model
    cfg = create_cost_model()
    print("Cost model parameters:")
    print(f"  commission_per_trade: ${cfg.commission_per_trade}")
    print(f"  commission_per_share: ${cfg.commission_per_share}")
    print(f"  slippage_cents: ${cfg.slippage_cents/100:.3f}")
    print(f"  slippage_proportional: {cfg.slippage_proportional:.4f} ({cfg.slippage_proportional*10000:.1f} bps)")
    print()

    # Load manifest and verify
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']} vs {DATASET_ID}"

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    # Load the full collected universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print(f"Loaded {len(ticker_order)} tickers: {', '.join(ticker_order)}")
    print()

    # Run cost-robustness check for each lookback
    print("=== 2. Momentum lookback cost-robustness check ===")
    print("Running walk-forward regime-stability stress tests for lookback 3, 5, 10 under cost model")
    print()

    cost_robustness_results = {}
    per_asset_summary = {}

    for ticker in ticker_order:
        print(f"Processing {ticker}...")
        bars = all_tickers[ticker]

        # Cost-robustness results for each lookback
        lookback_results = {}
        for lb in LOOKBACKS:
            result = segment_result(ticker, bars, lb, cfg)
            lookback_results[str(lb)] = result

        # Store for artifact
        cost_robustness_results[ticker] = lookback_results

        # Summary verdict for this asset
        verdict, edge_counts = robustness_verdict(lookback_results)
        per_asset_summary[ticker] = {
            "lookback_results": lookback_results,
            "robustness_verdict": verdict,
            "edge_counts": edge_counts,
        }

        # Print per-asset results
        print(f"  Lookback 3: {lookback_results['3']['verdict']} "
              f"(median: {lookback_results['3']['baseline_median_log_return']:+.3f}, "
              f"noise: {lookback_results['3']['noise_median_log_return']:+.3f})")
        print(f"  Lookback 5: {lookback_results['5']['verdict']} "
              f"(median: {lookback_results['5']['baseline_median_log_return']:+.3f}, "
              f"noise: {lookback_results['5']['noise_median_log_return']:+.3f})")
        print(f"  Lookback 10: {lookback_results['10']['verdict']} "
              f"(median: {lookback_results['10']['baseline_median_log_return']:+.3f}, "
              f"noise: {lookback_results['10']['noise_median_log_return']:+.3f})")
        print(f"  Cost-robustness verdict: {verdict}")
        print()

    # Universe-level cost-robustness summary
    print("=== 3. Universe-level cost-robustness summary ===")
    universe_robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "NO_COST_EDGE": 0}
    lookback_5_cost_robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "NO_COST_EDGE": 0}

    for ticker in ticker_order:
        verdict = per_asset_summary[ticker]["robustness_verdict"]
        universe_robustness_counts[verdict] += 1
        if "5" in per_asset_summary[ticker]["lookback_results"]:
            lb5_verdict = per_asset_summary[ticker]["lookback_results"]["5"]["verdict"]
            if lb5_verdict == "REGIME_STABLE" and per_asset_summary[ticker]["lookback_results"]["5"]["baseline_median_log_return"] > 0:
                lookback_5_cost_robustness_counts["COST_ROBUST"] += 1
            elif lb5_verdict == "REGIME_STABLE_LOSS" and per_asset_summary[ticker]["lookback_results"]["5"]["baseline_median_log_return"] < 0:
                lookback_5_cost_robustness_counts["NO_COST_EDGE"] += 1
            else:
                lookback_5_cost_robustness_counts["COST_SENSITIVE"] += 1

    print("Universe cost-robustness verdict counts:")
    for verdict, count in universe_robustness_counts.items():
        print(f"  {verdict}: {count}")
    print()

    print("Lookback 5 (reference) cost-robustness verdict counts:")
    for verdict, count in lookback_5_cost_robustness_counts.items():
        print(f"  {verdict}: {count}")
    print()

    print("Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset_summary[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["lookback_results"][str(lb)]["baseline_median_log_return"], p["lookback_results"][str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print(f"  {ticker}: segments={p['lookback_results']['5']['segments']:<8} medians3510={med:<40s} cost_robustness={p['robustness_verdict']:<12s} edge_counts={p['edge_counts']}")

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset_summary,
        universe=dict(
            lookback_5_cost_robustness_counts=lookback_5_cost_robustness_counts,
            cost_robustness_counts=universe_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={lb: per_asset_summary[t]["lookback_results"][lb]["baseline_median_log_return"]
                                     for lb in LOOKBACKS},
                    lookback_null_medians={lb: per_asset_summary[t]["lookback_results"][lb]["noise_median_log_return"]
                                         for lb in LOOKBACKS},
                    lookback_verdicts_5={lb: per_asset_summary[t]["lookback_results"][lb]["verdict"]
                                       for lb in LOOKBACKS},
                    edge_counts=per_asset_summary[t]["edge_counts"],
                    cost_robustness_verdict=per_asset_summary[t]["robustness_verdict"],
                    n_folds_5=per_asset_summary[t]["lookback_results"]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
        cost_model=dict(
            commission_per_trade=cfg.commission_per_trade,
            commission_per_share=cfg.commission_per_share,
            slippage_cents=cfg.slippage_cents,
            slippage_proportional=cfg.slippage_proportional,
            initial_capital=cfg.initial_capital,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Bounded cost-sensitivity check for momentum lookback 3/5/10: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())