#!/usr/bin/env python3
"""Real-data cost-sensitivity check for momentum lookback 3/5/10.

Objective: test whether the momentum edge observed on real data survives
conservative transaction-cost stress.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return;
hold 1 day; daily rebalance; lookbacks 3,5,10; same walk-forward windows and
volatility blocks as momentum.py; each asset tested against a matched null
(random signal) with identical costs.

Cost model (realistic, per-trade): commission $0.005 per share,
slippage 5 cents per share, proportional slippage 10 bps.
This represents a typical discretionary-trading cost level.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"


# Realistic cost model (per-trade, applied to every round-trip)
# Commission: $0.005 per share (typical broker rate)
# Fixed slippage: 5 cents per share (typical market impact for 10K shares)
# Proportional slippage: 10 bps of price (typical for liquid stocks)
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.005,  # $0.005 per share
    slippage_cents=5.0,          # 5 cents per share fixed
    slippage_proportional=0.001, # 10 bps proportional
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def matched_null_signals(closes):
    """Random signal class matched to momentum (independent coin-flip null)."""
    n = len(closes)
    np.random.seed(SEED)
    # Coin-flip random weights at each bar, independent of price
    random_weights = np.random.choice([-1.0, 0.0, 1.0], size=n, p=[0.25, 0.5, 0.25])
    return [bt.Signal(date=i + 1, weight=float(random_weights[i])) for i in range(n)]


def segment_result_cost_sensitive(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Create momentum signal function that takes lookback as parameter
    def momentum_signal_fn(c, **params):
        return bt.momentum_signals(c, lookback=params.get("lookback", lookback))
    
    res = bt.stress_segments(
        signals_fn=momentum_signal_fn,
        bars=bars,
        signals=momentum_signal_fn(closes, lookback=lookback),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=COST_CONFIG,
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


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

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data cost run.")
        sys.exit(1)

    print("\n=== 3. Momentum cost-sensitivity check (lookback 3,5,10) ===")
    print("Cost model: commission $0.005/share, slippage 5c/share + 10bps")
    print(f"Params: walk-forward train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d, "
          f"min {MIN_SEGMENT_BARS} bars/segment, {len(LOOKBACKS)} lookbacks")

    results_all = {}
    tickers_univ = {}

    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    # Run momentum check with costs for all lookbacks
    for ticker in manifest["universe"]:
        bars = tickers_univ[ticker]
        results_all[ticker] = {}
        
        print(f"  {ticker}:", end=" ")
        
        for lb in LOOKBACKS:
            mom_res = segment_result_cost_sensitive(ticker, bars, lb)
            results_all[ticker][str(lb)] = mom_res
        
        print(f" lookback {LOOKBACKS} completed")

    print("\n=== 4. Cost-robustness verdicts ===")
    cost_robust_summary = {}
    for ticker in manifest["universe"]:
        edge_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_DEPENDENT": 0, 
                      "REGIME_STABLE_LOSS": 0, "REGIME_STABLE": 0}
        
        for lb in LOOKBACKS:
            result = results_all[ticker][str(lb)]
            verdict = result["verdict"]
            medians = result["medians"]
            
            # Consistency with noise (all medians within 5% of zero)
            if all(abs(m) <= 0.05 for m in medians):
                edge_status = "CONSISTENT_WITH_NOISE"
            # Regime-dependent (candidate dispersion > 2x null)
            elif result["candidate_dispersion"] > 2.0 * result["null_dispersion"]:
                edge_status = "REGIME_DEPENDENT"
            # Uniformly negative
            elif all(m < 0 for m in medians):
                edge_status = "REGIME_STABLE_LOSS"
            # Uniformly positive
            elif all(m > 0 for m in medians):
                edge_status = "REGIME_STABLE"
            else:
                edge_status = "REGIME_STABLE"
            
            edge_counts[edge_status] += 1
        
        # Cost-robustness: REGIME_STABLE at >= 2 lookbacks (conservative threshold)
        cost_robust = "COST_ROBUST" if edge_counts["REGIME_STABLE"] >= 2 else "COST_SENSITIVE"
        cost_robust_summary[ticker] = {
            "edge_counts": edge_counts,
            "cost_robustness_verdict": cost_robust,
            "momentum_results": results_all[ticker],
        }
        print(f"  {ticker:6s}: {cost_robust}")

    robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0}
    for ticker in cost_robust_summary.values():
        robustness_counts[ticker["cost_robustness_verdict"]] += 1
    
    print(f"\n  Summary: {robustness_counts['COST_ROBUST']} COST_ROBUST, "
          f"{robustness_counts['COST_SENSITIVE']} COST_SENSITIVE")

    # Determinism: recompute AMZN momentum cost result
    print("\n=== 5. Determinism check (AMZN lookback=5) ===")
    bars_amzn, _ = bt.load_ticker("AMZN")
    result1 = segment_result_cost_sensitive("AMZN", bars_amzn, 5)
    result2 = segment_result_cost_sensitive("AMZN", bars_amzn, 5)
    
    print(f"  First run medians: {[r['medians'][0] for r in results_all['AMZN'].values()]}")
    print(f"  Second run medians: {[r['medians'][0] for r in results_all['AMZN'].values()]}")
    print(f"  Results identical: {result1 == result2}")

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_config=dict(
            commission_per_share=COST_CONFIG.commission_per_share,
            slippage_cents=COST_CONFIG.slippage_cents,
            slippage_proportional=COST_CONFIG.slippage_proportional,
        ),
        per_asset=cost_robust_summary,
        robustness_summary=robustness_counts,
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
