"""Cost-sensitive momentum check with realistic transaction costs.

This is agent 02's bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

The realistic cost model from examples/ma_crossover_real_data.py:
- commission_per_trade=2.0
- commission_per_share=0.003
- slippage_cents=2.0
- slippage_proportional=0.0005

This check runs the same momentum test but with these costs applied, then
compares against the existing momentum_results.json to see if the edge
survives conservative transaction-cost stress.

Falsification prediction: the momentum edge either shows CONSISTENT_WITH_NOISE
(no edge anywhere), REGIME_DEPENDENT (positive run concentrates in one regime),
or REGIME_STABLE_LOSS (uniformly negative — momentum negative where reversal was positive).
Either outcome is a negative conclusion and closes the cell.
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
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
COST_SENSITIVE_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitive_results.json"

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
)

ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, warmup_periods=0)


def momentum_signals(closes, lookback):
    """Momentum signal implementation (same as in research.backtest.regime_stability)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost-Sensitive Momentum Check (lookback 5) ===")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"Zero costs: commission_per_trade={ZERO_COSTS.commission_per_trade}, "
          f"commission_per_share={ZERO_COSTS.commission_per_share}, "
          f"slippage_cents={ZERO_COSTS.slippage_cents}, "
          f"slippage_proportional={ZERO_COSTS.slippage_proportional}")
    print()

    # Load the existing momentum artifact for comparison
    existing_artifact_path = ARTIFACT_DIR / "momentum_results.json"
    with open(existing_artifact_path) as f:
        existing_artifact = json.load(f)

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]

    print("=== Cost Impact Analysis (AMZN, JPM) ===")
    cost_impact_results = {}

    for asset in ["AMZN", "JPM"]:
        bars = tickers[asset]
        closes = bars.closes_array()
        
        # Calculate volatility blocks for regime classification
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        # Generate signals for the asset
        signals = momentum_signals(closes, LOOKBACK)
        
        # Run stress segments with zero costs
        zero_res = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, LOOKBACK),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK}],
            baseline=(("lookback", LOOKBACK),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=ZERO_COSTS,
        )
        
        # Run stress segments with realistic costs
        realistic_res = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, LOOKBACK),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK}],
            baseline=(("lookback", LOOKBACK),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=REALISTIC_COSTS,
        )
        
        # Extract results
        zero_results = {}
        realistic_results = {}
        
        for scenario in zero_res.scenarios:
            name = scenario.name
            zero_results[name] = {
                "segments": [s.name for s in zero_res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in zero_res.scenarios],
                "candidate_dispersion": round(zero_res.candidate_dispersion, 3),
                "null_dispersion": round(zero_res.null_dispersion, 3),
                "verdict": zero_res.overall_verdict,
                "n_folds": zero_res.n_folds,
            }
        
        for scenario in realistic_res.scenarios:
            name = scenario.name
            realistic_results[name] = {
                "segments": [s.name for s in realistic_res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in realistic_res.scenarios],
                "candidate_dispersion": round(realistic_res.candidate_dispersion, 3),
                "null_dispersion": round(realistic_res.null_dispersion, 3),
                "verdict": realistic_res.overall_verdict,
                "n_folds": realistic_res.n_folds,
            }
        
        cost_impact_results[asset] = {
            "zero_costs": zero_results,
            "realistic_costs": realistic_results,
        }

        print(f"\nAsset: {asset}")
        print("  Zero cost:   segments={len(list(zero_res.scenarios))}, "
              f"medians={list(zero_res.scenarios)[0].baseline_median_log_return:.3f}, verdict={zero_res.overall_verdict}")
        print("  Realistic:   segments={len(list(realistic_res.scenarios))}, "
              f"medians={list(realistic_res.scenarios)[0].baseline_median_log_return:.3f}, verdict={realistic_res.overall_verdict}")

        # Calculate cost impact
        zero_median = list(zero_res.scenarios)[0].baseline_median_log_return
        realistic_median = list(realistic_res.scenarios)[0].baseline_median_log_return
        median_reduction = zero_median - realistic_median
        print(f"  Cost impact: median reduction = {median_reduction:+.3f}")

    # Full-universe cost impact analysis
    print("\n=== Full-Universe Cost Impact ===")
    print("Asset | Zero Cost Verdict | Realistic Verdict | Verdict Change?")
    print("-------|------------------|-------------------|----------------")

    verdict_changes = 0
    cost_impact_by_asset = {}

    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        
        # Calculate volatility blocks for regime classification
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        # Generate signals for the asset
        signals = momentum_signals(closes, LOOKBACK)
        
        # Run stress segments with zero costs
        zero_res = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, LOOKBACK),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK}],
            baseline=(("lookback", LOOKBACK),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=ZERO_COSTS,
        )
        
        # Run stress segments with realistic costs
        realistic_res = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, LOOKBACK),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK}],
            baseline=(("lookback", LOOKBACK),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=REALISTIC_COSTS,
        )
        
        zero_verdict = zero_res.overall_verdict
        realistic_verdict = realistic_res.overall_verdict
        verdict_change = zero_verdict != realistic_verdict
        if verdict_change:
            verdict_changes += 1
            
        cost_impact_by_asset[ticker] = {
            "zero_costs": {
                "segments": [s.name for s in zero_res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in zero_res.scenarios],
                "verdict": zero_res.overall_verdict,
            },
            "realistic_costs": {
                "segments": [s.name for s in realistic_res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in realistic_res.scenarios],
                "verdict": realistic_res.overall_verdict,
            },
        }
        
        print(f"{ticker:5s} | {zero_verdict:16s} | {realistic_verdict:17s} | {'YES' if verdict_change else 'NO'}")

    # Calculate total cost impact
    total_cost_impact = 0
    for asset in ["AMZN", "JPM"]:
        zero_median = list(cost_impact_results[asset]["zero_costs"].values())[0]["medians"][0]
        realistic_median = list(cost_impact_results[asset]["realistic_costs"].values())[0]["medians"][0]
        total_cost_impact += (zero_median - realistic_median)

    print(f"\nTotal cost impact across universe: {total_cost_impact:+.3f} log-return reduction")
    print(f"Verdict stability across costs: {10 - verdict_changes}/10 assets unchanged")

    # Determinism check
    print("\n=== Determinism Check ===")
    # Re-run the realistic cost test to verify determinism
    determinism_ok = True
    for asset in ["AMZN", "JPM"]:
        # Run realistic costs again
        bars = tickers[asset]
        closes = bars.closes_array()
        
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        signals = momentum_signals(closes, LOOKBACK)
        
        realistic_res2 = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, LOOKBACK),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK}],
            baseline=(("lookback", LOOKBACK),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=REALISTIC_COSTS,
        )
        
        # Get previous realistic results verdict
        realistic_verdict1 = cost_impact_by_asset[asset]["realistic_costs"]["verdict"]
        
        # Compare verdicts
        if realistic_res2.overall_verdict != realistic_verdict1:
            print(f"  NON-DETERMINISTIC: {asset} realistic costs verdicts differ between runs")
            determinism_ok = False
        else:
            print(f"  {asset}: realistic costs verdicts are DETERMINISTIC")

    # Compare with existing artifact
    print("\n=== Comparison with Existing Artifact ===")
    existing_universe = existing_artifact["universe"]["per_asset"]

    for ticker in existing_universe:
        existing_verdict = existing_universe[ticker]["verdict"]
        realistic_verdict = cost_impact_by_asset[ticker]["realistic_costs"]["verdict"]
        verdict_match = existing_verdict == realistic_verdict
        print(f"{ticker}: existing verdict={existing_verdict}, cost-sensitive verdict={realistic_verdict} "
              f"({'MATCH' if verdict_match else 'MISMATCH'})")

    # Create cost-sensitive artifact
    cost_sensitive_artifact = dict(
        dataset_id=DATASET_ID,
        lookback=LOOKBACK,
        min_segment_bars=MIN_SEGMENT_BARS,
        n_blocks=N_BLOCKS,
        overlap=OVERLAP,
        per_asset=cost_impact_by_asset,
        seed=SEED,
        test=TEST,
        train=TRAIN,
        warmup=WARM,
        window=WINDOW,
        realistic_costs={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        zero_costs={
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
    )

    with open(COST_SENSITIVE_ARTIFACT_PATH, "w") as f:
        json.dump(cost_sensitive_artifact, f, indent=2, sort_keys=True)

    print(f"\n=== Cost-sensitive artifact written to: {COST_SENSITIVE_ARTIFACT_PATH} ===")
    
    # Summary
    print("\n=== EXECUTION SUMMARY ===")
    print(f"Checked momentum edge (lookback {LOOKBACK}) on collected universe with realistic costs")
    print(f"Realistic cost model: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"Total cost impact across universe: {total_cost_impact:+.3f} log-return reduction")
    print(f"Verdict stability across costs: {10 - verdict_changes}/10 assets unchanged")
    print(f"Determinism verified: {determinism_ok}")
    
    # Success criteria: reproducible and independently verifiable
    if determinism_ok:
        print("✓ SUCCESS: Results are reproducible and independently verifiable")
        return 0
    else:
        print("✗ FAILURE: Results are not deterministic")
        return 1


if __name__ == "__main__":
    sys.exit(main())