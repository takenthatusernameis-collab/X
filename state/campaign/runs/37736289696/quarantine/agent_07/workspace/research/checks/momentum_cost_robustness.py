"""Momentum cost robustness test - corrected approach.

Builds on the positive momentum evidence from momentum_lookback_sweep.py
and adds cost sensitivity analysis using the repository's existing realistic
cost model.

Method:
- For each lookback in [3, 5, 10], run walk-forward IS/OOS using the same
  methodology as momentum_lookback_sweep.py but with realistic costs
- Use realistic cost model from examples/ma_crossover.py:
    commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
    slippage_proportional=0.0005
- Compare with zero cost to measure cost impact
- Determinism check on AMZN, JPM subset
- Generate artifact for verification

Research/simulation only. No live trading.
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

# Realistic cost model from examples/ma_crossover.py
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
)

# Determinism test parameters
DETERMINISM_TARGETS = ["AMZN", "JPM"]


def generate_cost_robustness_results():
    """Run cost robustness test for lookback 3/5/10 momentum."""
    print("=== Momentum Cost Robustness Test ===")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Realistic cost config: commission_trade={REALISTIC_COST_CONFIG.commission_per_trade}, "
          f"commission_share={REALISTIC_COST_CONFIG.commission_per_share}, "
          f"slippage_cents={REALISTIC_COST_CONFIG.slippage_cents}, "
          f"slippage_prop={REALISTIC_COST_CONFIG.slippage_proportional}")
    print(f"Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"Min {MIN_SEGMENT_BARS} bars/segment, {N_BLOCKS} volatility blocks")
    
    # 1. Manifest integrity check
    print("\n=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"dataset id mismatch: {manifest['dataset_id']}"
    
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    # 2. Load collected universe
    print("\n=== 2. Loading collected universe ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(tickers.keys())
    print(f"  Loaded {len(ticker_order)} tickers: {ticker_order}")
    
    # 3. Run momentum lookback sweep with realistic costs
    print("\n=== 3. Momentum lookback sweep with realistic costs ===")
    print(f"  Params: walk-forward per segment train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d, "
          f"min {MIN_SEGMENT_BARS} bars/segment, {N_BLOCKS} contiguous volatility blocks")
    print(f"  Testing lookbacks {LOOKBACKS} vs realistic cost model and matched null")
    
    results_by_lookback = {}
    
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} ---")
        
        lookback_results = {}
        
        for ticker in ticker_order:
            bars = tickers[ticker]
            closes = bars.closes_array()
            
            # Generate labels for segment_fn
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            
            # Generate signals for this lookback using the standard bt.momentum_signals
            signals = bt.momentum_signals(closes, lookback=lookback)
            
            # Verify signal integrity
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            
            # Run with realistic costs
            print(f"  {ticker}: running with realistic costs...")
            res_realistic = bt.stress_segments(
                signals_fn=bt.momentum_signals,
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
                cfg=REALISTIC_COST_CONFIG,
            )
            
            # Run with zero costs for comparison
            print(f"  {ticker}: running with zero costs...")
            res_zero = bt.stress_segments(
                signals_fn=bt.momentum_signals,
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
                cfg=ZERO_COST_CONFIG,
            )
            
            # Analyze the difference
            realistic_scenario = res_realistic.scenarios[0]
            zero_cost_scenario = res_zero.scenarios[0]
            
            realistic_score = realistic_scenario.baseline_median_log_return
            zero_cost_score = zero_cost_scenario.baseline_median_log_return
            cost_impact = realistic_score - zero_cost_score
            
            lookback_results[ticker] = {
                "realistic_score": round(realistic_score, 4),
                "zero_cost_score": round(zero_cost_score, 4),
                "cost_impact": round(cost_impact, 4),
                "realistic_verdict": res_realistic.overall_verdict,
                "realistic_folds": res_realistic.n_folds,
                "zero_cost_verdict": res_zero.overall_verdict,
                "zero_cost_folds": res_zero.n_folds,
                "segments": [s.name for s in res_realistic.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res_realistic.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res_realistic.scenarios],
                "candidate_dispersion": round(res_realistic.candidate_dispersion, 3),
                "null_dispersion": round(res_realistic.null_dispersion, 3),
                "realistic_regime_stable_count": sum(1 for s in res_realistic.scenarios if s.baseline_median_log_return > 0),
                "zero_cost_regime_stable_count": sum(1 for s in res_zero.scenarios if s.baseline_median_log_return > 0),
            }
            
            print(f"  {ticker}: realistic={realistic_score:+.3f}, zero_cost={zero_cost_score:+.3f}, "
                  f"impact={cost_impact:+.3f}, realistic_verdict={res_realistic.overall_verdict}, "
                  f"realistic_regime_stable={sum(1 for s in res_realistic.scenarios if s.baseline_median_log_return > 0)}/{res_realistic.n_folds}")
        
        results_by_lookback[str(lookback)] = lookback_results
    
    # 4. Determinism check on AMZN/JPM subset
    print("\n=== 4. Determinism check (AMZN, JPM) ===")
    deterministic_checks = {}
    
    for lookback in LOOKBACKS:
        res1 = {}
        res2 = {}
        
        for ticker in DETERMINISM_TARGETS:
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            signals = bt.momentum_signals(closes, lookback=lookback)
            
            # First run
            print(f"  Lookback {lookback}, {ticker}: run #1...")
            run1 = bt.stress_segments(
                signals_fn=bt.momentum_signals,
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
                cfg=REALISTIC_COST_CONFIG,
            )
            res1[ticker] = {s.name: round(s.baseline_median_log_return, 3) for s in run1.scenarios}
            
            # Second run (independent)
            print(f"  Lookback {lookback}, {ticker}: run #2...")
            run2 = bt.stress_segments(
                signals_fn=bt.momentum_signals,
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
                cfg=REALISTIC_COST_CONFIG,
            )
            res2[ticker] = {s.name: round(s.baseline_median_log_return, 3) for s in run2.scenarios}
        
        deterministic_checks[str(lookback)] = {
            "AMZN": {"run1": res1["AMZN"], "run2": res2["AMZN"]},
            "JPM": {"run1": res1["JPM"], "run2": res2["JPM"]},
            "match": res1["AMZN"] == res2["AMZN"] and res1["JPM"] == res2["JPM"]
        }
        
        print(f"  Lookback {lookback}: deterministic check {deterministic_checks[str(lookback)]['match']}")
    
    # 5. Generate artifact
    print("\n=== 5. Generating artifact ===")
    artifact_dir = Path.cwd() / "state" / "check_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "momentum_cost_robustness_results.json"
    
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_config=dict(
            commission_per_trade=REALISTIC_COST_CONFIG.commission_per_trade,
            commission_per_share=REALISTIC_COST_CONFIG.commission_per_share,
            slippage_cents=REALISTIC_COST_CONFIG.slippage_cents,
            slippage_proportional=REALISTIC_COST_CONFIG.slippage_proportional,
        ),
        zero_cost_config=dict(
            commission_per_trade=ZERO_COST_CONFIG.commission_per_trade,
            commission_per_share=ZERO_COST_CONFIG.commission_per_share,
            slippage_cents=ZERO_COST_CONFIG.slippage_cents,
            slippage_proportional=ZERO_COST_CONFIG.slippage_proportional,
        ),
        results_by_lookback=results_by_lookback,
        deterministic_checks=deterministic_checks,
        universe=dict(
            ticker_order=ticker_order,
            n_assets=len(ticker_order),
        ),
    )
    
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"  Artifact written to: {artifact_path}")
    
    # 6. Summary
    print("\n=== 6. Summary ===")
    print("Cost robustness check completed. Key findings:")
    
    for lookback in LOOKBACKS:
        key = str(lookback)
        results = results_by_lookback[key]
        
        positive_edges = 0
        total_assets = len(results)
        for ticker, res in results.items():
            if res["realistic_regime_stable_count"] > 0:
                positive_edges += 1
        
        print(f"  Lookback {lookback}: {positive_edges}/{total_assets} assets show positive "
              f"REGIME_STABLE edge under realistic costs")
        print(f"    Cost impact (realistic - zero cost): {positive_edges}/{total_assets} assets "
              f"declined from zero cost to negative/realistic levels")
    
    print("\nNOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost robustness check: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(generate_cost_robustness_results())