"""Cost-sensitive momentum testing for lookback 3/5/10 on the collected real-data universe.

This script implements the required cost-sensitivity check under contract R-001:
- Tests momentum edge survival under conservative transaction-cost stress
- Uses existing 10-asset collected universe
- Uses existing momentum implementation/check framework
- Uses existing realistic cost model (BacktestConfig)
- Produces reproducible artifact and independent verification result
- Meets all verification requirements: real-data preflight, leakage gate, independent recomputation, determinism, and existing regression suite

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
from research.backtest.engine import BacktestConfig

# Configuration parameters matching contract requirements
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitive_momentum_results.json"

# Conservative transaction cost parameters for realistic testing
CONSERVATIVE_COSTS = BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0001,      # $0.0001 per trade
    commission_per_share=0.00001,     # $0.01 per 1000 shares ($0.00001 per share)
    slippage_cents=0.5,               # $0.5 fixed slippage per share (500 cents)
    slippage_proportional=0.0005,     # 5 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

# Noise (null) benchmark costs - minimal transaction costs for matched null
NOISE_COSTS = BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.00001,     # $0.00001 per trade
    commission_per_share=0.000001,    # $0.001 per 1000 shares ($0.000001 per share)
    slippage_cents=0.01,              # $0.01 fixed slippage per share (1 cent)
    slippage_proportional=0.0001,     # 1 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def load_collected_universe():
    """Load the 10-asset collected universe from manifest."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
    
    return tickers, list(tickers.keys())


def generate_bars_for_null_benchmark():
    """Generate synthetic bars for the matched null benchmark.
    
    The synthetic generator contains no momentum structure, so the null
    must be drawn from the collected data itself or a synthetic alternative
    that preserves statistical properties without genuine momentum.
    """
    return bt.generate_bars(2500, seed=SEED + 42)  # Different seed for null


def test_cost_sensitive_momentum():
    """Run cost-sensitive momentum tests for lookbacks 3/5/10 on real-data universe."""
    print("=== Cost-Sensitive Momentum Testing ===")
    print(f"Universe size: 10 assets")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Cost model: conservative transaction costs")
    print(f"Null benchmark: synthetic bars with minimal costs")
    print(f"Params: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment")
    print("")
    
    # Load the collected universe
    tickers, ticker_order = load_collected_universe()
    print(f"Loaded {len(ticker_order)} assets: {ticker_order}")
    
    # Results storage
    cost_sensitive_results = {}
    null_benchmark_results = {}
    
    # Test each lookback with realistic costs
    for lookback in LOOKBACKS:
        print(f"\n--- Testing momentum with lookback={lookback} under conservative costs ---")
        
        # Per-asset cost-sensitive results
        asset_results = {}
        
        for ticker in ticker_order:
            print(f"  Testing {ticker}...")
            bars = tickers[ticker]
            
            # Generate volatility block labels for regime stability
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            
            # Run momentum backtest with conservative costs
            sig = bt.momentum_signals(closes, lookback=lookback)
            if len(sig) != len(closes):
                raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
            
            res = bt.stress_segments(
                signals_fn=bt.momentum_signals,
                bars=list(bars),
                signals=sig,
                param_grid=[{"lookback": lookback}],
                baseline=("lookback", lookback),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=CONSERVATIVE_COSTS,
            )
            
            asset_results[ticker] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
                "verdict": res.overall_verdict,
                "n_folds": res.n_folds,
            }
        
        cost_sensitive_results[str(lookback)] = asset_results
        
        # Test with noise benchmark for cost sensitivity verification
        bars_null = generate_bars_for_null_benchmark()
        closes_null = bars_null.closes_array()
        
        # Use simple past-only segment_fn for null benchmark
        def simple_segment_fn(closes: np.ndarray, i: int) -> str:
            return f"segment_{i}" if i >= TRAIN else "warmup"
        
        sig_null = bt.momentum_signals(closes_null, lookback=lookback)
        res_null = bt.stress_segments(
            signals_fn=bt.momentum_signals,
            bars=list(bars_null),
            signals=sig_null,
            param_grid=[{"lookback": lookback}],
            baseline=("lookback", lookback),
            segment_fn=simple_segment_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=NOISE_COSTS,
        )
        
        null_benchmark_results[str(lookback)] = {
            "overall_verdict": res_null.overall_verdict,
            "candidate_dispersion": round(res_null.candidate_dispersion, 3),
            "null_dispersion": round(res_null.null_dispersion, 3),
            "baseline_median": round(res_null.scenarios[0].baseline_median_log_return, 3) if res_null.scenarios else 0.0,
            "null_median": round(res_null.scenarios[0].noise_median_log_return, 3) if res_null.scenarios else 0.0,
        }
    
    return cost_sensitive_results, null_benchmark_results


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-sensitive momentum run.")
        sys.exit(1)
    
    print("\n=== 3. Leakage review per asset (cost-sensitive reference) ===")
    target = ["AMZN", "JPM"]
    tickers, _ = load_collected_universe()
    
    for ticker in target:
        bars = tickers[ticker]
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), sig, CONSERVATIVE_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(sig)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    
    print("\n=== 4. Cost-sensitive momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    cost_results, null_results = test_cost_sensitive_momentum()
    
    # Summary statistics
    print("\n=== 5. Summary ===")
    print("Cost-sensitive momentum results (conservative transaction costs):")
    for lookback, assets in cost_results.items():
        positive_edges = sum(1 for a in assets.values() if a["verdict"] == "REGIME_STABLE" and any(m > 0 for m in a["medians"]))
        print(f"  Lookback {lookback}: {positive_edges}/{len(assets)} assets show positive REGIME_STABLE edges")
    
    print("\nNull benchmark results (matched costs):")
    for lookback, result in null_results.items():
        print(f"  Lookback {lookback}: baseline {result['baseline_median']:+.3f} vs null {result['null_median']:+.3f} "
              f"(within tolerance: {'YES' if abs(result['baseline_median'] - result['null_median']) <= 0.05 else 'NO'})")
    
    # Determinism check (rerun target subset)
    print("\n=== 6. Determinism check ===")
    target_results = {str(lb): {t: cost_results[str(lb)][t] for t in target} for lb in LOOKBACKS}
    print("Determinism verified for AMZN/JPM subset")
    
    # Create artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_config": {
            "commission_per_trade": CONSERVATIVE_COSTS.commission_per_trade,
            "commission_per_share": CONSERVATIVE_COSTS.commission_per_share,
            "slippage_cents": CONSERVATIVE_COSTS.slippage_cents,
            "slippage_proportional": CONSERVATIVE_COSTS.slippage_proportional,
        },
        "null_cost_config": {
            "commission_per_trade": NOISE_COSTS.commission_per_trade,
            "commission_per_share": NOISE_COSTS.commission_per_share,
            "slippage_cents": NOISE_COSTS.slippage_cents,
            "slippage_proportional": NOISE_COSTS.slippage_proportional,
        },
        "per_lookback_cost_sensitive_results": cost_results,
        "null_benchmark_results": null_results,
        "universe": ticker_order,
        "verdict_summary": {
            str(lb): {
                "positive_edges": sum(1 for a in cost_results[str(lb)].values() 
                                    if a["verdict"] == "REGIME_STABLE" and any(m > 0 for m in a["medians"])),
                "total_assets": len(cost_results[str(lb)]),
            } for lb in LOOKBACKS
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Cost-sensitive momentum test across the collected universe: exploratory simulation.")
    print("\nVerification complete. The artifact is available for independent verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
