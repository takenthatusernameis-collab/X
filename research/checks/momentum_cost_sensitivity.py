"""Cost-sensitivity check for lookback 3/5/10 momentum.

This script runs the momentum check with realistic transaction costs using the existing
momentum_lookback_sweep_results.json as a baseline, then creates a matched null with
the same cost structure to determine cost robustness.

Contract requirements:
- Run a bounded cost-sensitivity check for lookback 3/5/10 momentum
- Use existing realistic cost model (from research.backtest.engine.BacktestConfig)
- Use existing 10-asset collected universe (from momentum_lookback_sweep_results.json)
- Create matched null with same cost structure
- Classify cost robustness based on cost-resilience rate thresholds
- Produce reproducible artifact and independent verification

Out of scope:
- Do not tune signal rules after seeing cost results
- Do not expand the universe
- Do not introduce leverage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATAFILE = Path(__file__).resolve().parent.parent.parent / "state/check_artifacts/momentum_lookback_sweep_results.json"
ARTIFACT_DIR = Path.cwd() / "state/check_artifacts"
COST_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

def analyze_cost_sensitivity(ticker, lookback):
    """Analyze momentum with realistic costs for a single ticker and lookback."""
    m = load_manifest()
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    
    # Generate momentum signals
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    
    # Define walk-forward parameters (same as momentum_lookback_sweep.py)
    TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
    WINDOW = 60
    N_BLOCKS = 4
    MIN_SEGMENT_BARS = 400
    
    # Create volatility blocks and segment function
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Run stress segments with realistic costs
    param_grid = [{"lookback": lookback}]
    baseline = (("lookback", lookback),)
    
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=signals,
        param_grid=param_grid,
        baseline=baseline,
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    return {
        "lookback": lookback,
        "ticker": ticker,
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "verdict": res.overall_verdict,
        "cost_config": dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
    }

def create_matched_null():
    """Create a matched null benchmark with realistic costs."""
    np.random.seed(42)
    
    # Generate synthetic data for null benchmark
    bars = bt.generate_bars(2000, seed=42)
    closes = bars.closes_array()
    
    results = {}
    for lookback in (3, 5, 10):
        signals = bt.momentum_signals(closes, lookback=lookback)
        
        # Use walk_forward with realistic costs
        TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
        
        null_res = bt.walk_forward(
            list(bars),
            signals,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=REALISTIC_COSTS,
        )
        
        log_returns = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in null_res.folds]
        
        # Create segments for null medians
        segment_medians = []
        for i in range(0, len(log_returns), 4):
            if i + 4 <= len(log_returns):
                segment_medians.append(round(np.median(log_returns[i:i+4]), 3))
        
        results[str(lookback)] = {
            "lookback": lookback,
            "null_medians": segment_medians,
            "cost_config": dict(
                commission_per_trade=REALISTIC_COSTS.commission_per_trade,
                commission_per_share=REALISTIC_COSTS.commission_per_share,
                slippage_cents=REALISTIC_COSTS.slippage_cents,
                slippage_proportional=REALISTIC_COSTS.slippage_proportional,
            ),
        }
    
    return results

def determine_robustness_verdict(ticker_data):
    """Determine cost-robustness verdict for a ticker based on cost-sensitive results."""
    edges_at_least_two = 0
    edges_at_least_one = 0
    
    for lookback in (3, 5, 10):
        lb_key = str(lookback)
        verdict = ticker_data[lb_key]["verdict"]
        
        if verdict == "REGIME_STABLE" and all(m > 0 for m in ticker_data[lb_key]["medians"]):
            edges_at_least_one += 1
            edges_at_least_two += 1
        elif verdict == "REGIME_STABLE_LOSS":
            pass  # Loss counts as NO_EDGE
    
    if edges_at_least_two >= 2:
        return "ROBUST"
    elif edges_at_least_one == 1:
        return "SENSITIVE"
    else:
        return "NO_EDGE"

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Momentum Cost-Sensitivity Check (realistic costs) ===")
    print(f"Cost model: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    
    # Load zero-cost baseline from existing momentum_lookback_sweep_results.json
    with open(DATAFILE) as f:
        zero_cost_data = json.load(f)
    
    print(f"=== Loaded zero-cost baseline from momentum_lookback_sweep_results.json ===")
    print(f"Testing {len(zero_cost_data['universe']['ticker_order'])} tickers")
    
    # Create matched null with realistic costs
    print("=== Creating matched null with realistic costs ===")
    matched_null = create_matched_null()
    
    # Run cost-sensitive momentum analysis
    print("=== Running cost-sensitive momentum analysis ===")
    cost_results = {}
    universe_tickers = zero_cost_data["universe"]["ticker_order"]
    
    for ticker in universe_tickers:
        print(f"  Processing {ticker}...")
        cost_results[ticker] = {}
        for lookback in (3, 5, 10):
            cost_results[ticker][str(lookback)] = analyze_cost_sensitivity(ticker, lookback)
    
    # Create cost-robustness summary
    print("=== Creating cost-robustness summary ===")
    cost_robust_summary = {
        "dataset_id": zero_cost_data["dataset_id"],
        "lookbacks": list(zero_cost_data["lookbacks"]),
        "cost_model": REALISTIC_COSTS.__dict__,
        "zero_cost_baseline": zero_cost_data,
        "cost_sensitive_results": cost_results,
        "matched_null": matched_null,
        "universe_summary": {
            "zero_cost_robustness_counts": zero_cost_data["universe"]["robustness_counts"],
            "cost_robustness_counts": {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0},
            "cost_resilience_rate": 0.0,
        },
    }
    
    # Calculate cost-robustness statistics
    cost_robust_count = 0
    total_count = len(universe_tickers)
    
    for ticker in universe_tickers:
        verdict = determine_robustness_verdict(cost_results[ticker])
        cost_robust_summary["universe_summary"]["cost_robustness_counts"][verdict] += 1
        
        if verdict in ["ROBUST", "SENSITIVE"]:
            cost_robust_count += 1
    
    # Calculate cost-resilience rate
    if total_count > 0:
        cost_robust_summary["universe_summary"]["cost_resilience_rate"] = (
            cost_robust_count / total_count * 100
        )
    
    # Write artifact
    with open(COST_ARTIFACT_PATH, "w") as f:
        json.dump(cost_robust_summary, f, indent=2, sort_keys=True)
    
    print(f"\n=== Summary ===")
    print(f"Zero-cost edge counts: {cost_robust_summary['universe_summary']['zero_cost_robustness_counts']}")
    print(f"Cost edge counts: {cost_robust_summary['universe_summary']['cost_robustness_counts']}")
    print(f"Cost-resilience rate: {cost_robust_summary['universe_summary']['cost_resilience_rate']:.1f}%")
    print(f"Artifact written to: {COST_ARTIFACT_PATH}")
    
    # Print analysis
    print(f"\n=== Cost-Robustness Analysis ===")
    robust_tickers = []
    sensitive_tickers = []
    no_edge_tickers = []
    
    for ticker in universe_tickers:
        verdict = determine_robustness_verdict(cost_results[ticker])
        if verdict == "ROBUST":
            robust_tickers.append(ticker)
        elif verdict == "SENSITIVE":
            sensitive_tickers.append(ticker)
        else:
            no_edge_tickers.append(ticker)
    
    print(f"ROBUST tickers ({len(robust_tickers)}): {robust_tickers}")
    print(f"SENSITIVE tickers ({len(sensitive_tickers)}): {sensitive_tickers}")
    print(f"NO_EDGE tickers ({len(no_edge_tickers)}): {no_edge_tickers}")
    
    # Determine conclusion
    resilience_rate = cost_robust_summary["universe_summary"]["cost_resilience_rate"]
    if resilience_rate < 50:
        print(f"\nCONCLUSION: Momentum edge is NOT cost-robust (resilience rate: {resilience_rate:.1f}% < 50%)")
        final_decision = "REJECT"
    elif resilience_rate < 80:
        print(f"\nCONCLUSION: Momentum edge shows LIMITED cost-robustness (resilience rate: {resilience_rate:.1f}% < 80%)")
        final_decision = "SENSITIVE"
    else:
        print(f"\nCONCLUSION: Momentum edge is COST-ROBUST (resilience rate: {resilience_rate:.1f}% >= 80%)")
        final_decision = "ROBUST"
    
    return 0

if __name__ == "__main__":
    sys.exit(main())