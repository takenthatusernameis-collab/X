import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

# Realistic institutional cost model for US equities
# Source: Industry standard transaction cost analysis
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=5.0,           # $5 per trade (typical broker fee)
    commission_per_share=0.01,          # 1 cent per share
    slippage_cents=5.0,                 # $0.05 slippage per share (tight market)
    slippage_proportional=0.0005,       # 5 bps proportional slippage (liquid stocks)
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,                     # No margin for conservative testing
    margin_call_liquidate=False,
)

# Matched null cost config (same costs, different signals)
MATCHED_NULL_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=5.0,
    commission_per_share=0.01,
    slippage_cents=5.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)


def segment_cost_result(ticker, bars, lookback):
    """Walk-forward cost-robustness result for one ticker and lookback.
    
    Uses fresh momentum signals and cost application to compute median log returns.
    """
    closes = bars.closes_array()
    n_bars = len(bars)
    
    # Split into volatility segments (4 blocks as in other checks)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    segment_fn = bt.segment_fn_from_labels(labels)
    
    # Get momentum signals
    signals = bt.momentum_signals(closes, lookback=lookback)
    
    # Run momentum with realistic costs using stress_segments
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=segment_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=REALISTIC_COST_CONFIG,
    )
    
    # Get matched null using noise_benchmark (random signals with same cost model)
    # This creates a proper coin-flip null that matches the cost structure
    from research.backtest.perturbation import noise_benchmark
    
    null_res = noise_benchmark(
        bars=list(bars),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=MATCHED_NULL_COST_CONFIG,
        periods_per_year=252,
        seed=SEED + 1000,
    )
    
    # Extract key metrics from first segment (calm)
    cand_scenario = res.scenarios[0]
    
    # For the null, we need to get the baseline (coin-flip) result
    # The noise_benchmark returns a SweepSummary, not a RegimeStressResult
    # We'll use the baseline_median_log_return as our comparison
    null_median = null_res.baseline_median_log_return
    
# Determine cost robustness verdict based on candidate medians (simplified logic)
    # Compute overall verdict based on all segment medians combined
    overall_cand_medians = cand_scenario.median_log_returns
    if all(abs(m) <= 0.05 for m in overall_cand_medians):
        cost_verdict = "CONSISTENT_WITH_NOISE"
    elif all(m < 0 for m in overall_cand_medians):
        cost_verdict = "REGIME_STABLE_LOSS"
    elif all(m > 0 for m in overall_cand_medians):
        cost_verdict = "REGIME_STABLE"
    else:
        cost_verdict = "REGIME_DEPENDENT"
    
    # Determine if cost edge is present
    cost_edge_present = abs(cand_scenario.baseline_median_log_return - null_median) > 0.05
    
    # For null dispersion, use the noise_fold_median_log_returns if available,
    # otherwise estimate from the effective_tolerance
    if null_res.noise_fold_median_log_returns:
        null_dispersion = float(np.std(null_res.noise_fold_median_log_returns, ddof=1))
    else:
        null_dispersion = null_res.effective_tolerance
    
# For null dispersion, use the noise_fold_median_log_returns if available,
    # otherwise estimate from the effective_tolerance
    if null_res.noise_fold_median_log_returns:
        null_dispersion = float(np.std(null_res.noise_fold_median_log_returns, ddof=1))
    else:
        null_dispersion = null_res.effective_tolerance
    
    return {
        "segments": [s.name for s in res.scenarios],  # res is RegimeStressResult
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(null_median, 3) for _ in res.scenarios],  # Use same null for all segments
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(null_dispersion, 3),
        "cost_robustness_verdict": cost_verdict,
        "cost_edge_present": cost_edge_present,
        "n_folds": res.n_folds,
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verify manifest integrity
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']}"
    
    print("=== Cost robustness test for lookback 3 / 5 / 10 momentum ===")
    print(f"Dataset: {manifest['dataset_id']} | Universe: {len(manifest['universe'])} assets")
    print(f"Cost model: realistic institutional trading (commission + slippage)")
    print(f"Lookback windows: {LOOKBACKS}")
    print(f"Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"Segments: {N_BLOCKS} volatility blocks, min {MIN_SEGMENT_BARS} bars/segment")
    print()
    
    # Load all tickers from collected universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    
    ticker_order = list(tickers.keys())
    print(f"Testing across {len(ticker_order)} assets: {ticker_order}")
    print()
    
    # Run cost robustness tests for each lookback
    results = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_config": {
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
        },
        "matched_null_cost_config": {
            "commission_per_trade": MATCHED_NULL_COST_CONFIG.commission_per_trade,
            "commission_per_share": MATCHED_NULL_COST_CONFIG.commission_per_share,
            "slippage_cents": MATCHED_NULL_COST_CONFIG.slippage_cents,
            "slippage_proportional": MATCHED_NULL_COST_CONFIG.slippage_proportional,
        },
        "per_lookback": {},
    }
    
    for lookback in LOOKBACKS:
        print(f"=== Lookback {lookback} ===")
        lookback_results = {}
        
        for ticker in ticker_order:
            result = segment_cost_result(ticker, tickers[ticker], lookback)
            lookback_results[ticker] = result
            
            # Print summary
            edge_status = "EDGE" if result["cost_edge_present"] else "NO_EDGE"
            print(f"  {ticker:6s}: {edge_status:6s} | cand={result['medians'][0]:+.3f}, "
                  f"null={result['null_medians'][0]:+.3f}, disp={result['candidate_dispersion']:.3f}")
        
        results["per_lookback"][str(lookback)] = lookback_results
    
    # Compute summary statistics
    print("\n=== Summary ===")
    edge_counts = {lb: 0 for lb in LOOKBACKS}
    robust_counts = {lb: 0 for lb in LOOKBACKS}
    
    for lb in LOOKBACKS:
        for ticker in ticker_order:
            result = results["per_lookback"][str(lb)][ticker]
            if result["cost_edge_present"]:
                edge_counts[lb] += 1
                if result["cost_robustness_verdict"] == "REGIME_STABLE":
                    robust_counts[lb] += 1
    
    for lb in LOOKBACKS:
        edge_pct = edge_counts[lb] / len(ticker_order) * 100
        robust_pct = robust_counts[lb] / len(ticker_order) * 100
        print(f"Lookback {lb}: {edge_counts[lb]}/{len(ticker_order)} edges ({edge_pct:.1f}%) | "
              f"{robust_counts[lb]} robust ({robust_pct:.1f}%)")
    
    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Cost robustness test for momentum lookback 3/5/10 with realistic transaction costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
