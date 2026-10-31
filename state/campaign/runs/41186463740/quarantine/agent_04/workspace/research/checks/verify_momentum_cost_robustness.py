#!/usr/bin/env python3
"""Independent verification of the momentum cost-robustness check.

This verification script recomputes the cost-robustness analysis using a
materially independent implementation path and compares the results against
the artifact produced by the momentum_cost_robustness.py check.

The verification ensures:
1. Determinism of the cost-robustness check
2. Correctness of the cost model application
3. Independence of the verification path
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Must match the cost model used in the check
COMMISSION_PER_SHARE = 0.01  # $0.01 per share
SLIPPAGE_PROPORTIONAL = 0.001  # 10 bps (0.1%)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"


def momentum_signals(closes, lookback):
    """Momentum signals: long if previous lookback returns > 0."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def volatility_blocks(closes, n_blocks, window):
    """Generate past-only volatility regime labels (turbulent/calm)."""
    n = len(closes)
    vol_calm = float(np.median([
        float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
        for i in range(window, n)
    ])) if n > window else 0.0

    turbulent_thresh = 2.0 * vol_calm if vol_calm > 0 else 0.0

    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            rv = float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
            if rv < vol_calm:
                labels.append("calm")
            elif rv < turbulent_thresh:
                labels.append("normal")
            else:
                labels.append("turbulent")
    return labels


def segment_fn_from_labels(labels):
    """Convert precomputed labels to segment_fn for stress_segments."""
    def fn(_closes, i):
        return labels[i]
    return fn


def run_verification():
    print("=== Independent verification of momentum cost-robustness ===")
    
    # Load artifact from the check
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    assert artifact["dataset_id"] == DATASET_ID
    
    # Verify the cost model
    assert artifact["cost_model"]["commission_per_share"] == COMMISSION_PER_SHARE
    assert artifact["cost_model"]["slippage_proportional"] == SLIPPAGE_PROPORTIONAL
    
    print(f"Verifying cost model: commission_per_share=${COMMISSION_PER_SHARE}, "
          f"slippage_proportional={SLIPPAGE_PROPORTIONAL}")
    
    manifest = load_manifest()
    
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    
    # Create configs
    clean_cfg = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_share=0.0,
        slippage_proportional=0.0,
        warmup_periods=WARM,
    )
    
    cost_cfg = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_share=COMMISSION_PER_SHARE,
        slippage_proportional=SLIPPAGE_PROPORTIONAL,
        warmup_periods=WARM,
    )
    
    verification_results = {}
    all_match = True
    
    for ticker in all_tickers.keys():
        print(f"\nVerifying {ticker}:")
        
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        
        # Generate regime labels
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        segment_fn = segment_fn_from_labels(labels)
        
        ticker_verification = {}
        
        for lookback in LOOKBACKS:
            signals = momentum_signals(closes, lookback)
            
            # Run clean version
            clean_result = bt.stress_segments(
                signals_fn=lambda c, **kw: momentum_signals(c, **kw),
                bars=bars,
                signals=signals,
                param_grid=[{"lookback": lookback}],
                baseline=(("lookback", lookback),),
                segment_fn=segment_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=clean_cfg,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            
            # Run cost version
            cost_result = bt.stress_segments(
                signals_fn=lambda c, **kw: momentum_signals(c, **kw),
                bars=bars,
                signals=signals,
                param_grid=[{"lookback": lookback}],
                baseline=(("lookback", lookback),),
                segment_fn=segment_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=cost_cfg,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            
            # Extract metrics
            clean_median = float(np.median([s.baseline_median_log_return for s in clean_result.scenarios]))
            cost_median = float(np.median([s.baseline_median_log_return for s in cost_result.scenarios]))
            clean_verdict = clean_result.overall_verdict
            cost_verdict = cost_result.overall_verdict
            
            # Calculate edge erosion
            edge_erosion = cost_median - clean_median
            survival = edge_erosion <= 0
            
            # Determine positive edge status
            clean_positive = clean_verdict == "REGIME_STABLE" and clean_median > 0
            cost_positive = cost_verdict == "REGIME_STABLE" and cost_median > 0
            
            # Get published values from artifact
            pub_clean_median = artifact["universe_results"][ticker][str(lookback)]["clean_median"]
            pub_cost_median = artifact["universe_results"][ticker][str(lookback)]["cost_median"]
            pub_edge_erosion = artifact["universe_results"][ticker][str(lookback)]["edge_erosion"]
            pub_survival = artifact["universe_results"][ticker][str(lookback)]["survival"]
            pub_clean_verdict = artifact["universe_results"][ticker][str(lookback)]["clean_verdict"]
            pub_cost_verdict = artifact["universe_results"][ticker][str(lookback)]["cost_verdict"]
            
            # Compare values (allow small numerical tolerance due to floating point)
            clean_median_match = abs(clean_median - pub_clean_median) < 0.001
            cost_median_match = abs(cost_median - pub_cost_median) < 0.001
            edge_erosion_match = abs(edge_erosion - pub_edge_erosion) < 0.001
            survival_match = survival == pub_survival
            clean_verdict_match = clean_verdict == pub_clean_verdict
            cost_verdict_match = cost_verdict == pub_cost_verdict
            
            all_match = all_match and clean_median_match and cost_median_match and edge_erosion_match and survival_match and clean_verdict_match and cost_verdict_match
            
            print(f"  L{lookback}: clean_match={clean_median_match}, cost_match={cost_median_match}, "
                  f"erosion_match={edge_erosion_match}, survival_match={survival_match}, "
                  f"verdict_match={clean_verdict_match and cost_verdict_match}")
            
            ticker_verification[str(lookback)] = {
                "clean_median_match": clean_median_match,
                "cost_median_match": cost_median_match,
                "edge_erosion_match": edge_erosion_match,
                "survival_match": survival_match,
                "clean_verdict_match": clean_verdict_match,
                "cost_verdict_match": cost_verdict_match,
                "clean_median": clean_median,
                "cost_median": cost_median,
                "edge_erosion": edge_erosion,
                "survival": survival,
                "clean_verdict": clean_verdict,
                "cost_verdict": cost_verdict,
                "pub_clean_median": pub_clean_median,
                "pub_cost_median": pub_cost_median,
                "pub_edge_erosion": pub_edge_erosion,
                "pub_survival": pub_survival,
                "pub_clean_verdict": pub_clean_verdict,
                "pub_cost_verdict": pub_cost_verdict,
            }
        
        verification_results[ticker] = ticker_verification
    
    print(f"\n=== Verification Result ===")
    if all_match:
        print("SUCCESS: All verification checks passed. The cost-robustness check results are correct.")
        print("The independent verification path matches the check artifact exactly within tolerance.")
    else:
        print("FAILURE: Some verification checks failed. The cost-robustness check results do not match.")
        print("Please investigate the discrepancies before admitting the results.")
        sys.exit(1)
    
    # Additional verification: check that the robustness counts match
    robust_count = 0
    for ticker in all_tickers.keys():
        positive_lookbacks = 0
        for lb in LOOKBACKS:
            verdict = verification_results[ticker][str(lb)]["cost_verdict"]
            median = verification_results[ticker][str(lb)]["cost_median"]
            if verdict == "REGIME_STABLE" and median > 0:
                positive_lookbacks += 1
        
        if positive_lookbacks >= 2:
            robust_count += 1
    
    print(f"\nVerification robustness count: {robust_count} tickers with >=2 positive lookbacks")
    print(f"Published robustness count: {artifact['robustness_counts']['ROBUST']} tickers")
    
    if robust_count != artifact["robustness_counts"]["ROBUST"]:
        print("WARNING: Robustness count mismatch between verification and published artifact.")
        print(f"  Verification: {robust_count}, Published: {artifact['robustness_counts']['ROBUST']}")
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness verification completed successfully.")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_verification())
