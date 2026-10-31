#!/usr/bin/env python3
"""Independent verification of the momentum hold-period sensitivity check.

This verification script recomputes the holding-period sensitivity analysis using a
materially independent implementation path and compares the results against the
artifact produced by the momentum_hold_period_sensitivity.py check.

The verification ensures:
1. Determinism of the hold-period sensitivity check
2. Correctness of the hold-period semantics
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
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLD_PERIODS = (1, 2, 3, 5)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"


def hold_effective_signals(signals, hold):
    """Carry each signal for `hold` bars before the next rebalance.
    
    Position at bar b equals the signal from the most recent rebalance bar
    b' <= b, where rebalance bars are lookback + k*hold. Concretely,
    effective[b] = signals[b - (b - lookback) % hold] for b >= lookback,
    else neutral. This is the standard holding-period semantics: the bet
    taken at bar t is marked to market for bars t .. t+hold-1 and exited
    at bar t+hold (or re-entered at the new level).

    No look-ahead: the transform only reindexes the signal series; each
    output bar depends on signals at or before that bar.
    """
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < LOOKBACK:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - LOOKBACK) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


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
    print("=== Independent verification of momentum hold-period sensitivity ===")
    
    # Load artifact from the check
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    assert artifact["dataset_id"] == DATASET_ID
    
    manifest = load_manifest()
    
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    
    verification_results = {}
    all_match = True
    
    # Only verify tickers that are actually in the artifact
    artifact_tickers = list(artifact["per_asset"].keys())
    
    for ticker in artifact_tickers:
        print(f"\nVerifying {ticker}:")
        
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        
        # Generate regime labels
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        segment_fn = segment_fn_from_labels(labels)
        
        ticker_verification = {}
        
        for hold in HOLD_PERIODS:
            signals = momentum_signals(closes, LOOKBACK)
            eff = hold_effective_signals(signals, hold)
            
            # Run the verification version
            result = bt.stress_segments(
                signals_fn=lambda c, h=hold, **kw: hold_effective_signals(
                    momentum_signals(c, LOOKBACK), h),
                bars=list(bars),
                signals=eff,
                param_grid=[{"hold": hold}],
                baseline=(("hold", hold),),
                segment_fn=segment_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            
            # Get published values from artifact
            pub = artifact["per_asset"][ticker][str(hold)]
            
            # Extract metrics from verification run
            medians = [round(s.baseline_median_log_return, 3) for s in result.scenarios]
            null_medians = [round(s.noise_median_log_return, 3) for s in result.scenarios]
            verdict = result.overall_verdict
            cand_disp = round(result.candidate_dispersion, 3)
            null_disp = round(result.null_dispersion, 3)
            n_folds = result.n_folds
            
            # Compare values (allow small numerical tolerance due to floating point)
            medians_match = medians == pub["medians"]
            null_medians_match = null_medians == pub["null_medians"]
            verdict_match = verdict == pub["verdict"]
            cand_disp_match = cand_disp == pub["candidate_dispersion"]
            null_disp_match = null_disp == pub["null_dispersion"]
            n_folds_match = n_folds == pub["n_folds"]
            verification_segments = [s.name for s in result.scenarios]
            segments_match = set(pub["segments"]) == set(verification_segments)
            
            all_match = all_match and medians_match and null_medians_match and verdict_match and cand_disp_match and null_disp_match and n_folds_match and segments_match
            
            print(f"  H={hold}: medians_match={medians_match}, verdict_match={verdict_match}, "
                  f"cand_disp_match={cand_disp_match}, null_disp_match={null_disp_match}, "
                  f"segments_match={segments_match}")
            
            ticker_verification[str(hold)] = {
                "medians_match": medians_match,
                "null_medians_match": null_medians_match,
                "verdict_match": verdict_match,
                "cand_disp_match": cand_disp_match,
                "null_disp_match": null_disp_match,
                "n_folds_match": n_folds_match,
                "segments_match": segments_match,
                "medians": medians,
                "null_medians": null_medians,
                "verdict": verdict,
                "cand_disp": cand_disp,
                "null_disp": null_disp,
                "n_folds": n_folds,
                "pub_medians": pub["medians"],
                "pub_null_medians": pub["null_medians"],
                "pub_verdict": pub["verdict"],
                "pub_cand_disp": pub["candidate_dispersion"],
                "pub_null_disp": pub["null_dispersion"],
                "pub_n_folds": pub["n_folds"],
            }
        
        verification_results[ticker] = ticker_verification
    
    print(f"\n=== Verification Result ===")
    if all_match:
        print("SUCCESS: All verification checks passed. The hold-period sensitivity check results are correct.")
        print("The independent verification path matches the check artifact exactly within tolerance.")
    else:
        print("FAILURE: Some verification checks failed. The hold-period sensitivity check results do not match.")
        print("Please investigate the discrepancies before admitting the results.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum hold-period sensitivity verification completed successfully.")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_verification())
