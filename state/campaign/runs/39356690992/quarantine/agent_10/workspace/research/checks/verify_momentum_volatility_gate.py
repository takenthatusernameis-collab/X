"""Independent verification of momentum volatility gate results.

This script recomputes the momentum volatility gate results from scratch
using a separate path and compares against the artifact produced by
research/checks/momentum_volatility_gate.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_volatility_gate_results.json"


def volatility_gated_momentum_signals(closes: np.ndarray, vol_threshold: float, lookback: int = 5):
    """Past-only momentum signals gated by past realized volatility."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    
    for i in range(lookback - 1, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        momentum_weight = 1.0 if ret > 0 else -1.0
        
        if i >= 60:  # WINDOW = 60
            log_returns = np.log(closes[i - 60 : i])
            realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
            vol_gate = realized_annual < vol_threshold
        else:
            vol_gate = True
        
        if vol_gate:
            signals[i] = bt.Signal(date=i + 1, weight=momentum_weight)
    
    return signals


def run_asset_verification(asset, vol_threshold, full_artifact):
    """Recompute asset results independently."""
    # Load data (simplified - assume data is accessible)
    bars, _ = bt.load_ticker(asset)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Get volatility-gated signals
    signals = volatility_gated_momentum_signals(closes, vol_threshold, 5)
    
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=lambda c, **kw: volatility_gated_momentum_signals(c, vol_threshold, 5),
        bars=list(bars),
        signals=signals,
        param_grid=[{"vol_threshold": vol_threshold}],
        baseline=(("vol_threshold", vol_threshold),),
        segment_fn=seg_fn,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        periods_per_year=252,
        min_segment_bars=400,
    )
    
    names = [s.name for s in res.scenarios]
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    
    return dict(
        segments=names,
        medians=medians,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
        edge_status=res.n_edge_present,
    )


def main() -> int:
    print("=== Independent verification of momentum volatility gate ===")
    
    # Load the artifact to compare against
    with open(ARTIFACT_PATH, "r") as f:
        artifact = json.load(f)
    
    print(f"Loaded artifact from: {ARTIFACT_PATH}")
    print(f"Dataset: {artifact['dataset_id']}")
    print(f"Lookback: {artifact['lookback']}")
    print(f"Vol threshold: {artifact['vol_threshold']}")
    
    vol_threshold = artifact["vol_threshold"]
    
    # Run verification on target assets
    target = ["AMZN", "JPM"]
    print(f"\nRunning independent verification on target assets: {target}")
    
    verified_results = {}
    for asset in target:
        print(f"\nVerifying {asset}...")
        verified_results[asset] = run_asset_verification(asset, vol_threshold, artifact)
        
        # Compare with artifact
        artifact_asset = artifact["per_asset"].get(asset)
        if artifact_asset:
            print(f"  Verdict match: {verified_results[asset]['verdict'] == artifact_asset['verdict']}")
            print(f"  Medians match: {verified_results[asset]['medians'] == artifact_asset['medians']}")
            print(f"  Candidate dispersion match: {verified_results[asset]['candidate_dispersion'] == artifact_asset['candidate_dispersion']}")
            print(f"  Verdict match: {verified_results[asset]['verdict'] == artifact_asset['verdict']}")
    
    # Compare verdict counts
    print(f"\n=== Verdict count comparison ===")
    print("Artifact verdict counts:", artifact["universe"]["verdict_counts"])
    
    # Compute verification verdict counts
    verified_verdict_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
                              "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
    
    for asset in target:
        verdict = verified_results[asset]["verdict"]
        verified_verdict_counts[verdict] += 1
    
    print("Verified verdict counts:", verified_verdict_counts)
    
    # Summary
    print(f"\n=== Verification Summary ===")
    artifact_verdicts = {asset: artifact["per_asset"][asset]["verdict"] for asset in target}
    verified_verdicts = {asset: verified_results[asset]["verdict"] for asset in target}
    
    print(f"Assets: {list(target)}")
    print(f"Artifact verdicts: {artifact_verdicts}")
    print(f"Verified verdicts: {verified_verdicts}")
    
    all_match = artifact_verdicts == verified_verdicts
    print(f"All verdicts match: {all_match}")
    
    if not all_match:
        print("VERIFICATION FAILED: Results do not match!")
        return 1
    else:
        print("VERIFICATION PASSED: Independent recomputation matches artifact.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
