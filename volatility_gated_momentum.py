"""Volatility-gated momentum (lookback-5) on the collected universe.

Background: this test examines whether a past-only volatility gate creates
a predeclared robustness improvement for the qualified momentum effect.

Method (fixed a-priori, not tuned): apply a volatility gate to lookback-5
momentum: long the previous 5-day return, hold 1 day, daily rebalance, but
only trade when recent realized annualized volatility (20-day std of log
returns) is below 0.20. The gate is applied post-signal: momentum signal
* volatility_gate (binary). This is past-only (vol window uses only closes
available at each bar) and the gate threshold is predeclared (0.20, not
fitted to the data). Walk-forward settings match the existing regime-
stability framework (train=252d/test=84d/warmup=60d/overlap=60d, 4
contiguous volatility blocks, coin-flip null on the same segments).

Falsification prediction: the gated momentum either shows CONSISTENT_WITH_NOISE
(no edge anywhere) or REGIME_DEPENDENT (the positive run concentrates in one
regime mix), or REGIME_STABLE_LOSS (uniformly negative), closing the cell.

Research/simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42

# Predeclared volatility gate parameters (declared before execution, not tuned)
VOL_LOOKBACK = 20  # 20-day realized volatility window
VOL_THRESHOLD = 0.20  # 20% annualized volatility threshold

# Standard walk-forward parameters (consistent with existing methodology)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60  # volatility block window
N_BLOCKS = 4  # number of contiguous volatility blocks for regime stability
MIN_SEGMENT_BARS = 400

# Predeclared lookback-5 momentum (fixed, not tuned)
MOM_LOOKBACK = 5

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "volatility_gated_momentum_results.json"


def volatility_gated_momentum_signals(closes: np.ndarray, vol_lookback: int, vol_threshold: float, mom_lookback: int):
    """Past-only volatility-gated momentum signals.

    Step 1: generate lookback-5 momentum signals (long previous 5-day return, hold 1 day)
    Step 2: apply past-only volatility gate (only trade when realized vol < threshold)
    Both steps use only closes available at each bar (no lookahead).
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    # Precompute momentum signals first
    momentum_signals = bt.momentum_signals(closes, lookback=mom_lookback)
    
    sqrt252 = np.sqrt(252.0)
    for i in range(vol_lookback - 1, n):
        # Calculate realized annualized volatility over vol_lookback window
        log_returns = np.log(closes[i - vol_lookback + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        
        # Apply volatility gate: only trade when volatility is below threshold
        if realized_annual < vol_threshold:
            # Use the momentum signal if volatility gate is satisfied
            signals[i] = momentum_signals[i]
        else:
            signals[i] = bt.Signal(date=i + 1, weight=0.0)
    
    # Handle bars before vol_lookback (neutral until we can calculate volatility)
    for i in range(vol_lookback - 1):
        signals[i] = bt.Signal(date=i + 1, weight=0.0)
    
    return signals


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    
    # Create volatility blocks for regime-stability testing
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Test volatility-gated momentum
    signals = volatility_gated_momentum_signals(closes, VOL_LOOKBACK, VOL_THRESHOLD, MOM_LOOKBACK)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=lambda c, **p: volatility_gated_momentum_signals(
            c, VOL_LOOKBACK, VOL_THRESHOLD, MOM_LOOKBACK),
        bars=list(bars),
        signals=signals,
        param_grid=[{"vol_lookback": VOL_LOOKBACK, "vol_threshold": VOL_THRESHOLD, "mom_lookback": MOM_LOOKBACK}],
        baseline=(("vol_lookback", VOL_LOOKBACK), ("vol_threshold", VOL_THRESHOLD), ("mom_lookback", MOM_LOOKBACK)),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    return {
        "asset": asset,
        "segments": [s.name for s in res.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "verdict": res.overall_verdict,
        "n_folds": res.n_folds,
    }


def main():
    # 1. Load universe and manifest
    print("\n=== 1. Load universe and manifest ===")
    manifest = load_manifest()
    print(f"  Dataset: {manifest['dataset_id']}")
    print(f"  Collection: {manifest['collection_date']}")
    print(f"  Universe: {len(manifest['universe'])} assets")
    
    # Load the first 10 assets for the test
    target = manifest["universe"][:10]  # First 10 assets
    tickers = {}
    for ticker in target:
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars
        print(f"  {ticker}: {bars.n_bars} bars")
    
    # 2. Test volatility-gated momentum across assets
    print("\n=== 2. Volatility-gated momentum (lookback-5, vol window 20, threshold 0.20) ===")
    print(f"Parameters: MOM_LOOKBACK={MOM_LOOKBACK}, VOL_LOOKBACK={VOL_LOOKBACK}, VOL_THRESHOLD={VOL_THRESHOLD}")
    print(f"Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"Segments: {N_BLOCKS} contiguous volatility blocks, each compared vs coin-flip null")
    print()
    print("Asset      Segments                    Medians                Candidate  Null     Verdict")
    print("-" * 95)
    
    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        
        segs = "[" + ",".join(res["segments"]) + "]"
        meds = "[" + ",".join(str(m) for m in res["medians"]) + "]"
        print(
            f"  {asset:5s} {segs:28s} {meds:28s} "
            f"{res['candidate_dispersion']:+10.3f}  {res['null_dispersion']:+10.3f}  {res['verdict']}")
    
    # 3. Summarize across assets
    print("\n=== 3. Summary across assets ===")
    verdict_counts = {}
    for asset in target:
        verdict = results_all[asset]["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    
    for verdict, count in verdict_counts.items():
        print(f"  {verdict}: {count}")
    
    # Determine overall verdict
    positive_stable = sum(1 for asset in target 
                         if results_all[asset]["verdict"] in ["REGIME_STABLE", "ROBUST_POSITIVE"])
    if positive_stable > 0 and sum(1 for v in verdict_counts.values()) - positive_stable == 0:
        overall = "ROBUST_POSITIVE"
        conclusion = "The volatility-gated momentum shows a robust positive edge across all tested assets and regimes."
    elif any(v in ["REGIME_DEPENDENT", "REGIME_STABLE_LOSS"] for v in verdict_counts.keys()):
        overall = "NEGATIVE_RESULT"
        conclusion = "The volatility-gated momentum shows regime-dependence or stable losses, indicating the gate did not create a robust improvement."
    else:
        overall = "CONSISTENT_WITH_NOISE"
        conclusion = "The volatility-gated momentum is consistent with noise (no reproducible edge)."
    
    print(f"\n  Overall verdict: {overall}")
    print(f"  Conclusion: {conclusion}")
    
    # 4. Determinism check
    print("\n=== 4. Determinism verification ===")
    results_all2 = {}
    for asset in target:
        results_all2[asset] = run_asset(asset, tickers)
    
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(results_all2, sort_keys=True)
    deterministic = out1 == out2
    print(f"Determinism: r1 == r2: {deterministic}")
    if not deterministic:
        print("ERROR: Non-deterministic output")
        sys.exit(1)
    
    # 5. Generate artifact
    print("\n=== 5. Generate artifact ===")
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "parameters": {
            "mom_lookback": MOM_LOOKBACK,
            "vol_lookback": VOL_LOOKBACK,
            "vol_threshold": VOL_THRESHOLD,
            "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
            "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        },
        "pre_asset_results": results_all,
        "verdict_counts": verdict_counts,
        "overall_verdict": overall,
        "conclusion": conclusion,
        "deterministic": deterministic,
    }
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"  artifact written to: {ARTIFACT_PATH}")
    
    print("\n" + "="*80)
    print("VOLATILITY-GATED MOMENTUM TEST COMPLETED")
    print("="*80)
    print(f"Parameters: lookback={MOM_LOOKBACK}, vol_window={VOL_LOOKBACK}, threshold={VOL_THRESHOLD}")
    print(f"Overall verdict: {overall}")
    print(f"Conclusion: {conclusion}")
    print(f"Deterministic: {deterministic}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())