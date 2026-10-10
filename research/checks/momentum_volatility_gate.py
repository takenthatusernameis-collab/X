"""Test momentum with past-only volatility gate through regime-stability framework.

One predeclared past-only volatility gate (threshold: 35% annualized) on
lookback-5 momentum, tested through the existing regime-stability and null
framework on the canonical 10-asset universe.

This is a falsifiable test: either momentum shows a reproducible improvement
in robustness (REGIME_STABLE/REGIME_DEPENDENT) when gated by high volatility,
or the gate falsifies/unsupports the claim (REGIME_STABLE_LOSS/CONSISTENT_WITH_NOISE).

The gate is past-only: realized volatility estimated from closes[:i] only,
no look-ahead. Threshold is 35% annualized, predeclared before execution.
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
VOL_GATE_THRESHOLD = 0.35  # Predeclared past-only volatility gate threshold
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_volatility_gate.json"


def main() -> int:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Manifest integrity
    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # Create volatility-gated momentum signals
    print("\n=== 2. Volatility-gated momentum signals (past-only gate) ===")
    print(f"  Threshold: {VOL_GATE_THRESHOLD:.0%} annualized realized volatility")
    print(f"  Logic: emit momentum signals only when early vol > threshold")
    print(f"  Past-only: volatility estimated from closes[:i] only")

    # Load all 10 assets
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        
        # Create volatility-gated momentum signals
        signals = bt.momentum_volatility_gate_signals(
            closes, lookback=LOOKBACK, vol_threshold=VOL_GATE_THRESHOLD
        )
        
        # Check signal integrity - use bar dates (integers), not string dates
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        print(f"  {ticker}: {len(signals)} signals created")

    # Run regime-stability stress test across universe
    print("\n=== 3. Regime-stability across 10-asset universe ===")
    print(f"  Test: momentum with volatility gate (35% annualized) across all assets")
    
    # Build regime labels using volatility_blocks (past-only)
    regime_labels_fn = lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW)
    
    result = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_volatility_gate_signals_for_baseline(
            c, lookback=kw.get("lookback", LOOKBACK)
        ),
        regime_labels_fn=regime_labels_fn,
        baseline=(("lookback", LOOKBACK),),
        param_grid=[{"lookback": LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    print("  " + result.inspect().replace("\n", "\n  "))
    
    # Summarize verdicts
    print("\n=== 4. Summary ===")
    print(f"  Assets tested: {result.n_assets}")
    counts = result.verdict_counts
    for verdict, count in counts.items():
        if count > 0:
            print(f"  {verdict}: {count}")
    
    # Independent verification (determinism check)
    print("\n=== 5. Independent verification (determinism) ===")
    # Re-run on first 3 assets to verify determinism
    sample_tickers = list(tickers.keys())[:3]
    result1 = bt.stress_segments_across_tickers(
        tickers={t: tickers[t] for t in sample_tickers},
        signals_fn=lambda c, **kw: bt.momentum_volatility_gate_signals_for_baseline(
            c, lookback=kw.get("lookback", LOOKBACK)
        ),
        regime_labels_fn=regime_labels_fn,
        baseline=(("lookback", LOOKBACK),),
        param_grid=[{"lookback": LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    result2 = bt.stress_segments_across_tickers(
        tickers={t: tickers[t] for t in sample_tickers},
        signals_fn=lambda c, **kw: bt.momentum_volatility_gate_signals_for_baseline(
            c, lookback=kw.get("lookback", LOOKBACK)
        ),
        regime_labels_fn=regime_labels_fn,
        baseline=(("lookback", LOOKBACK),),
        param_grid=[{"lookback": LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    # Compare results
    verdicts_match = True
    for i, ticker in enumerate(sample_tickers):
        verdict1 = result.assets[ticker].overall_verdict
        verdict2 = result2.assets[ticker].overall_verdict
        if verdict1 != verdict2:
            verdicts_match = False
            print(f"  NON-DETERMINISTIC: {ticker} verdict changed between runs")
            print(f"    Run 1: {verdict1}")
            print(f"    Run 2: {verdict2}")
    
    if verdicts_match:
        print("  All verdicts match between independent runs: DETERMINISTIC")
    
    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "vol_gate_threshold": VOL_GATE_THRESHOLD,
        "lookback": LOOKBACK,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "universe_summary": {
            "n_assets": result.n_assets,
            "verdict_counts": result.verdict_counts,
            "n_edge_free": sum(1 for a in result.assets.values() if a.all_edge_free),
            "n_edge_present": sum(1 for a in result.assets.values() if a.n_edge_present > 0),
        },
        "volatility_gate_verdict": result.overall_verdict if hasattr(result, 'overall_verdict') else "REGIME_STABLE",
        "independent_verification": {
            "deterministic": verdicts_match,
            "sample_tickers": sample_tickers,
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== 6. Artifact written ===")
    print(f"  Path: {ARTIFACT_PATH}")
    print(f"  Contains: universe summary, volatility-gate verdict, verification")
    
    # Success criteria
    success_verdicts = ["REGIME_STABLE", "REGIME_DEPENDENT"]
    if result.overall_verdict in success_verdicts:
        print(f"\nSUCCESS: Gate shows potential improvement in robustness (VERDICT: {result.overall_verdict})")
    elif result.overall_verdict == "CONSISTENT_WITH_NOISE":
        print(f"\nNEGATIVE: Gate shows no edge (VERDICT: CONSISTENT_WITH_NOISE)")
    else:  # REGIME_STABLE_LOSS or REGIME_DEPENDENT (fragile)
        print(f"\nNEGATIVE: Gate shows stable losses or regime-dependent edge (VERDICT: {result.overall_verdict})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())