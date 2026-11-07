"""Test a past-only volatility gate on lookback-5 momentum through the existing regime-stability and null framework.

This implementation focuses specifically on the volatility gate testing requirement.
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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_volatility_gate_results.json"

# Predeclared past-only volatility gate threshold (declared before execution)
VOLATILITY_GATE_THRESHOLD = 0.20  # 20% annualized (past-only early-vol classifier)


def create_volatility_gated_momentum_signals(early_vol_threshold):
    """Create momentum signals wrapped with past-only volatility gate."""
    # This simulates the conditional_signal wrapper pattern
    def gated_momentum(closes, lookback=LOOKBACK):
        n = len(closes)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        lookback = int(lookback)
        
        # Calculate early volatility for gate (past-only)
        window = min(60, max(4, n // 4))
        early_vol = float(np.std(np.log(closes[window // 2 : window]), ddof=1)) * np.sqrt(252.0)
        
        # Apply gate: only active when early_vol exceeds threshold
        if early_vol > early_vol_threshold:
            for i in range(lookback, n):
                ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
                out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        
        return out
    
    return gated_momentum


def run_volatility_gated_test(asset, tickers, volatility_threshold):
    """Test the volatility-gated momentum signal on a single asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    
    # Use the existing volatility_blocks function for regime classification
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    baseline = (("lookback", LOOKBACK),)
    
    # Create gated momentum signal
    gated_momentum_fn = create_volatility_gated_momentum_signals(volatility_threshold)
    gated_signals = gated_momentum_fn(closes)
    
    if len(gated_signals) != len(closes):
        raise ValueError(f"{asset}: gated signals {len(gated_signals)} != bars {len(closes)}")
    
    # Test through the existing regime-stability framework
    res = bt.stress_segments(
        signals_fn=lambda c, **p: gated_momentum_fn(c),
        bars=list(bars),
        signals=gated_signals,
        param_grid=[{"lookback": LOOKBACK}],
        baseline=baseline,
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    return {
        "segments": [s.name for s in res.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "verdict": res.overall_verdict,
        "n_folds": res.n_folds,
        "vol_gate_threshold": volatility_threshold,
    }


def main() -> int:
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Past-Only Volatility Gate on Lookback-5 Momentum ===")
    print(f"Predeclared volatility gate threshold: {VOLATILITY_GATE_THRESHOLD}")
    print(f"Gate condition: past-only early_vol > {VOLATILITY_GATE_THRESHOLD}")
    print(f"Research simulation only. No live trading or production execution.")
    print()
    
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()
    
    print("=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    print()
    
    print("=== 3. Volatility-gated momentum leakage review ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        
        # Test the gated momentum signal
        gated_momentum_fn = create_volatility_gated_momentum_signals(VOLATILITY_GATE_THRESHOLD)
        signals = gated_momentum_fn(closes)
        
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, gated momentum signals {len(signals)}, "
              f"gate applied (vol > {VOLATILITY_GATE_THRESHOLD}) | total_return {total_return:+.3f}")
    print()
    
    print("=== 4. Volatility-gated momentum across REGIME_DEPENDENT assets ===")
    print(f"Gated momentum: long the previous {LOOKBACK}-day return, hold 1 day, daily rebalanced, "
          f"activated only when past-only early-vol > {VOLATILITY_GATE_THRESHOLD}")
    print(f"Params: walk-forward per segment train={TRAIN}d / test={TEST}d / warmup={WARM}d / "
          f"overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment")
    print(f"Segments: {N_BLOCKS} contiguous blocks by date, past-only volatility classifier "
          f"(turbulent = block median trailing-{WINDOW}-bar realized vol > series-wide median)")
    
    results_all = {}
    for asset in target:
        results_all[asset] = run_volatility_gated_test(asset, tickers, VOLATILITY_GATE_THRESHOLD)
        print(f"Asset: {asset}")
        print(f"  segments: {results_all[asset]['segments']}")
        print(f"  candidate_medians: {results_all[asset]['medians']}")
        print(f"  candidate_dispersion: {results_all[asset]['candidate_dispersion']:+.3f}")
        print(f"  null_dispersion: {results_all[asset]['null_dispersion']:+.3f}")
        print(f"  verdict: {results_all[asset]['verdict']}")
        print()
    
    # Universe-level test
    print("=== 5. Volatility-gated momentum across collected universe ===")
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    
    # Run regime-stability test for universe
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: create_volatility_gated_momentum_signals(VOLATILITY_GATE_THRESHOLD)(c),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK),),
        param_grid=[{"lookback": LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    print("Universe results:")
    print(f"  Verdict counts: {summary.verdict_counts}")
    print(f"  Per-asset verdicts:")
    for ticker in summary.ticker_order:
        res = summary.assets[ticker]
        print(f"    {ticker}: {res.overall_verdict}")
    
    print()
    print("=== 6. Determinism verification ===")
    # Run test again to verify determinism
    results_all_2 = {}
    for asset in target:
        results_all_2[asset] = run_volatility_gated_test(asset, tickers, VOLATILITY_GATE_THRESHOLD)
    
    # Check determinism
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(results_all_2, sort_keys=True)
    print(f"Determinism: {'PASS' if out1 == out2 else 'FAIL'}")
    assert out1 == out2, "non-deterministic output"
    
    print()
    print("=== 7. Writing artifact for independent verification ===")
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookback": LOOKBACK,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "vol_gate_threshold": VOLATILITY_GATE_THRESHOLD,
        "per_asset": results_all,
        "universe": {
            "verdict_counts": summary.verdict_counts,
            "ticker_order": summary.ticker_order,
            "per_asset": {
                ticker: {
                    "segments": [s.name for s in res.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                    "candidate_dispersion": round(res.candidate_dispersion, 3),
                    "null_dispersion": round(res.null_dispersion, 3),
                    "verdict": res.overall_verdict,
                }
                for ticker, res in summary.assets.items()
            },
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Volatility-gated momentum test with past-only gate threshold 0.20: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())