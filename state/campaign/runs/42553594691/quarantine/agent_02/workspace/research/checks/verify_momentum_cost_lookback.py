"""Independent verification of `research/checks/momentum_cost_lookback.py`.

One verification gate:

1. Recompute cost impact (median reduction, edge preservation, verdict change) 
   via a separate implementation path for each lookback and each asset, comparing
   against the artifact. The artifact stores per-asset results directly, so this
   verification recomputes each result and confirms they match.

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

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

COST_ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_lookback_results.json"

REALISTIC_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
)

ZERO_COST_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result_with_cost(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker and one lookback with given config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def main():
    print("=== Independent verification of momentum cost-lookback sensitivity ===\n")
    
    # Load cost artifact
    with open(COST_ARTIFACT_PATH) as f:
        cost_artifact = json.load(f)
    
    # Verify dataset ID
    assert cost_artifact["dataset_id"] == DATASET_ID, f"Dataset ID mismatch"
    
    # Load manifest for ticker data
    manifest = load_manifest()
    
    print("=== 1. Recompute cost impact and verify artifact ===")
    # Recompute cost impact for each lookback
    all_recomputed_consistent = True
    
    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} ---")
        
        # Load full universe
        tickers = {}
        for entry in manifest["entries"]:
            ticker = entry["ticker"]
            tickers[ticker] = bt.load_ticker(ticker)[0]
        
        # Recompute cost impact for each asset
        for ticker in tickers.keys():
            # Recompute zero-cost results
            zero_results = segment_result_with_cost(ticker, tickers[ticker], lb, ZERO_COST_CFG)
            
            # Recompute realistic cost results
            realistic_results = segment_result_with_cost(ticker, tickers[ticker], lb, REALISTIC_CFG)
            
            # Calculate cost impact (same logic as main script)
            median_reduction = np.mean([
                realistic_results["medians"][i] - zero_results["medians"][i]
                for i in range(len(realistic_results["medians"]))
            ])
            
            edge_preserved = any(m > 0.05 for m in realistic_results["medians"])
            verdict_change = "NO_CHANGE" if realistic_results["verdict"] == zero_results["verdict"] else "CHANGED"
            
            # Get artifact data
            artifact_data = cost_artifact["cost_sensitivity"][str(lb)]["per_asset"][ticker]
            
            # Compare
            if (abs(artifact_data["median_reduction"] - median_reduction) > 0.001 or
                artifact_data["edge_preserved"] != edge_preserved or
                artifact_data["verdict_change"] != verdict_change):
                all_recomputed_consistent = False
                print(f"  RECOMPUTATION MISMATCH: {ticker}")
                print(f"    Median reduction: artifact={artifact_data['median_reduction']}, "
                      f"recomputed={round(median_reduction, 3)}")
                print(f"    Edge preserved: artifact={artifact_data['edge_preserved']}, "
                      f"recomputed={edge_preserved}")
                print(f"    Verdict change: artifact={artifact_data['verdict_change']}, "
                      f"recomputed={verdict_change}")
            else:
                print(f"  ✓ {ticker}: cost impact matches (median reduction={artifact_data['median_reduction']}, "
                      f"edge preserved={artifact_data['edge_preserved']}, verdict change={artifact_data['verdict_change']})")
    
    if all_recomputed_consistent:
        print("\n✓ All cost impact calculations match the artifact")
    else:
        print("\n✗ Cost impact calculation mismatches found")
        sys.exit(1)
    
    print("\n=== 2. Robustness check ===")
    # Verify robustness statistics
    robust_consistent = True
    
    for lb in LOOKBACKS:
        artifact_robust = cost_artifact["cost_sensitivity"][str(lb)]["robust_at_realistic"]
        artifact_total = cost_artifact["cost_sensitivity"][str(lb)]["total_assets"]
        
        # Recompute robustness
        tickers = {}
        for entry in manifest["entries"]:
            ticker = entry["ticker"]
            tickers[ticker] = bt.load_ticker(ticker)[0]
        
        recomputed_robust = 0
        for ticker in tickers.keys():
            # Recompute realistic results
            realistic_results = segment_result_with_cost(ticker, tickers[ticker], lb, REALISTIC_CFG)
            
            # Check if edge is preserved
            if any(m > 0.05 for m in realistic_results["medians"]):
                recomputed_robust += 1
        
        if artifact_robust != recomputed_robust:
            robust_consistent = False
            print(f"  ROBUSTNESS MISMATCH: lookback={lb}, artifact={artifact_robust}, "
                  f"recomputed={recomputed_robust}")
        else:
            print(f"  ✓ Lookback {lb}: robustness matches ({artifact_robust}/{artifact_total})")
    
    if robust_consistent:
        print("\n✓ All robustness statistics match the artifact")
    else:
        print("\n✗ Robustness statistic mismatches found")
        sys.exit(1)
    
    print("\n=== Verification Summary ===")
    print("✓ Cost impact recomputation: PASSED")
    print("✓ Robustness statistics: PASSED")
    print("\nAll verification gates passed. The cost-sensitive momentum lookback sweep results are valid.")
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-sensitive lookback independent verification.")


if __name__ == "__main__":
    main()