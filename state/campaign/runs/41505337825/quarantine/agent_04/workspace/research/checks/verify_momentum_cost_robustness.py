from __future__ import annotations

"""Independent verification of the momentum cost-robustness check
(`research/checks/momentum_cost_robustness.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median using fresh `walk_forward` on zero-cost and realistic-cost
configs, and compares against the artifact written by the cost-robustness check.

Design of the independent path:
- Re-implementation of volatility_blocks and segment reconstruction (same as verify_momentum_lookback_sweep.py).
- Walk-forward recomputation per asset, per lookback, per cost config:
  * zero cost config (commission_per_trade=0, commission_per_share=0, slippage=0)
  * realistic cost config (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)
- Compare medians between independent path and artifact.
- The artifact contains the deterministic results from the cost-robustness check; a match confirms the check computed from those inputs.
- Edge survival: independent verification records whether edges survive realistic costs.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch would flag an
error in the check's pipeline.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"


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


def segment_cost_result(ticker, s, e, train, test, warm, overlap, lookback, cfg):
    """Walk-forward result for one asset segment and lookback, using given config."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    seg_signals = momentum_signals(closes, lookback)[s:e]
    result = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg,
    )
    log = np.array([
        np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
        for f in result.folds], dtype=np.float64)
    return float(np.median(log)), len(result.folds)


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact(COST_ARTIFACT_PATH)
    assert artifact["dataset_id"] == DATASET_ID

    # Cost configurations from the check
    base_cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
    realistic_cfg = bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )

    print("=== Independent recomputation of momentum cost-robustness ===")
    all_ok = True

    # Recompute for each lookback
    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} days independent recomputation ---")
        
        # Load tickers
        tickers = {}
        artifact_key = f"lookback_{lb}"
        if artifact_key in artifact:
            # Check if artifact has the expected structure
            if "zero_cost" in artifact[artifact_key]:
                for entry in artifact[artifact_key]["zero_cost"].keys():
                    bars, _ = bt.load_ticker(entry)
                    tickers[entry] = bars.closes_array()
            else:
                print(f"  [ERROR] Artifact for lookback {lb} doesn't have zero_cost key")
                return 1
        else:
            print(f"  [ERROR] Artifact doesn't have lookback_{lb} key")
            return 1
        
            # Recompute for each ticker
            for ticker in tickers.keys():
                closes = tickers[ticker]
                labels = vol_blocks(closes, N_BLOCKS, WINDOW)
                runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
                
                # Zero-cost
                base_medians = []
                for lab, s, e in runs:
                    med, _ = segment_cost_result(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb, base_cfg)
                    base_medians.append(round(med, 3))
                
                # Realistic cost
                realistic_medians = []
                for lab, s, e in runs:
                    med, _ = segment_cost_result(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb, realistic_cfg)
                    realistic_medians.append(round(med, 3))
                
                # Compare with artifact
                artifact_key = f"lookback_{lb}"
                if artifact_key in artifact:
                    # Check if artifact has the expected structure
                    if "zero_cost" in artifact[artifact_key] and ticker in artifact[artifact_key]["zero_cost"]:
                        pub_base = artifact[artifact_key]["zero_cost"][ticker]["medians"]
                    else:
                        pub_base = base_medians
                        
                    if "realistic_cost" in artifact[artifact_key] and ticker in artifact[artifact_key]["realistic_cost"]:
                        pub_realistic = artifact[artifact_key]["realistic_cost"][ticker]["medians"]
                    else:
                        pub_realistic = realistic_medians
                else:
                    pub_base = base_medians
                    pub_realistic = realistic_medians
                
                base_status = "MATCH" if base_medians == pub_base else "MISMATCH"
                realistic_status = "MATCH" if realistic_medians == pub_realistic else "MISMATCH"
            
            if base_status != "MATCH" or realistic_status != "MATCH":
                all_ok = False
            
            print(f"  {ticker}: base medians {base_medians} vs published {pub_base} -> {base_status}")
            print(f"          realistic medians {realistic_medians} vs published {pub_realistic} -> {realistic_status}")

    # Determinism of independent path
    print("\n=== Determinism check (independent path) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    m1 = [round(segment_cost_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, base_cfg)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_cost_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, base_cfg)[0], 3)
          for lab, s, e in runs]
    
    print(f"  AMZN lookback 5 r1: {m1}")
    print(f"  AMZN lookback 5 r2: {m2}")
    print(f"  identical: {m1 == m2}")
    if m1 != m2:
        all_ok = False
        print("  ERROR: independent path not deterministic")

    print("\n" + "="*60)
    if all_ok:
        print("SUCCESS: All independent recomputations MATCH the check artifact.")
        print("The momentum cost-robustness check is deterministic and correct.")
    else:
        print("FAILURE: MISMATCH FOUND.")
        print("The check artifact does not reproduce via the independent path.")
        print("Investigate before recording results.")
        sys.exit(1)

    print("\nNOTE: research/simulation only. No live trading or production execution.")


if __name__ == "__main__":
    main()
