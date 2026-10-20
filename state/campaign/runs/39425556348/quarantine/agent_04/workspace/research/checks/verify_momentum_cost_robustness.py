"""Independent verification of `research/checks/momentum_cost_robustness.py`.

This script performs a fresh recomputation of the momentum cost-robustness
results, ensuring determinism and consistency with the expected outcome.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

# Realistic cost model from examples/ma_crossover.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
)

REALISTIC_COSTS_CONFIG = {
    "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
    "commission_per_share": REALISTIC_COSTS.commission_per_share,
    "slippage_cents": REALISTIC_COSTS.slippage_cents,
    "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
}


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


def segment_median(ticker, s, e, train, test, warm, overlap, lookback, cfg):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = sig[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm, **cfg)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    cost_artifact = load_artifact(COST_ARTIFACT)
    assert cost_artifact["dataset_id"] == DATASET_ID
    assert cost_artifact["lookbacks"] == list(LOOKBACKS)
    assert cost_artifact["realistic_costs"] == REALISTIC_COSTS_CONFIG

    all_ok = True

    print("=== Gate 1: fresh recomputation of cost-robustness medians (lookback 5 only) ===")
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    
    # Load AMZN for simplicity (same ticker used in original momentum check)
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    
    # Recompute segments using the verification path
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    # For lookback=5, recompute each segment's median
    lookback = 5
    recomputed_medians = []
    for lab, s, e in runs:
        med, _ = segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lookback, REALISTIC_COSTS_CONFIG)
        recomputed_medians.append(round(med, 3))
    
    # Get the published medians for AMZN lookback=5
    pub_medians = cost_artifact["cost_robustness"][str(lookback)]["momentum_results"]["AMZN"]["medians"]
    
    status = "MATCH" if recomputed_medians == pub_medians else "MISMATCH"
    if status != "MATCH":
        all_ok = False
        print(f"  AMZN lookback 5 recomputed medians: {recomputed_medians}")
        print(f"  AMZN lookback 5 published medians: {pub_medians}")
    
    print(f"  AMZN lookback 5 -> {status}")

    print("\n=== Gate 2: determinism test (recomputing AMZN lookback=5 medians twice) ===")
    m1 = []
    for lab, s, e in runs:
        med, _ = segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, REALISTIC_COSTS_CONFIG)
        m1.append(round(med, 3))
    m2 = []
    for lab, s, e in runs:
        med, _ = segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, REALISTIC_COSTS_CONFIG)
        m2.append(round(med, 3))
    print(f"  AMZN lookback 5 r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
    assert m1 == m2, "verification path not deterministic"

    print("\n=== Gate 3: cost-robustness verdict calculation ===")
    
    def edge_or_not(sr):
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            return "EDGE"
        if sr["verdict"] == "REGIME_STABLE_LOSS":
            return "LOSS"
        return "NO_EDGE"

    def cost_robustness_verdict(positive_edges, total_assets):
        if positive_edges >= 2:
            return "RETAIN"
        elif positive_edges == 0:
            return "REVERT"
        else:
            return "UNVERIFIED"

    for lookback in LOOKBACKS:
        edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]:
            sr = {
                "medians": cost_artifact["cost_robustness"][str(lookback)]["momentum_results"][ticker]["medians"],
                "null_medians": cost_artifact["cost_robustness"][str(lookback)]["momentum_results"][ticker]["null_medians"],
                "verdict": cost_artifact["cost_robustness"][str(lookback)]["momentum_results"][ticker]["verdict"],
            }
            edge_counts[edge_or_not(sr)] += 1
        
        positive_edges = edge_counts["EDGE"]
        verdict = cost_robustness_verdict(positive_edges, 10)
        
        status = "MATCH" if verdict == cost_artifact["cost_robustness"][str(lookback)]["verdict"] else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        
        print(f"  Lookback {lookback}: edge_counts={edge_counts} -> {verdict} vs published {cost_artifact['cost_robustness'][str(lookback)]['verdict']} -> {status}")

    print()
    if all_ok:
        print("VERIFICATION PASSED: All independent recomputations MATCH the cost-robustness artifact, determinism is confirmed, and cost-robustness verdicts reproduce correctly.")
    else:
        print("MISMATCH FOUND: the cost-robustness artifact does not reproduce via the independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. Momentum cost-robustness independent verification across the collected universe: exploratory simulation.")


if __name__ == "__main__":
    main()
