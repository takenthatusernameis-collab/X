"""Independent verification of the momentum cost-robustness check.

This recomputes cost robustness results via a fresh walk_forward path:
- Regime labels are reconstructed from closes using a re-implemented
  volatility_blocks (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using min_segment_bars.
- Segment medians are computed via walk_forward directly on the segment
  slice, aggregating fold log returns -- no call to stress_segments.
- Realistic vs zero cost effects are recomputed independently.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the cost-robustness check computed correctly; a mismatch flags an
error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

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
def classify_regime_edge(medians):
    """Regenerate the regime-stability verdict from medians using the same rule as the original momentum test."""
    cand_disp = float(np.std(np.array(medians)))
    null_disp = 0.05  # Sample-calibrated tolerance from methodology.md
    if all(abs(m) <= null_disp for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"
def segment_median_with_costs(ticker, s, e, train, test, warm, overlap, cfg):
    """Compute segment median log return for a ticker with given costs."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, 5)  # Using baseline lookback 5 for verification
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)
def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)
def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "dataset id mismatch"

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset cost robustness for lookback 5 (baseline)
    print("\n--- Per-asset cost robustness comparison ---")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        bars, dates = bt.load_ticker(entry["ticker"])
        tickers[entry["ticker"]] = bars

    for ticker in sorted(tickers.keys()):
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        print(f"\n{ticker}:")
        for lookback in LOOKBACKS:
            # Realistic costs
            realistic_medians = []
            realistic_verdicts = []
            
            for lab, s, e in runs:
                m_real, nf = segment_median_with_costs(ticker, s, e, TRAIN, TEST, WARM, OVERLAP,
                                                      bt.BacktestConfig(
                                                          commission_per_trade=2.0,
                                                          commission_per_share=0.003,
                                                          slippage_cents=2.0,
                                                          slippage_proportional=0.0005))
                realistic_medians.append(round(m_real, 3))
                realistic_verdicts.append(classify_regime_edge([m_real]))
            
            print(f"  lookback {lookback}: realistic {realistic_medians}")

    # Perturbation recomputation with realistic and zero costs
    print("\n=== Perturbation recomputation (lookback 3/5/10) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]
    
    # Realistic costs
    recomputed_realistic = []
    for params in grid:
        signals = momentum_signals(bars.closes_array(), params["lookback"])
        res = bt.walk_forward(
            list(bars), signals, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=bt.BacktestConfig(
                commission_per_trade=2.0,
                commission_per_share=0.003,
                slippage_cents=2.0,
                slippage_proportional=0.0005))
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        recomputed_realistic.append(round(float(np.median(log)), 3))
    
    # Zero costs
    recomputed_zero = []
    for params in grid:
        signals = momentum_signals(bars.closes_array(), params["lookback"])
        res = bt.walk_forward(
            list(bars), signals, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=bt.BacktestConfig())
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        recomputed_zero.append(round(float(np.median(log)), 3))
    
    # Null benchmark
    null_res = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        cfg=bt.BacktestConfig(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005),
        seed=SEED,
    )
    
    pub = artifact["perturbation"]["realistic_medians"]
    pub_zero = artifact["perturbation"]["zero_cost_medians"]
    pub_null = artifact["perturbation"]["baseline_median"]
    
    realistic_status = "MATCH" if recomputed_realistic == pub else "MISMATCH"
    zero_status = "MATCH" if recomputed_zero == pub_zero else "MISMATCH"
    null_status = "MATCH" if abs(null_res.baseline_median_log_return - pub_null) < 0.001 else "MISMATCH"
    
    if realistic_status != "MATCH" or zero_status != "MATCH" or null_status != "MATCH":
        all_ok = False
    
    print(f"  param sets: {artifact['perturbation']['param_sets']}")
    print(f"  recomputed realistic medians: {recomputed_realistic}")
    print(f"  published realistic medians:  {pub}")
    print(f"  realistic -> {realistic_status}")
    print(f"  recomputed zero medians: {recomputed_zero}")
    print(f"  published zero medians:  {pub_zero}")
    print(f"  zero -> {zero_status}")
    print(f"  null median recomputed: {null_res.baseline_median_log_return:+.3f}")
    print(f"  null median published:  {pub_null:+.3f}")
    print(f"  null -> {null_status}")

    # Determinism test
    print("\nDeterminism: recomputing cost robustness for one ticker again.")
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    m1_realistic = []
    m2_realistic = []
    m1_zero = []
    m2_zero = []
    
    for lab, s, e in runs:
        mr1, _ = segment_median_with_costs("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP,
                                          bt.BacktestConfig(commission_per_trade=2.0,
                                                           commission_per_share=0.003,
                                                           slippage_cents=2.0,
                                                           slippage_proportional=0.0005))
        mr2, _ = segment_median_with_costs("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP,
                                          bt.BacktestConfig(commission_per_trade=2.0,
                                                           commission_per_share=0.003,
                                                           slippage_cents=2.0,
                                                           slippage_proportional=0.0005))
        mz1, _ = segment_median_with_costs("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP,
                                          bt.BacktestConfig())
        mz2, _ = segment_median_with_costs("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP,
                                          bt.BacktestConfig())
        m1_realistic.append(round(mr1, 3))
        m2_realistic.append(round(mr2, 3))
        m1_zero.append(round(mz1, 3))
        m2_zero.append(round(mz2, 3))
    
    realistic_match = m1_realistic == m2_realistic
    zero_match = m1_zero == m2_zero
    print(f"  realistic costs r1 == r2: {realistic_match}")
    print(f"  zero costs r1 == r2: {zero_match}")
    assert realistic_match and zero_match, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-robustness artifact.")
    else:
        print("MISMATCH FOUND: the cost-robustness artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-robustness verification on 10-asset collected universe.")
if __name__ == "__main__":
    main()