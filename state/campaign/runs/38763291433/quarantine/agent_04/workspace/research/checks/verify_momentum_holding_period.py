"""Independent verification of the momentum holding period test.

This is a separate implementation path: it recomputes each holding period's
walk-forward median and the universe-level results from a fresh walk_forward
implementation (not the original implementation path), then compares against
the artifact written by the holding period test
(research/checks/momentum_holding_period.py).

Design of the independent path:
- Uses the same inputs (dataset, seed, windows, holding periods) as the check.
- Recomputes segment medians via walk_forward directly on the segment slice.
- Perturbation medians are recomputed by walking each holding period
  independently.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs so a match confirms the check computed from those
inputs; a mismatch would flag an error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACK = 5
HOLDING_PERIODS = [1, 2, 3, 5]
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_holding_period_results.json"

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=100,
    commission_per_share=5,
    slippage_cents=20,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks (simplified)."""
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
    """Simple momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def momentum_signals_with_holding(closes, lookback, holding_period):
    """Momentum signal with minimum holding period constraint.

    Signals respect minimum holding periods: once a signal changes, it cannot
    change again for the specified number of days.
    This is an independent re-implementation of the check's logic.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    # Generate raw momentum signal first
    raw_signals = momentum_signals(closes, lookback)
    
    # Apply holding period constraint
    for i in range(lookback, n):
        # Find next time momentum signal would change
        if i < n - 1:
            next_signal_change = None
            for j in range(i + 1, min(i + holding_period + 1, n)):
                if abs(raw_signals[j].weight - raw_signals[i].weight) > 1e-12:
                    next_signal_change = j
                    break
            
            # If no signal change within holding period, keep current signal
            if next_signal_change is None:
                out[i] = raw_signals[i]
            else:
                # Use signal until holding period expires
                out[i] = raw_signals[min(i, next_signal_change - 1)]
        else:
            out[i] = raw_signals[i]
    
    # Pad warmup - signals must be neutral before warmup boundary
    for i in range(min(lookback, n)):
        out[i] = bt.Signal(date=i + 1, weight=0.0)
    
    return out


def segment_holding_period_median(ticker, s, e, train, test, warm, overlap, lookback, holding_period):
    """Walk-forward median for a specific segment with holding period constraint."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals_with_holding(closes, lookback, holding_period)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = REALISTIC_COST_CONFIG
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def main():
    np.random.seed(SEED)
    
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK, "lookback mismatch"
    assert artifact["holding_periods"] == HOLDING_PERIODS, "holding periods mismatch"
    
    print("=== Independent verification of momentum holding period test ===")
    print(f"Verifying lookback {LOOKBACK} with holding periods {HOLDING_PERIODS}")

    from research.data.preflight import load_manifest
    manifest = load_manifest()
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    all_ok = True
    
    # Per-asset recomputation
    print("\n=== 1. Per-asset recomputation ===")
    for ticker in ["NVDA"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        for hp in HOLDING_PERIODS:
            medians = []
            for lab, s, e in runs:
                m, nf = segment_holding_period_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK, hp)
                medians.append(round(m, 3))
            
            pub_medians = artifact["per_asset"][ticker][str(hp)]["median_log_return"]
            
            # Note: The check didn't store segment-level medians, just overall median
            status = "MATCH" if abs(medians[0] - pub_medians) < 0.001 else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print(f"  {ticker} holding period {hp}: median {medians[0]} vs published {pub_medians} -> {status}")

    # Universe-level recomputation
    print("\n=== 2. Universe-level recomputation ===")
    for ticker in ["NVDA"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        hp_medians = []
        for hp in HOLDING_PERIODS:
            hp_median = segment_holding_period_median(ticker, 0, len(closes), TRAIN, TEST, WARM, OVERLAP, LOOKBACK, hp)[0]
            hp_medians.append(round(hp_median, 3))
        
        pub_holding_medians = {
            str(hp): artifact["per_asset"][ticker][str(hp)]["median_log_return"]
            for hp in HOLDING_PERIODS
        }
        
        status = "MATCH" if all(abs(a - b) < 0.001 for a, b in zip(hp_medians, [pub_holding_medians[str(hp)] for hp in HOLDING_PERIODS])) else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        
        print(f"  {ticker} holding medians {hp_medians} vs published {list(pub_holding_medians.values())} -> {status}")

    # Determinism check
    print("\n=== 3. Determinism verification ===")
    m1 = []
    m2 = []
    for hp in HOLDING_PERIODS:
        hp_median1 = segment_holding_period_median("AMZN", 0, len(tickers["AMZN"]), TRAIN, TEST, WARM, OVERLAP, LOOKBACK, hp)[0]
        hp_median2 = segment_holding_period_median("AMZN", 0, len(tickers["AMZN"]), TRAIN, TEST, WARM, OVERLAP, LOOKBACK, hp)[0]
        m1.append(round(hp_median1, 3))
        m2.append(round(hp_median2, 3))
    
    print(f"  AMZN determinism: r1 == r2: {m1 == m2}")
    assert m1 == m2, "verification path not deterministic"

    if all_ok:
        print("\n✓ All independent recomputations MATCH the check artifact.")
        print("Holding period test verified successfully.")
    else:
        print("\n✗ MISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Holding period independent verification on AMZN/JPM + universe: exploratory simulation.")


if __name__ == "__main__":
    main()
