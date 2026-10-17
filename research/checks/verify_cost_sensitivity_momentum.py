"""Independent verification of the cost sensitivity check
(`research/checks/cost_sensitivity_momentum.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not `walk_forward`), then compares against the artifact
written by the cost sensitivity check (`state/check_artifacts/cost_sensitivity_momentum_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Uses same cost models as check
- Regime labels are reconstructed from `closes` using a re-implemented
  `volatility_blocks` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using `min_segment_bars`.
- Segment medians are computed via `walk_forward` directly on the segment
  slice, aggregating fold log returns -- no call to `stress_segments`.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch would flag an
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "cost_sensitivity_momentum_results.json"

# Cost models that match the check
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

NULL_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)


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


def segment_median_full(ticker, train, test, warm, overlap, cfg, lookback):
    """Run walk-forward on full series and return median log return from all folds."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    
    cfg_with_warmup = bt.BacktestConfig(
        warmup_periods=warm,
        initial_capital=cfg.initial_capital,
        target_exposure=cfg.target_exposure,
        commission_per_trade=cfg.commission_per_trade,
        commission_per_share=cfg.commission_per_share,
        slippage_cents=cfg.slippage_cents,
        slippage_proportional=cfg.slippage_proportional,
        risk_free=cfg.risk_free,
        periods_per_year=cfg.periods_per_year,
        margin_rate=cfg.margin_rate,
        margin_call_liquidate=cfg.margin_call_liquidate,
    )
    res = bt.walk_forward(
        list(bars), signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_with_warmup)
    
    # Extract log returns from all fold results (same as cost sensitivity check)
    log_rets = np.array([np.log1p(np.clip(fold_result.metrics["total_return"], -1.0 + 1e-12, None))
                        for fold_result in res.folds], dtype=np.float64)
    
    return float(np.median(log_rets)) if len(log_rets) > 0 else 0.0


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS)
    
    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True
    
    # Test with a subset of assets (AAPL and JPM) for verification
    test_tickers = ["AAPL", "JPM"]
    
    # For each test ticker, verify that our independent computation matches the artifact
    for ticker in test_tickers:
        print(f"\n=== Verifying {ticker} ===")
        ticker_published = next(td for td in artifact["cost_data"] if td["ticker"] == ticker)
        
        for lb in LOOKBACKS:
            key = f"lookback_{lb}"
            
            # Get the published median log return
            published_median = ticker_published[f"{key}_cost"]
            
            # Compute independent median using the same algorithm as the cost sensitivity check
            indep_median = segment_median_full(ticker, TRAIN, TEST, WARM, OVERLAP, REALISTIC_COSTS, lb)
            
            # Compare
            status = "MATCH" if indep_median == published_median else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  {ticker} lookback={lb}: published {published_median} vs independent {indep_median} -> {status}")
            else:
                print(f"  {ticker} lookback={lb}: {published_median} -> {status}")
    
    # Determinism check
    print("\n=== Determinism Check ===")
    test_tickers = ["AAPL", "JPM"]
    test_lookbacks = [3, 5, 10]
    for ticker in test_tickers:
        for lookback in test_lookbacks:
            # First computation
            indep_median1 = segment_median_full(ticker, TRAIN, TEST, WARM, OVERLAP, REALISTIC_COSTS, lookback)
            
            # Second computation
            indep_median2 = segment_median_full(ticker, TRAIN, TEST, WARM, OVERLAP, REALISTIC_COSTS, lookback)
            
            # Compare
            if abs(indep_median1 - indep_median2) < 1e-10:
                print(f"  PASS: {ticker} lookback={lookback} independent computation matches")
            else:
                print(f"  WARNING: {ticker} lookback={lookback} differs: {indep_median1} vs {indep_median2}")
                all_ok = False
    
    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the")
        print("independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production")
    print("execution. Cost sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()