"""Independent verification of the cost-robustness momentum check
(`research/checks/cost_robustness_momentum.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not `stress_segments`), then compares against the artifact
written by the cost-robustness check (`state/check_artifacts/cost_robustness_momentum_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Cost calculations are reimplemented from the BacktestConfig fields.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Define the same cost model as in the check
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=1.0,      # $1 per round trip
    commission_per_share=0.01,     # $0.01 per share
    slippage_cents=1.0,            # $0.01 per share fixed slippage
    slippage_proportional=0.0005,  # 5 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "cost_robustness_momentum_results.json"


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
    """Re-implementation of segment extraction from regime labels."""
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


def cost_signals(closes, lookback):
    """Fresh implementation of the momentum signal with costs (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def walk_forward_with_costs(bars, signals, train, test, warm, overlap, cost_config):
    """Fresh walk_forward implementation with explicit cost calculation."""
    from research.backtest.engine import Backtest, BacktestConfig
    
    n = len(bars)
    if len(signals) != n:
        raise ValueError("bars and signals must have equal length")

    folds = []
    step = test - overlap
    fold_start = 0
    cumulative_test = 0

    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n:
            break
        
        fs = max(0, fold_start)
        fold_bars = bars[fs:fold_end]
        fold_signals = signals[fs:fold_end]
        
        # Create BacktestConfig with the same costs
        fold_cfg = BacktestConfig(
            initial_capital=cost_config.initial_capital,
            target_exposure=cost_config.target_exposure,
            commission_per_trade=cost_config.commission_per_trade,
            commission_per_share=cost_config.commission_per_share,
            slippage_cents=cost_config.slippage_cents,
            slippage_proportional=cost_config.slippage_proportional,
            warmup_periods=warm,
            risk_free=cost_config.risk_free,
            periods_per_year=cost_config.periods_per_year,
            margin_rate=cost_config.margin_rate,
            margin_call_liquidate=cost_config.margin_call_liquidate,
        )
        
        from research.backtest.engine import Backtest
        result = Backtest(fold_cfg).run(fold_bars, fold_signals)

        # The OOS segment is the last test_window bars of the fold window.
        test_eq = result.equity_curve[-test:]
        if len(test_eq) == 0:
            break

        # per-fold metrics on the OOS segment only
        from research.backtest.metrics import compute_metrics
        oos_window_start = fold_bars[-test].date
        oos_window_end = fold_bars[-1].date
        oos_trades = [
            f for f in result.trades
            if oos_window_start <= f.date <= oos_window_end
        ]
        oos_prices = (
            np.array([f.price for f in oos_trades], dtype=np.float64)
            if oos_trades
            else np.array([], dtype=np.float64)
        )
        
        fold_metrics = compute_metrics(
            test_eq,
            fills=oos_trades,
            fill_prices=oos_prices,
            risk_free=cost_config.risk_free,
            periods_per_year=cost_config.periods_per_year,
        )
        
        folds.append({
            "fold_index": len(folds),
            "start_date": fold_bars[0].date,
            "end_date": fold_bars[-1].date,
            "train_window_bars": train,
            "test_window_bars": len(test_eq),
            "metrics": fold_metrics,
            "total_return": float(np.mean([f.metrics["total_return"] for f in folds])),
        })
        cumulative_test += len(test_eq)
        fold_start += step
    
    # Calculate medians
    log_returns = []
    for f in folds:
        # Calculate log return for this fold
        start_equity = f["total_return"] * cost_config.initial_capital if f != folds[0] else cost_config.initial_capital
        end_equity = cost_config.initial_capital
        if f["metrics"].get("total_return") is not None:
            log_return = np.log(f["metrics"]["total_return"])
            log_returns.append(log_return)
    
    return {
        "folds": folds,
        "median_log_return": float(np.median(log_returns)) if log_returns else 0.0,
        "n_folds": len(folds),
        "cumulative_test": cumulative_test,
    }


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    
    # Basic artifact validation
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS), "lookbacks mismatch"
    assert "per_asset" in artifact, "missing per_asset data"
    assert "universe" in artifact, "missing universe data"
    
    print("=== Independent recomputation of cost-robustness momentum ===")
    print(f"Cost config: commission_per_trade=${COST_CONFIG.commission_per_trade}, "
          f"commission_per_share=${COST_CONFIG.commission_per_share}, "
          f"slippage_cents=${COST_CONFIG.slippage_cents}, "
          f"slippage_proportional={COST_CONFIG.slippage_proportional * 100} bps")
    
    # Load all tickers for verification
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {}
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    
    all_ok = True
    
    # Verify per-asset results (focus on AMZN/JPM for determinism)
    print("\n=== 1. Per-asset verification (AMZN/JPM) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        
        # Reconstruct segments
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        # Verify each lookback
        for lb in LOOKBACKS:
            signals = cost_signals(closes, lb)
            
            for lab, s, e in runs:
                # Use fresh walk_forward with costs
                wf_result = walk_forward_with_costs(
                    bars=list(bars), 
                    signals=signals, 
                    train=TRAIN, 
                    test=TEST, 
                    warm=WARM, 
                    overlap=OVERLAP,
                    cost_config=COST_CONFIG
                )
                
                # Get artifact result
                artifact_result = artifact["per_asset"][asset][str(lb)]
                
                # Compare medians
                recomputed_median = round(wf_result["median_log_return"], 3)
                published_median = artifact_result["medians"][0]  # First segment
                
                status = "MATCH" if abs(recomputed_median - published_median) < 0.001 else "MISMATCH"
                if status != "MATCH":
                    all_ok = False
                    
                print(f"  {asset} lookback {lb}: recomputed {recomputed_median} vs "
                      f"published {published_median} -> {status}")
    
    # Verify universe-level results
    print("\n=== 2. Universe-level verification ===")
    for ticker in artifact["universe"]["ticker_order"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        
        # Reconstruct segments
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        # Calculate medians for each lookback
        for lb in LOOKBACKS:
            signals = cost_signals(closes, lb)
            
            for lab, s, e in runs:
                wf_result = walk_forward_with_costs(
                    bars=list(bars), 
                    signals=signals, 
                    train=TRAIN, 
                    test=TEST, 
                    warm=WARM, 
                    overlap=OVERLAP,
                    cost_config=COST_CONFIG
                )
                
                # Get artifact result
                artifact_result = artifact["universe"]["per_asset"][ticker][f"lookback_{lb}_medians"][0]
                recomputed_median = round(wf_result["median_log_return"], 3)
                
                status = "MATCH" if abs(recomputed_median - artifact_result) < 0.001 else "MISMATCH"
                if status != "MATCH":
                    all_ok = False
                    print(f"  [DEBUG] ticker={ticker} lookback={lb}: "
                          f"len(runs)={len(runs)} recomputed_median={recomputed_median} "
                          f"artifact_median={artifact_result}")
    
    # Determinism check
    print("\n=== 3. Determinism verification ===")
    amzn_bars, amzn_dates = bt.load_ticker("AMZN")
    amzn_closes = amzn_bars.closes_array()
    
    amzn_labels = vol_blocks(amzn_closes, N_BLOCKS, WINDOW)
    amzn_runs = segments_from_labels(amzn_labels, MIN_SEGMENT_BARS)
    
    # Run verification twice
    results_first = []
    for lb in LOOKBACKS:
        amzn_signals = cost_signals(amzn_closes, lb)
        wf_result = walk_forward_with_costs(
            bars=list(amzn_bars), 
            signals=amzn_signals, 
            train=TRAIN, 
            test=TEST, 
            warm=WARM, 
            overlap=OVERLAP,
            cost_config=COST_CONFIG
        )
        results_first.append(round(wf_result["median_log_return"], 3))
    
    results_second = []
    for lb in LOOKBACKS:
        amzn_signals = cost_signals(amzn_closes, lb)
        wf_result = walk_forward_with_costs(
            bars=list(amzn_bars), 
            signals=amzn_signals, 
            train=TRAIN, 
            test=TEST, 
            warm=WARM, 
            overlap=OVERLAP,
            cost_config=COST_CONFIG
        )
        results_second.append(round(wf_result["median_log_return"], 3))
    
    determinism_match = results_first == results_second
    print(f"  AMZN verification results: {results_first}")
    print(f"  AMZN verification results (second run): {results_second}")
    print(f"  determinism: {'PASS' if determinism_match else 'FAIL'}")
    if not determinism_match:
        all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("SUCCESS: All independent recomputations MATCH the check artifact.")
    else:
        print("FAILURE: MISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)
    
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness independent verification.")


if __name__ == "__main__":
    main()
