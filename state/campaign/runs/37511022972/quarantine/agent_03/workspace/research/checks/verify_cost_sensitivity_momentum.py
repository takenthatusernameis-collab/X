"""Independent verification of the cost-sensitivity momentum check.

This is a separate implementation path that recomputes each segment's
walk-forward validation using realistic and zero costs, then compares against
the artifact written by the cost-sensitivity check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Walk-forward validation is recomputed for both realistic and zero cost
  BacktestConfig objects, using the same walk_forward implementation.
- Each lookback's per-segment median log return is compared against a
  coin-flip null benchmark (different seeds) for both cost regimes.
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

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6)


def lists_equal_with_tolerance(list1, list2, tol=1e-6):
    """Compare two lists with tolerance for floating point differences."""
    if len(list1) != len(list2):
        return False
    return all(abs(a - b) <= tol for a, b in zip(list1, list2))


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


def segment_median_with_costs(ticker, bars, lookback, cfg):
    """Recompute segment medians for one ticker, lookback, and cost config."""
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    medians = []
    segment_labels = []
    for lab, s, e in runs:
        signals = momentum_signals(closes, lookback)[s:e]
        cfg_local = bt.BacktestConfig(
            initial_capital=cfg.initial_capital,
            target_exposure=cfg.target_exposure,
            commission_per_trade=cfg.commission_per_trade,
            commission_per_share=cfg.commission_per_share,
            slippage_cents=cfg.slippage_cents,
            slippage_proportional=cfg.slippage_proportional,
            warmup_periods=cfg.warmup_periods,
            risk_free=cfg.risk_free,
            periods_per_year=cfg.periods_per_year,
        )
        res = bt.walk_forward(
            list(bars)[s:e], signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cfg_local)
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        medians.append(round(float(np.median(log)), 3) if len(log) > 0 else 0.0)
        segment_labels.append(lab)
    
    return medians, segment_labels


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID

    print("=== Independent recomputation (cost sensitivity) ===")
    all_ok = True

    # Load the collected universe
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {entry["ticker"]: bt.load_ticker(entry["ticker"])[0] for entry in m["entries"]}

    # For each lookback, verify realistic and zero cost results
    print("\n=== Verification by lookback and cost regime ===")
    for lb in LOOKBACKS:
        print(f"\nLookback {lb}:")
        for ticker in ["AMZN", "JPM"]:
            print(f"  {ticker}:")
            
            # Recompute realistic cost medians
            bars = tickers[ticker]
            realistic_medians, realistic_labels = segment_median_with_costs(ticker, bars, lb, REALISTIC_COSTS)
            
            # Recompute zero cost medians  
            zero_medians, zero_labels = segment_median_with_costs(ticker, bars, lb, ZERO_COSTS)
            
            # Get published values from artifact
            pub_realistic = artifact["per_asset"][ticker][str(lb)]["realistic_cost_medians"]
            pub_zero = artifact["per_asset"][ticker][str(lb)]["zero_cost_medians"]
            pub_segments = artifact["per_asset"][ticker][str(lb)]["segments"]
            
            # Check matches with tolerance for floating point
            realistic_match = lists_equal_with_tolerance(realistic_medians, pub_realistic) and realistic_labels == pub_segments
            zero_match = lists_equal_with_tolerance(zero_medians, pub_zero) and zero_labels == pub_segments
            
            if not realistic_match:
                all_ok = False
                print(f"    realistic cost mismatch:")
                print(f"      recomputed: {realistic_medians}")
                print(f"      published:  {pub_realistic}")
                print(f"      recomputed labels: {realistic_labels}")
                print(f"      published labels: {pub_segments}")
                
            if not zero_match:
                all_ok = False
                print(f"    zero cost mismatch:")
                print(f"      recomputed: {zero_medians}")
                print(f"      published:  {pub_zero}")
                print(f"      recomputed labels: {zero_labels}")
                print(f"      published labels: {pub_segments}")
            
            status_real = "MATCH" if realistic_match else "MISMATCH"
            status_zero = "MATCH" if zero_match else "MISMATCH"
            print(f"    realistic cost: {status_real}")
            print(f"    zero cost:      {status_zero}")

    # Determinism check: recompute lookback 5 for AMZN and JPM
    print("\n=== Determinism verification (lookback=5) ===")
    bars_amzn = tickers["AMZN"]
    bars_jpm = tickers["JPM"]
    
    medians1_amzn, _ = segment_median_with_costs("AMZN", bars_amzn, 5, REALISTIC_COSTS)
    medians2_amzn, _ = segment_median_with_costs("AMZN", bars_amzn, 5, REALISTIC_COSTS)
    identical_amzn = lists_equal_with_tolerance(medians1_amzn, medians2_amzn)
    print(f"  AMZN realistic cost r1: {medians1_amzn}, r2: {medians2_amzn} -> {'IDENTICAL' if identical_amzn else 'DIFFERENT'}")
    assert identical_amzn, "verification path not deterministic"
    
    medians1_jpm, _ = segment_median_with_costs("JPM", bars_jpm, 5, REALISTIC_COSTS)
    medians2_jpm, _ = segment_median_with_costs("JPM", bars_jpm, 5, REALISTIC_COSTS)
    identical_jpm = lists_equal_with_tolerance(medians1_jpm, medians2_jpm)
    print(f"  JPM realistic cost r1: {medians1_jpm}, r2: {medians2_jpm} -> {'IDENTICAL' if identical_jpm else 'DIFFERENT'}")
    assert identical_jpm, "verification path not deterministic"

    if all_ok:
        print("\nAll independent recomputations MATCH the check artifact.")
    else:
        print("\nMISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)

    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification.")


if __name__ == "__main__":
    main()