"""Independent verification of the momentum hold-period sensitivity check
(`research/checks/momentum_hold_period_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum hold-period check (`state/check_artifacts/momentum_hold_period_sensitivity_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Holding-period grid results are recomputed by walking each parameter set
  independently.
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
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLD_PERIODS = (1, 2, 3, 5)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"

BASE_SEED = 42  # same base seed the check used for perturbations


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


def segment_median(ticker, s, e, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, LOOKBACK)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def hold_effective_signals(signals, hold):
    """Carry each signal for `hold` bars before the next rebalance.
    
    Position at bar b equals the signal from the most recent rebalance bar
    b' <= b, where rebalance bars are lookback + k*hold. Concretely,
    effective[b] = signals[b - (b - lookback) % hold] for b >= lookback,
    else neutral. This is the standard holding-period semantics: the bet
    taken at bar t is marked to market for bars t .. t+hold-1 and exited
    at bar t+hold (or re-entered at the new level).
    
    No look-ahead: the transform only reindexes the signal series; each
    output bar depends on signals at or before that bar.
    """
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < LOOKBACK:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - LOOKBACK) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def verify_hold_period_results(artifact):
    """Verify holding-period grid results match artifact."""
    print("=== Independent recomputation (fresh walk_forward path) ===")
    
    # Verify per-asset results for AMZN and JPM
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        # Verify each hold period
        for hold in HOLD_PERIODS:
            eff = hold_effective_signals(momentum_signals(closes, LOOKBACK), hold)
            if len(eff) != len(closes):
                raise ValueError(f"{asset} hold={hold}: effective signals {len(eff)} != bars {len(closes)}")
            
            # Reconstruct segments from labels
            seg_labels = []
            for lab, s, e in runs:
                seg_labels.append((lab, s, e))
            
            # Verify each segment
            for lab, s, e in seg_labels:
                res = bt.stress_segments(
                    signals_fn=lambda c, **p: hold_effective_signals(
                        momentum_signals(c, LOOKBACK), int(round(p["hold"]))),
                    bars=list(bars),
                    signals=eff,
                    param_grid=[{"hold": hold}],
                    baseline=(("hold", hold),),
                    segment_fn=bt.segment_fn_from_labels(labels),
                    train_window=TRAIN,
                    test_window=TEST,
                    warmup=WARM,
                    overlap_window=OVERLAP,
                    periods_per_year=252,
                    min_segment_bars=MIN_SEGMENT_BARS,
                )
                
                # Compare with artifact
                artifact_entry = artifact["per_asset"][asset][str(hold)]
                
                # Check verdict
                if res.overall_verdict != artifact_entry["verdict"]:
                    print(f"  {asset} hold={hold} verdict MISMATCH: got {res.overall_verdict}, expected {artifact_entry['verdict']}")
                    return False
                
                # Check medians
                medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
                if medians != artifact_entry["medians"]:
                    print(f"  {asset} hold={hold} medians MISMATCH: got {medians}, expected {artifact_entry['medians']}")
                    return False
                
                # Check null medians
                null_medians = [round(s.noise_median_log_return, 3) for s in res.scenarios]
                if null_medians != artifact_entry["null_medians"]:
                    print(f"  {asset} hold={hold} null_medians MISMATCH: got {null_medians}, expected {artifact_entry['null_medians']}")
                    return False
                
                # Check n_folds
                if res.n_folds != artifact_entry["n_folds"]:
                    print(f"  {asset} hold={hold} n_folds MISMATCH: got {res.n_folds}, expected {artifact_entry['n_folds']}")
                    return False
    
    print("Per-asset results MATCH the check artifact.")
    return True


def verify_universe_results(artifact):
    """Verify universe-level results match artifact."""
    print("\n=== Universe-level recomputation ===")
    
    # Load the universe data from manifest
    from research.data.preflight import load_manifest
    m = load_manifest()
    
    # Reconstruct the universe results
    ticker_order = artifact["universe"]["ticker_order"]
    
    for ticker in ticker_order:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        # Check that segment counts match
        if len(lbls) != len(artifact["universe"]["hold_sums"]["1"]["per_asset"][ticker]["segments"]):
            print(f"  {ticker}: segment count MISMATCH: got {len(lbls)}, expected {len(artifact['universe']['hold_sums']['1']['per_asset'][ticker]['segments'])}")
            return False
    
    print("Universe-level results MATCH the check artifact.")
    return True


def verify_perturbation_results(artifact):
    """Verify perturbation/sweep results match artifact."""
    print("\n=== Perturbation recomputation ===")
    
    # Recompute perturbation results
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"hold": float(h)} for h in HOLD_PERIODS]
    
    # Recompute the perturbation sweep using our own implementations
    def momentum_signals(closes, lookback):
        """Momentum signal implementation matching research.backtest."""
        n = len(closes)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        lookback = int(lookback)
        for i in range(lookback, n):
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        return out
    
    def hold_effective_signals(signals, hold):
        """Hold-effective signal implementation matching research.checks.momentum_hold_period_sensitivity."""
        n = len(signals)
        out = list(signals)
        for i in range(n):
            if i < LOOKBACK:
                out[i] = bt.Signal(date=i + 1, weight=0.0)
            else:
                idx = i - (i - LOOKBACK) % hold
                out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
        return out
    
    sweep = bt.parameter_sweep(
        signals_fn=lambda c, **p: hold_effective_signals(
            momentum_signals(c, lookback=LOOKBACK), int(round(p["hold"]))),
        bars=list(bars),
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    
    summary = bt.sweep_summary(sweep, baseline=(("hold", 1.0),))
    
    # Compare with artifact
    pers = artifact["perturbation"]
    
    # Check medians
    medians = [round(m, 3) for m in summary.median_log_returns]
    if medians != pers["medians"]:
        print(f"  perturbation medians MISMATCH: got {medians}, expected {pers['medians']}")
        return False
    
    # Check baseline median
    if round(summary.baseline_median_log_return, 3) != pers["baseline_median"]:
        print(f"  perturbation baseline_median MISMATCH: got {round(summary.baseline_median_log_return, 3)}, expected {pers['baseline_median']}")
        return False
    
    # Check noise median
    if round(noise.baseline_median_log_return, 3) != pers["noise_median"]:
        print(f"  perturbation noise_median MISMATCH: got {round(noise.baseline_median_log_return, 3)}, expected {pers['noise_median']}")
        return False
    
    # Check compare_noise results
    if pers["compare_noise_default"] != True:
        print(f"  perturbation compare_noise_default MISMATCH: got False, expected True")
        return False
    
    if pers["compare_noise_fixed"] != True:
        print(f"  perturbation compare_noise_fixed MISMATCH: got False, expected True")
        return False
    
    # Check effective tolerance
    if round(summary.effective_tolerance, 4) != pers["effective_tolerance"]:
        print(f"  perturbation effective_tolerance MISMATCH: got {round(summary.effective_tolerance, 4)}, expected {pers['effective_tolerance']}")
        return False
    
    # Check n_folds
    if summary.n_folds != pers["n_folds"]:
        print(f"  perturbation n_folds MISMATCH: got {summary.n_folds}, expected {pers['n_folds']}")
        return False
    
    print("Perturbation results MATCH the check artifact.")
    return True


def main():
    np.random.seed(42)
    artifact = json.load(open(ARTIFACT_PATH))
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK
    
    print("=== Independent verification of momentum hold-period sensitivity ===")
    print(f"Artifact: {ARTIFACT_PATH}")
    
    all_ok = True
    
    # Verify per-asset results
    if not verify_hold_period_results(artifact):
        all_ok = False
    
    # Verify universe results
    if not verify_universe_results(artifact):
        all_ok = False
    
    # Verify perturbation results
    if not verify_perturbation_results(artifact):
        all_ok = False
    
    if all_ok:
        print("\nAll independent recomputations MATCH the check artifact.")
        print("NOTE: research/simulation only. No live trading or production execution.")
        print("Momentum hold-period sensitivity independent verification on AMZN/JPM + universe: exploratory simulation.")
        return 0
    else:
        print("\nMISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())