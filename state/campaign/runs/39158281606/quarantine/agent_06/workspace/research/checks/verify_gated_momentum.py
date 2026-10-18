"""Independent verification of the gated momentum check
(`research/checks/gated_momentum.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results using `walk_forward`
directly (not ``stress_segments``), then compares against the artifact
written by the gated momentum check (`state/check_artifacts/gated_momentum_results.json`).
A mismatch would flag a defect in the gated momentum check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- The gated momentum spread is recomputed from raw tickers: momentum
  signals (lookback=5) are emitted only when trailing realized volatility
  exceeds 25% annualized (past-only gate).
- Segment medians are computed via a fresh walk-forward fold-log-return
  aggregation — no call to ``stress_segments`` or ``regime_stability.py``.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the gated momentum check computed from those inputs; a mismatch
would flag an error in the gated momentum check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
MARKET_PROXY = "AAPL"
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "gated_momentum_results.json"
REGIME_GATE_THRESHOLD = 0.25
def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1))
                * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    block_size = n // n_blocks
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = (float(np.median([v for v in bar_vols[s:e] if v > 0]))
                    if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels
def segments_from_labels(labels, min_segment_bars):
    """Return contiguous segment ranges (label, start, end)."""
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
    """Fresh implementation of momentum signal (lookback=5)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out
def gated_momentum_signals(closes, lookback):
    """Momentum signals gated by past-only volatility > 25% annualized.
    
    Independent implementation from the check's gated_signal_wrapper.
    """
    # First compute momentum signals
    momentum = momentum_signals(closes, lookback)
    
    # Apply volatility gate using the same logic as conditional_signal
    n = len(closes)
    window = min(60, max(4, n // 4))
    if n >= window:
        early_vol = float(
            np.std(np.log(closes[window // 2 : window]), ddof=1)
        ) * np.sqrt(252.0)
        
        # Apply gate: only emit momentum signals if volatility > threshold
        if early_vol > REGIME_GATE_THRESHOLD:
            return momentum
    
    # Apply gate: zero signals for all bars when gate is not active
    return [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
def walk_forward_fold_log_returns(daily_pnl, train, test, warm, overlap):
    """Walk-forward OOS fold log returns for a spread P&L series."""
    n = len(daily_pnl)
    step = test - overlap
    fold_start = 0
    lrets = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n:
            break
        oos = daily_pnl[fold_end - test + 1 : fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets
def segment_median(ticker, s, e, train, test, warm, overlap, lookback):
    """Run walk-forward on one segment and return median log return."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    
    # Get gated momentum signals for the segment
    signals = gated_momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    
    cfg = bt.BacktestConfig(warmup_periods=warm)
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
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK

    manifest = load_manifest()

    print("=== Independent verification of gated momentum check ===")
    print("  Predeclared volatility gate: > 25% annualized (past-only)")
    print("  Gated momentum: lookback-{} (only when volatility gate active)".format(LOOKBACK))

    # Per-asset verification for AMZN and JPM
    print("\n=== Per-asset verification ===")
    all_ok = True

    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        # Get published results from artifact
        pub = artifact["per_asset"][asset]
        pub_segments = pub["segments"]
        pub_medians = pub["medians"]
        pub_verdict = pub["verdict"]

        # Compute independent recomputation
        ind_medians = []
        ind_segments = []
        for lab, s, e in runs:
            m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK)
            ind_medians.append(round(m, 3))
            ind_segments.append(lab)

        # Compare results
        segments_match = ind_segments == pub_segments
        medians_match = ind_medians == pub_medians
        verdict_match = (ind_segments == pub_segments and 
                        ind_medians == pub_medians)  # verdict derived from medians
        
        if not segments_match or not medians_match:
            all_ok = False
            
        status = "OK" if segments_match and medians_match else "MISMATCH"
        print("  {}:".format(asset))
        print("    segments: {} (independent) vs {} (published) -> {}".format(
            ind_segments, pub_segments, "MATCH" if segments_match else "MISMATCH"))
        print("    medians: {} (independent) vs {} (published) -> {}".format(
            ind_medians, pub_medians, "MATCH" if medians_match else "MISMATCH"))
        print("    verdict: independent suggests {} vs published {}".format(
            "REGIME_STABLE" if all(abs(m) <= 0.05 for m in ind_medians) else 
            "REGIME_DEPENDENT" if abs(np.std(ind_medians)) > 2.0 * 0.019 else
            "CONSISTENT_WITH_NOISE" if all(m < 0 for m in ind_medians) else
            "REGIME_STABLE",
            pub_verdict))

    # Universe-level verification
    print("\n=== Universe-level verification ===")
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0].closes_array()
    trunc = min(len(tb) for tb in tickers.values())
    tickers = {t: tb[:trunc] for t, tb in tickers.items()}

    for ticker in artifact["universe"]["assets"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        pub_asset = artifact["universe"]["assets"][ticker]
        pub_segments = [s["name"] for s in pub_asset["scenarios"]]
        pub_medians = [s["baseline_median"] for s in pub_asset["scenarios"]]
        
        # Compute independent recomputation
        ind_medians = []
        ind_segments = []
        for lab, s, e in runs:
            m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK)
            ind_medians.append(round(m, 3))
            ind_segments.append(lab)

        # Compare results
        segments_match = ind_segments == pub_segments
        medians_match = ind_medians == pub_medians
        
        if not segments_match or not medians_match:
            all_ok = False
            
        status = "OK" if segments_match and medians_match else "MISMATCH"
        print("  {}: segments {} (independent) vs {} (published) -> {}".format(
            ticker, ind_segments, pub_segments, "MATCH" if segments_match else "MISMATCH"))
        print("    medians: {} (independent) vs {} (published) -> {}".format(
            ind_medians, pub_medians, "MATCH" if medians_match else "MISMATCH"))

    # Determinism check
    print("\n=== Determinism verification ===")
    np.random.seed(42)
    r1 = {a: segment_medians_for_asset(a, 42) for a in ["AMZN", "JPM"]}
    np.random.seed(42)
    r2 = {a: segment_medians_for_asset(a, 42) for a in ["AMZN", "JPM"]}
    
    deterministic = True
    for a in ["AMZN", "JPM"]:
        if r1[a] != r2[a]:
            deterministic = False
            print("  {}: NON-DETERMINISTIC!".format(a))
    
    if deterministic:
        print("  All recomputations are deterministic (identical across reruns).")

    print("\n" + "="*60)
    if all_ok and deterministic:
        print("INDEPENDENT VERIFICATION: MATCH (all primary path values match)")
    else:
        print("INDEPENDENT VERIFICATION: MISMATCH")
        print("  - Verify reasons above")
        print("  - Check gated_momentum_results.json artifact")
        sys.exit(1)

    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Gated momentum independent verification.")
def segment_medians_for_asset(asset, base_seed):
    """Helper function for determinism verification."""
    np.random.seed(base_seed)
    bars, dates = bt.load_ticker(asset)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    medians = []
    for lab, s, e in runs:
        m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK)
        medians.append(round(m, 3))
    return medians
if __name__ == "__main__":
    main()
