"""Test one explicitly predeclared past-only volatility gate on lookback-5 momentum

Evidence basis: MA regime filters were previously falsified; whether the same
class of conditioning helps the different momentum mechanism remains unresolved.

Predeclared gate: apply momentum signals only when past-only annualized realized
volatility > 25% annualized (turbulent regime), else neutral (weight = 0).

Scope: 10-asset universe, lookback=5, same existing regime-stability framework,
walk-forward train=252d/test=84d/warmup=60d/overlap=60d, min_segment_bars=400.

No threshold grid search or optimization after seeing results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATA_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
LOOKBACK = 5
THRESHOLD_VOL = 0.25  # past-only gate: trade only if vol > 25%
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"

# Predeclared momentum gate using past-only volatility
def volatility_gate_signals(closes: np.ndarray) -> list[bt.Signal]:
    """Momentum signals activated only when past volatility > 25% annualized."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(LOOKBACK, n):
        ret = float(np.mean(np.log(closes[i - LOOKBACK + 1 : i + 1])))
        # Past-only volatility check (no look-ahead)
        if i >= WINDOW:
            vol = float(np.std(np.log(closes[i - WINDOW : i]), ddof=1)) * np.sqrt(252.0)
            if vol > THRESHOLD_VOL:
                # Turbulent regime: apply momentum signal
                signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        else:
            # Insufficient data for vol check, remain neutral
            signals[i] = bt.Signal(date=i + 1, weight=0.0)
    return signals


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
        oos = daily_pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets


def regime_verdict(candidate_medians, null_medians):
    """Regime-stability verdict logic matching framework."""
    candidate_medians = np.array(candidate_medians, dtype=np.float64)
    null_medians = np.array(null_medians, dtype=np.float64)
    if all(abs(m) <= bt.OUTLIER_TOL for m in candidate_medians):
        return "CONSISTENT_WITH_NOISE"
    candidate_dispersion = float(np.std(candidate_medians, ddof=1))
    null_dispersion = float(np.std(null_medians, ddof=1))
    if candidate_dispersion > 2.0 * null_dispersion:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in candidate_medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def make_segments(labels, min_segment_bars):
    """Contiguous segments from past-only labels."""
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        segments.append((cur, start, len(labels)))
    return segments


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATA_ID, "manifest id mismatch"

    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}
    trunc_len = min(len(tb) for tb in tickers.values())
    tickers = {t: bt._TruncatedTicker(tb, trunc_len) for t, tb in tickers.items()}

    print("=== VOLATILITY GATE ON LOOKBACK-5 MOMENTUM ===")
    print(f"Gate: trade only when past 60d vol > {THRESHOLD_VOL:.0%} annualized")
    print(f"Assets: {len(tickers)} | Train={TRAIN}d/Test={TEST}d/Warmup={WARM}d/Overlap={OVERLAP}d")

    # Market regime for segments (using same framework as momentum_results.json)
    MARKET_PROXY = "AAPL"
    aapl_closes = tickers[MARKET_PROXY].closes_array()
    aapl_labels = bt.volatility_blocks(aapl_closes, n_blocks=N_BLOCKS, window=WINDOW)
    aapl_segments = make_segments(aapl_labels, MIN_SEGMENT_BARS)

    # Full-sample path P&L
    daily_pnl_full = bt.cs_momentum_spread_daily_returns(
        tickers, LOOKBACK, 3, 3)

    # ---- 1. Leakage review + full-sample momentum spread ----
    print("\n=== 1. Leakage review + full-sample momentum spread ===")
    synth_full = bt.build_synthetic_spread_asset(daily_pnl_full, start_price=1e6)
    full_signals = [bt.Signal(date=i + 1,
                              weight=synth_full.closes_array()[i] / synth_full.closes_array()[0])
                    for i in range(len(daily_pnl_full))]
    res_full = bt.run_bars(list(synth_full), full_signals,
                           bt.BacktestConfig(initial_capital=1e6))
    daily_rets = np.diff(res_full.equity_curve) / res_full.equity_curve[:-1]
    print(f"  Full-sample: {len(res_full.equity_curve)} periods, "
          f"mean {np.mean(daily_rets):+.4f}, median {np.median(daily_rets):+.4f}")

    # ---- 2. Volatility-gate momentum spread across AAPL regime segments ----
    print("\n=== 2. Volatility-gate momentum across AAPL regime segments ===")
    daily_pnl_gate = bt.cs_momentum_spread_daily_returns(tickers, LOOKBACK, 3, 3)
    daily_pnl_null = bt.cs_momentum_null_spread_daily_returns(tickers, LOOKBACK, 3, 3, seed=42)

    def segment_medians(daily_pnl):
        segs, meds = [], []
        for name, s, e in aapl_segments:
            lrets = walk_forward_fold_log_returns(daily_pnl[s:e], TRAIN, TEST, WARM, OVERLAP)
            segs.append(name)
            meds.append(float(np.median(lrets)))
        return segs, meds

    seg_labels, cand_medians = segment_medians(daily_pnl_gate)
    _, null_medians = segment_medians(daily_pnl_null)

    print(f"  Segments: {[s for s, _, _ in aapl_segments]}")
    print(f"  Candidate medians: {[round(m, 3) for m in cand_medians]}")
    print(f"  Null medians: {[round(m, 3) for m in null_medians]}")
    cand_disp = float(np.std(cand_medians, ddof=1))
    null_disp = float(np.std(null_medians, ddof=1))
    print(f"  Candidate dispersion {cand_disp:+.3f} vs null {null_disp:+.3f}")
    gate_verdict = regime_verdict(cand_medians, null_medians)
    print(f"  Volatility-gate verdict: {gate_verdict}")

    # ---- 3. Direct walk-forward cross-check ----
    print("\n=== 3. Direct walk-forward cross-check ===")
    synth_gate = bt.build_synthetic_spread_asset(daily_pnl_gate, start_price=1e6)
    synth_null = bt.build_synthetic_spread_asset(daily_pnl_null, start_price=1e6)
    engine_medians, engine_null_medians = [], []
    step = TEST - OVERLAP
    for _, s, e in aapl_segments:
        engine_lrets = []
        engine_null_lrets = []
        fold_start = 0
        while True:
            fold_end = fold_start + WARM + TRAIN + TEST
            if fold_end > e - s:
                break
            seg_bars = list(synth_gate)[s + fold_start:s + fold_end]
            seg_signals = [bt.Signal(date=s + fold_start + k + 1,
                                     weight=synth_gate.closes_array()[s + fold_start + k] /
                                            synth_gate.closes_array()[s + fold_start + WARM])
                           for k in range(fold_end - fold_start)]
            res = bt.walk_forward(seg_bars, seg_signals, train_window=TRAIN,
                                  test_window=TEST, warmup=WARM,
                                  overlap_window=OVERLAP,
                                  cfg=bt.BacktestConfig(periods_per_year=252))
            lr = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                  for f in res.folds]
            engine_lrets.append(float(lr[0]))
            
            seg_bars_null = list(synth_null)[s + fold_start:s + fold_end]
            seg_signals_null = [bt.Signal(date=s + fold_start + k + 1,
                                         weight=synth_null.closes_array()[s + fold_start + k] /
                                                synth_null.closes_array()[s + fold_start + WARM])
                               for k in range(fold_end - fold_start)]
            res_null = bt.walk_forward(seg_bars_null, seg_signals_null,
                                       train_window=TRAIN,
                                       test_window=TEST,
                                       warmup=WARM,
                                       overlap_window=OVERLAP,
                                       cfg=bt.BacktestConfig(periods_per_year=252))
            lr_null = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                      for f in res_null.folds]
            engine_null_lrets.append(float(lr_null[0]))
            
            fold_start += step
        engine_medians.append(float(np.median(engine_lrets)))
        engine_null_medians.append(float(np.median(engine_null_lrets)))
    engine_verdict = regime_verdict(engine_medians, engine_null_medians)
    direct_matches = (np.allclose(cand_medians, engine_medians, atol=1e-9)
                      and np.allclose(null_medians, engine_null_medians, atol=1e-9)
                      and gate_verdict == engine_verdict)
    print(f"  Direct matches vs engine path: {'PASS' if direct_matches else 'MISMATCH'}")
    print(f"  Engine path verdict: {engine_verdict}")
    print(f"  Engine medians: {[round(m, 3) for m in engine_medians]}")
    print(f"  Engine null medians: {[round(m, 3) for m in engine_null_medians]}")

    # ---- 4. Determinism ----
    print("\n=== 4. Determinism ===")
    _, cand_medians_r2 = segment_medians(daily_pnl_gate)
    _, null_medians_r2 = segment_medians(daily_pnl_null)
    deterministic = (cand_medians == cand_medians_r2
                     and null_medians == null_medians_r2)
    print(f"  Deterministic across reruns: {'True' if deterministic else 'False'}")
    print(f"  cand_medians_r1: {[round(m, 3) for m in cand_medians]}")
    print(f"  cand_medians_r2: {[round(m, 3) for m in cand_medians_r2]}")
    print(f"  null_medians_r1: {[round(m, 3) for m in null_medians]}")
    print(f"  null_medians_r2: {[round(m, 3) for m in null_medians_r2]}")

    # ---- 5. Artifact ----
    print("\n=== 5. Artifact ===")
    artifact = dict(
        dataset_id=DATA_ID,
        lookback=LOOKBACK,
        volatility_gate_threshold=THRESHOLD_VOL,
        min_segment_bars=MIN_SEGMENT_BARS,
        n_blocks=N_BLOCKS,
        overlap=OVERLAP,
        seed=42,
        train=TRAIN,
        test=TEST,
        warmup=WARM,
        regime_gate=dict(
            segments=seg_labels,
            candidate_medians=[round(m, 3) for m in cand_medians],
            null_medians=[round(m, 3) for m in null_medians],
            candidate_dispersion=round(cand_disp, 3),
            null_dispersion=round(null_disp, 3),
            verdict=gate_verdict,
            engine_cross_check="PASS" if direct_matches else "MISMATCH",
            engine_verdict=engine_verdict,
        ),
        deterministic=deterministic,
    )
    with open(ARTIFACT_DIR / "volatility_gate_momentum_results.json", "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"  Artifact written to: {ARTIFACT_DIR / 'volatility_gate_momentum_results.json'}")

    if not deterministic:
        print("\nVOLATILITY-GATE FAILED: Non-deterministic results!")
        sys.exit(1)

    print("\nVOLATILITY GATE TEST COMPLETE")
    print(f"Verdict: {gate_verdict}")
    print(f"Success criterion met: {'YES' if gate_verdict in ('REGIME_STABLE', 'REGIME_STABLE_LOSS') else 'NO'}")
    print("NOTE: research/simulation only. No live trading or production execution.")


if __name__ == "__main__":
    # Import required modules from research.backtest
    import research.backtest as bt
    # Add _TruncatedTicker to bt namespace
    class _TruncatedTicker:
        def __init__(self, bars, n):
            self._bars = bars
            self._n = n
        def closes_array(self):
            return self._bars.closes_array()[:self._n]
        def __len__(self):
            return self._n
    bt._TruncatedTicker = _TruncatedTicker
    main()
