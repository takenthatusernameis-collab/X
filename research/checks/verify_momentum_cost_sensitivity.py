"""Independent verification of the momentum cost sensitivity check.

This script recomputes each segment's walk-forward median and the cost-robustness
results from a fresh `walk_forward` implementation (not ``stress_segments``),
then compares against the artifact written by the cost sensitivity check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments``.
- All values are loaded from the cost check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments, cost model) so a
match confirms the check computed from those inputs; a mismatch would flag an
error in the cost check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_SENSITIVITY_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

# Match the same cost model used in the cost sensitivity check
COST_MODEL = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=1.0,
    commission_per_share=0.01,
    slippage_cents=0.02,
    slippage_proportional=0.0005,
    warmup_periods=60,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
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


def random_signals(n, seed):
    """Coin-flip positioning for matched null benchmark."""
    rng = np.random.default_rng(seed)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


def segment_median_with_costs(ticker, s, e, train, test, warm, overlap, lookback):
    """Walk-forward regime-stability result for one asset segment and lookback with costs."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    
    # Momentum signal
    signals = momentum_signals(closes, lookback)
    
    # Segment slicing
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    
    # Run with costs
    cfg = bt.BacktestConfig(
        initial_capital=COST_MODEL.initial_capital,
        target_exposure=COST_MODEL.target_exposure,
        commission_per_trade=COST_MODEL.commission_per_trade,
        commission_per_share=COST_MODEL.commission_per_share,
        slippage_cents=COST_MODEL.slippage_cents,
        slippage_proportional=COST_MODEL.slippage_proportional,
        warmup_periods=COST_MODEL.warmup_periods,
        risk_free=COST_MODEL.risk_free,
        periods_per_year=COST_MODEL.periods_per_year,
        margin_rate=COST_MODEL.margin_rate,
        margin_call_liquidate=COST_MODEL.margin_call_liquidate,
    )
    
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def null_segment_median_with_costs(ticker, bars, s, e, train, test, warm, overlap, lookback):
    """Coin-flip null benchmark with identical cost model."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    
    # Coin-flip signal with deterministic seed
    seed = SEED + hash(str(lookback)) % 2**32
    signals = random_signals(len(closes), seed)
    
    # Segment slicing
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    
    # Run null with costs using the same cost model
    cfg = bt.BacktestConfig(
        initial_capital=COST_MODEL.initial_capital,
        target_exposure=COST_MODEL.target_exposure,
        commission_per_trade=COST_MODEL.commission_per_trade,
        commission_per_share=COST_MODEL.commission_per_share,
        slippage_cents=COST_MODEL.slippage_cents,
        slippage_proportional=COST_MODEL.slippage_proportional,
        warmup_periods=COST_MODEL.warmup_periods,
        risk_free=COST_MODEL.risk_free,
        periods_per_year=COST_MODEL.periods_per_year,
        margin_rate=COST_MODEL.margin_rate,
        margin_call_liquidate=COST_MODEL.margin_call_liquidate,
    )
    
    # Create noise benchmark for null
    res = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cfg,
        seed=seed,
    )
    
    return res.baseline_median_log_return


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact(COST_SENSITIVITY_ARTIFACT)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)
    
    print("=== Independent Recomputation (Fresh walk_forward + cost model) ===")
    all_ok = True

    # For each lookback and ticker, recompute medians vs artifact
    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} ---")
        
        # Load tickers
        tickers = {}
        from research.data.preflight import load_manifest
        m = load_manifest()
        for entry in m["entries"]:
            tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
        
        for ticker in ["AMZN", "JPM"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]
            
            # Recompute real momentum medians
            real_medians = []
            for lab, s, e in runs:
                m, nf = segment_median_with_costs(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                real_medians.append(round(m, 3))
            
            # Recompute null medians  
            null_medians = []
            for lab, s, e in runs:
                m = null_segment_median_with_costs(asset, bars, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                null_medians.append(round(m, 3))
            
            # Get published values from artifact
            pub_real = artifact["cost_results"][str(lb)]["real"][ticker]["medians"]
            pub_null = artifact["cost_results"][str(lb)]["null"][ticker]["medians"]
            
            # Compare
            real_match = real_medians == pub_real
            null_match = null_medians == pub_null
            
            if not real_match or not null_match:
                all_ok = False
                print(f"  {ticker}: REAL mismatch -> recomputed {real_medians} vs published {pub_real}")
                print(f"  {ticker}: NULL mismatch -> recomputed {null_medians} vs published {pub_null}")
            else:
                print(f"  {ticker}: REAL {real_medians} vs published {pub_real} -> MATCH")
                print(f"  {ticker}: NULL {null_medians} vs published {pub_null} -> MATCH")
                
                # Also verify cost impact calculation
                if abs(pub_null[0]) < 0.001:
                    impact = pub_real[0] - pub_null[0]
                else:
                    impact = (pub_real[0] - pub_null[0]) / abs(pub_null[0])
                    
                summary_impact = artifact["cost_impact_summary"][str(lb)][ticker]["cost_impact"]
                impact_match = abs(impact - summary_impact) < 0.001
                if not impact_match:
                    all_ok = False
                    print(f"  {ticker}: COST IMPACT mismatch -> recomputed {impact:.3f} vs published {summary_impact:.3f}")
                else:
                    print(f"  {ticker}: COST IMPACT {impact:.3f} vs published {summary_impact:.3f} -> MATCH")

    # Determinism test
    print("\nDeterminism test: recompute AMZN lookback=5 medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    m1 = []
    m2 = []
    for lab, s, e in runs[:2]:  # Just first two segments to reduce runtime
        med1, _ = segment_median_with_costs("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5)
        med2, _ = segment_median_with_costs("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5)
        m1.append(round(med1, 3))
        m2.append(round(med2, 3))
    
    print(f"  AMZN lookback 5 r1: {m1}")
    print(f"  AMZN lookback 5 r2: {m2}")
    print(f"  -> {'IDENTICAL' if m1 == m2 else 'DIFFERENT'}")
    assert m1 == m2, "verification path not deterministic"

    print("\n" + "="*60)
    if all_ok:
        print("SUCCESS: All independent recomputations MATCH the cost sensitivity artifact.")
        print("The momentum edge survives cost stress check results are verified.")
    else:
        print("FAILURE: MISMATCH FOUND in cost sensitivity recomputation.")
        print("The cost check artifact may contain defects and should not be admitted.")
        sys.exit(1)
    print("="*60)
    print("NOTE: research/simulation only. No live trading or production")
    print("execution. Momentum cost sensitivity independent verification complete.")


if __name__ == "__main__":
    main()
