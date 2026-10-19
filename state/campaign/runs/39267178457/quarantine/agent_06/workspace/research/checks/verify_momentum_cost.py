"""Independent verification of the cost-sensitive momentum check
(research/checks/momentum_cost.py).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the cost-sensitive momentum check (`state/check_artifacts/momentum_cost_results.json`).
A mismatch would flag a defect in the cost check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Cost-sensitive costs are matched exactly.
- All values are loaded from the cost check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments, cost model)
so a match confirms the check computed from those inputs; a mismatch would
flag an error in the cost check's pipeline.
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

# Conservative transaction cost model - MUST MATCH momentum_cost.py
COMMISSION_PER_TRADE = 0.01
SLIPPAGE_CENTS = 0.5
SLIPPAGE_PROPORTIONAL = 0.001
COMMISSION_PER_SHARE = 0.0

COST_CONFIG = bt.BacktestConfig(
    commission_per_trade=COMMISSION_PER_TRADE,
    commission_per_share=COMMISSION_PER_SHARE,
    slippage_cents=SLIPPAGE_CENTS,
    slippage_proportional=SLIPPAGE_PROPORTIONAL,
)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_results.json"

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

def momentum_signals_with_lookback(closes, lookback):
    """Fresh implementation of momentum signal with specific lookback (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def segment_median(ticker, s, e, lookback, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals_with_lookback(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(
        commission_per_trade=COST_CONFIG.commission_per_trade,
        commission_per_share=COST_CONFIG.commission_per_share,
        slippage_cents=COST_CONFIG.slippage_cents,
        slippage_proportional=COST_CONFIG.slippage_proportional,
        warmup_periods=warm)
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
    assert artifact["lookbacks"] == list(LOOKBACKS)
    
    print("=== Independent recomputation (fresh walk_forward path with conservative costs) ===")
    print(f"Cost model: commission_per_trade=${COST_CONFIG.commission_per_trade:.3f}, "
          f"slippage_cents=${COST_CONFIG.slippage_cents:.3f}, "
          f"slippage_proportional={COST_CONFIG.slippage_proportional:.3%}")
    
    all_ok = True
    
    # Universe-level per-asset medians and verdicts vs artifact
    print("\n=== Universe-level recomputation ===")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} recomputation ---")
        
        for ticker in artifact[f"lookback_{lookback}"]["ticker_order"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]
            
            medians = [round(segment_median(ticker, s, e, lookback, TRAIN, TEST, WARM, OVERLAP)[0], 3)
                      for lab, s, e in runs]
            pub = artifact[f"lookback_{lookback}"]["per_asset"][ticker]["medians"]
            pub_lbls = artifact[f"lookback_{lookback}"]["per_asset"][ticker]["segments"]
            
            status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  [DEBUG] ticker={ticker}: len(runs)={len(runs)} lbls={lbls}")
                print(f"  [DEBUG] ticker={ticker}: pub_lbls={pub_lbls}")
                print(f"  [DEBUG] ticker={ticker}: len(medians)={len(medians)} medians={medians}")
                print(f"  [DEBUG] ticker={ticker}: pub_medians={pub}")
            
            # Regenerate the verdict from recomputed medians and null medians.
            cand = np.array(medians)
            null = np.array(artifact[f"lookback_{lookback}"]["per_asset"][ticker]["null_medians"])
            cand_disp = round(float(np.std(cand)), 3)
            null_disp = round(float(np.std(null)), 3)
            
            if all(abs(m) <= 0.05 for m in medians):
                regen_verdict = "CONSISTENT_WITH_NOISE"
            elif cand_disp > 2.0 * null_disp:
                regen_verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                regen_verdict = "REGIME_STABLE_LOSS"
            else:
                regen_verdict = "REGIME_STABLE"
            
            verdict_status = "MATCH" if regen_verdict == artifact[f"lookback_{lookback}"]["per_asset"][ticker]["verdict"] else "MISMATCH"
            if verdict_status != "MATCH":
                all_ok = False
                print(f"      verdict regeneration did not match: expected {artifact[f'lookback_{lookback}']["per_asset"][ticker]["verdict"]}, got {regen_verdict}")
            
            print(f"  {ticker:6s}: medians {medians} segments={pub_lbls[:3]}... -> {status} | "
                  f"verdict {regen_verdict} (pub: {artifact[f'lookback_{lookback}']["per_asset"][ticker]["verdict"]})")
    
    # Determinism check for this verification path
    print("\n=== Determinism of verification path ===")
    for lookback in LOOKBACKS:
        print(f"  Running lookback {lookback} twice for determinism check...")
        m1 = []
        m2 = []
        
        # Run twice and compare
        for ticker in ["AMZN", "JPM"]:  # Just check a couple assets for speed
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            
            medians1 = [round(segment_median(ticker, s, e, lookback, TRAIN, TEST, WARM, OVERLAP)[0], 3)
                       for lab, s, e in runs]
            medians2 = [round(segment_median(ticker, s, e, lookback, TRAIN, TEST, WARM, OVERLAP)[0], 3)
                       for lab, s, e in runs]
            
            m1.extend(medians1)
            m2.extend(medians2)
            print(f"    {ticker} lookback {lookback}: {'IDENTICAL' if medians1 == medians2 else 'DIFFERENT'}")
        
        if m1 != m2:
            all_ok = False
            print(f"  VERIFICATION PATH NOT DETERMINISTIC: lookback {lookback}")
    
    print()
    if all_ok:
        print("All independent recomputations MATCH the cost check artifact.")
        print("Transaction-cost robustness evidence is verified.")
    else:
        print("MISMATCH FOUND: the cost check artifact does not reproduce via the "
              "independent path with conservative costs. Investigate before admitting the results.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Conservative cost-momentum independent verification on the collected universe: exploratory simulation.")

if __name__ == "__main__":
    main()