"""Independent verification of the momentum cost check.

Two verification gates:

1. Fresh recomputation of the per-asset, per-lookback momentum results via
   a separate implementation path (re-implemented volatility_blocks, segment
   reconstruction, momentum signals, walk_forward), comparing against the artifact
   written by the momentum check (`research/checks/momentum_cost.py`). The
   coin-flip null is recomputed through the framework's own
   ``noise_benchmark`` (a separate code path from ``stress_segments``), so the
   artifact's null medians are also independently reproduced.

2. Cost-robustness verdict reproducibility: the cost-robustness verdict is
   re-derived from the artifact's medians/nulls/verdicts via the documented
   rule (COST_ROBUST if a positive REGIME_STABLE edge appears at >= 2 lookbacks),
   independently of the check's own verdict computation.
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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_results.json"


# Same cost model as the check
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.005,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
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


def segment_result(ticker, bars, lookback, signals_fn):
    """Walk-forward regime-stability result for one asset segment and lookback,
    computed with fresh components: volatility_blocks, segment slicing,
    momentum signals, walk_forward, and the framework noise_benchmark. No call
    to stress_segments."""
    if isinstance(bars, bt.BarSequence):
        closes = bars.closes_array()
        bars_list = list(bars)
    else:
        closes = np.array([float(b.close) for b in bars], dtype=np.float64)
        bars_list = bars
    
    seg_bars = bars_list
    
    if callable(signals_fn):
        signals = signals_fn(closes, lookback)
    else:
        signals = signals_fn
    
    # Convert signals to correct format if needed
    if not isinstance(signals[0], bt.Signal):
        signals = [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(signals)]
    
    # Get segments using our implementation
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    results = {}
    for lab, s, e in runs:
        seg_bars_slice = seg_bars[s:e]
        seg_signals_slice = signals[s:e]
        
        res = bt.walk_forward(
            seg_bars_slice, seg_signals_slice, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=COST_CONFIG)
        cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                            for f in res.folds], dtype=np.float64)
        
        noise = noise_benchmark(
            bars=seg_bars_slice,
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=0,
            overlap_window=0,
            periods_per_year=252)
        null_log = noise.baseline_median_log_return
        
        results[(lab, s, e)] = {
            "median_log_return": float(np.median(cand_log)),
            "null_log_return": float(null_log),
            "n_folds": len(res.folds),
        }
    
    return results, labels


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS)

    print("=== Independent recomputation (fresh walk_forward + noise_benchmark) ===")
    all_ok = True

    # Recompute per-asset results
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    
    for ticker in artifact["per_asset"]:
        print(f"\n  Processing {ticker}:")
        bars, dates = bt.load_ticker(ticker)
        labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        for lb in LOOKBACKS:
            print(f"    Lookback {lb}:")
            
            # Recompute momentum with costs
            mom_results, _ = segment_result(ticker, bars, lb, momentum_signals)
            mom_medians = [round(r["median_log_return"], 3) for r in mom_results.values()]
            mom_nulls = [round(r["null_log_return"], 3) for r in mom_results.values()]
            
            # Get published results
            pub_mom = artifact["per_asset"][ticker]["momentum_results"][str(lb)]
            pub_null = pub_mom["null_medians"]
            
            # Compare
            mom_match = mom_medians == pub_mom["medians"]
            null_match = mom_nulls == pub_null
            
            if not mom_match or not null_match:
                all_ok = False
                print(f"      MISMATCH:")
                if not mom_match:
                    print(f"        Medians: recomputed {mom_medians} vs published {pub_mom['medians']}")
                if not null_match:
                    print(f"        Nulls: recomputed {mom_nulls} vs published {pub_null}")
            else:
                print(f"      MATCH: medians {mom_medians}")

    # Cost-robustness verdict reproducibility
    print("\n=== Cost-robustness verdict reproducibility ===")
    rmismatch = False
    
    def edge_status_from_artifact(medians, nulls, verdict):
        """Derive edge status from artifact data using the same logic as the check."""
        if all(abs(m) <= 0.05 for m in medians):
            return "CONSISTENT_WITH_NOISE"
        elif len(medians) > 1 and (max(medians) - min(medians)) / max(abs(m) for m in medians) > 2.0 * (max(abs(n) for n in nulls) - min(abs(n) for n in nulls)) / max(abs(n) for n in nulls) if max(abs(n) for n in nulls) > 0 else False:
            return "REGIME_DEPENDENT"
        elif all(m < 0 for m in medians):
            return "REGIME_STABLE_LOSS"
        else:
            return "REGIME_STABLE"
    
    for ticker in artifact["per_asset"]:
        edge_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_DEPENDENT": 0, 
                      "REGIME_STABLE_LOSS": 0, "REGIME_STABLE": 0}
        
        for lb in LOOKBACKS:
            result = artifact["per_asset"][ticker]["momentum_results"][str(lb)]
            verdict = result["verdict"]
            medians = result["medians"]
            nulls = result["null_medians"]
            
            edge_status = edge_status_from_artifact(medians, nulls, verdict)
            edge_counts[edge_status] += 1
        
        # Recompute cost-robustness verdict
        regime_stable_count = edge_counts["REGIME_STABLE"]
        cost_robust_recomputed = "COST_ROBUST" if regime_stable_count >= 2 else "COST_SENSITIVE"
        cost_robust_published = artifact["per_asset"][ticker]["cost_robustness_verdict"]
        
        if cost_robust_recomputed != cost_robust_published:
            rmismatch = True
            all_ok = False
            print(f"  {ticker}: MISMATCH - recomputed {cost_robust_recomputed} vs published {cost_robust_published}")
            print(f"    Edge counts: {edge_counts}")
        else:
            print(f"  {ticker}: {cost_robust_published}")

    print("\nDeterminism: recomputing AMZN lookback=5 momentum cost result twice.")
    bars_amzn, _ = bt.load_ticker("AMZN")
    results1, _ = segment_result("AMZN", bars_amzn, 5, momentum_signals)
    results2, _ = segment_result("AMZN", bars_amzn, 5, momentum_signals)
    
    medians1 = [round(r["median_log_return"], 3) for r in results1.values()]
    medians2 = [round(r["median_log_return"], 3) for r in results2.values()]
    print(f"  AMZN lookback 5 r1: {medians1}; r2: {medians2} -> {'identical' if medians1 == medians2 else 'DIFFERENT'}")
    assert medians1 == medians2, "verification path not deterministic"

    print("\n" + "="*70)
    if all_ok and not rmismatch:
        print("SUCCESS: All independent recomputations MATCH the check artifact,")
        print("and cost-robustness verdicts reproduce.")
    else:
        print("FAILURE: MISMATCH FOUND - the check artifact does not reproduce via the")
        print("independent path. Investigate before recording results.")
        sys.exit(1)
    print("="*70)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()
