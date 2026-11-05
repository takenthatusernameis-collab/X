"""Cost sensitivity check for momentum lookbacks 3/5/10 under realistic transaction cost model.

This is R-001: runs a bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

Primary question: Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?

Scope: Uses the existing 10-asset collected universe and existing momentum implementation/check framework.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitivity_results.json"

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    commission_per_trade=2.0,
)


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks (independent)."""
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
    """Re-implementation of segment extraction (independent)."""
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
    """Fresh implementation of momentum signal (independent)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_median_with_costs(ticker, lookback, s, e, train, test, warm, overlap, cfg):
    """Segment median using walk_forward with costs (independent)."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    result = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in result.folds], dtype=np.float64)
    return float(np.median(log)), len(result.folds)


def null_segment_median_with_costs(ticker, lookback, s, e, train, test, warm, overlap, cfg, seed):
    """Coin-flip null segment median using walk_forward with costs (matched)."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    # Use the framework's noise_benchmark for the matched null, same as other verification scripts
    from research.backtest.perturbation import noise_benchmark
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cfg)
    null_median = noise.baseline_median_log_return
    # Get candidate median via walk_forward
    result = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in result.folds], dtype=np.float64)
    candidate_median = float(np.median(log))
    return candidate_median, null_median, len(result.folds)


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost sensitivity check: lookback 3 / 5 / 10 across collected universe ===")
    print("Method: long previous N-day return, hold 1 day, daily rebalance.")
    print("Cost model: realistic (commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005).")
    print("Params: walk-forward per segment train={}d / test={}d / warmup={}d / overlap={}d, min {} bars/segment".format(
        TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    results = {}
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} days ---")
        
        # Track per-asset results
        per_asset = {}
        all_medians = []
        all_null_medians = []
        
        for ticker in manifest["universe"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            
            medians = []
            null_medians = []
            for lab, s, e in runs:
                # Candidate momentum with costs
                med, _ = segment_median_with_costs(ticker, lookback, s, e, TRAIN, TEST, WARM, OVERLAP, REALISTIC_COST_CONFIG)
                medians.append(round(med, 3))
                
                # Null with costs (using framework's noise_benchmark for consistency with other verification scripts)
                seed = SEED + hash((ticker, lookback, lab)) % 10000
                from research.backtest.perturbation import noise_benchmark
                bars_seg, dates_seg = bt.load_ticker(ticker)
                closes_seg = bars_seg.closes_array()[s:e]
                signals = momentum_signals(closes_seg, lookback)
                noise = noise_benchmark(
                    bars=list(bars_seg)[s:e],
                    param_grid=[{"lookback": lookback}],
                    train_window=TRAIN, test_window=TEST,
                    warmup=0, overlap_window=0, periods_per_year=252,
                    cfg=REALISTIC_COST_CONFIG)
                null_med = noise.baseline_median_log_return
                null_medians.append(round(null_med, 3))
            
            all_medians.extend(medians)
            all_null_medians.extend(null_medians)
            
            per_asset[ticker] = {
                "lookback": lookback,
                "segments": [lab for lab, _, _ in runs],
                "medians": medians,
                "null_medians": null_medians,
                "candidate_dispersion": round(float(np.std(medians)), 3),
                "null_dispersion": round(float(np.std(null_medians)), 3),
            }
        
        # Determine verdict for this lookback
        candidate_disp = round(float(np.std(all_medians)), 3)
        null_disp = round(float(np.std(all_null_medians)), 3)
        
        if all(abs(m) <= 0.05 for m in all_medians):
            verdict = "CONSISTENT_WITH_NOISE"
        elif candidate_disp > 2.0 * null_disp:
            verdict = "REGIME_DEPENDENT"
        elif all(m < 0 for m in all_medians):
            verdict = "REGIME_STABLE_LOSS"
        else:
            verdict = "REGIME_STABLE"
        
        results[f"lookback_{lookback}"] = {
            "lookback": lookback,
            "n_assets": len(manifest["universe"]),
            "per_asset": per_asset,
            "all_candidate_medians": [round(m, 3) for m in all_medians],
            "all_null_medians": [round(m, 3) for m in all_null_medians],
            "candidate_dispersion": candidate_disp,
            "null_dispersion": null_disp,
            "verdict": verdict,
            "cost_config": {
                "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
                "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
                "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
                "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            }
        }
        
        print(f"  {lookback} days: {len(manifest['universe'])} assets")
        print(f"    candidate dispersion: {candidate_disp:.3f}, null dispersion: {null_disp:.3f}")
        print(f"    verdict: {verdict}")
        
        # Show first few medians for reference
        print(f"    candidate medians (first 3 assets): {[round(m, 3) for m in all_medians[:3]]}")
        print(f"    null medians (first 3 assets): {[round(m, 3) for m in all_null_medians[:3]]}")

    print("\n=== 4. Summary of cost robustness ===")
    for lb_key, result in results.items():
        lookback = result["lookback"]
        verdict = result["verdict"]
        candidate_medians = result["all_candidate_medians"]
        null_medians = result["all_null_medians"]
        
        print(f"  Lookback {lookback}:")
        print(f"    Verdict: {verdict}")
        print(f"    Candidate medians: {candidate_medians[:5]}{'...' if len(candidate_medians) > 5 else ''}")
        print(f"    Null medians: {null_medians[:5]}{'...' if len(null_medians) > 5 else ''}")

    # Determinism check: rerun first lookback
    print("\n=== 5. Determinism check ===")
    first_lookback = LOOKBACKS[0]
    print(f"Re-running lookback {first_lookback} with identical inputs...")
    
    first_result = results[f"lookback_{first_lookback}"]
    first_ticker = manifest["universe"][0]
    bars, dates = bt.load_ticker(first_ticker)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    medians_rerun = []
    for lab, s, e in runs[:2]:  # Just check first 2 segments
        med, _ = segment_median_with_costs(first_ticker, first_lookback, s, e, TRAIN, TEST, WARM, OVERLAP, REALISTIC_COST_CONFIG)
        medians_rerun.append(round(med, 3))
    
    original_medians = first_result["per_asset"][first_ticker]["medians"]
    print(f"  Original medians: {original_medians[:2]}")
    print(f"  Rerun medians: {medians_rerun}")
    print(f"  Determinism: {'PASS' if medians_rerun == original_medians[:2] else 'FAIL'}")
    
    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "cost_config": REALISTIC_COST_CONFIG.__dict__,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warm": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "results": results,
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Cost sensitivity check for momentum lookback 3/5/10: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())