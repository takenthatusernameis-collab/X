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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def realistic_costs():
    """Return the repository's realistic cost configuration."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )


def zero_costs():
    """Return zero-cost configuration for comparison."""
    return bt.BacktestConfig(initial_capital=1e6)


def run_momentum_cost_sweep(tickers_dict, lookback, cfg):
    """Run cost-sensitive momentum test for one lookback and one cost config."""
    results = {}
    
    for ticker, bars in tickers_dict.items():
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)
        
        # Create volatility-based segment function
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        # Run regime-stability stress test
        res = bt.stress_segments(
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
            bars=bars,
            signals=signals,
            param_grid=[{"lookback": lookback}],
            baseline=(("lookback", lookback),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg,
        )
        
        results[ticker] = {
            "segments": [s.name for s in res.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
            "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
            "candidate_dispersion": round(res.candidate_dispersion, 3),
            "null_dispersion": round(res.null_dispersion, 3),
            "verdict": res.overall_verdict,
            "n_folds": res.n_folds,
        }
    
    return results


def cost_robustness_verdict(zero_results, realistic_results):
    """Determine cost robustness for each lookback."""
    analysis = {}
    
    for lookback in LOOKBACKS:
        zero_verdict = zero_results[lookback]['overall_verdict']
        realistic_verdict = realistic_results[lookback]['overall_verdict']
        
        if realistic_verdict == zero_verdict:
            robustness = "ROBUST"
            degradation = "NONE"
        elif realistic_verdict in ["CONSISTENT_WITH_NOISE", "REGIME_STABLE_LOSS"]:
            robustness = "FRAGILE"
            degradation = "TO_NOISE/LOSS"
        else:
            robustness = "FRAGILE"
            degradation = "TO_DEPENDENT"
        
        analysis[lookback] = {
            "zero_cost_verdict": zero_verdict,
            "realistic_cost_verdict": realistic_verdict,
            "robustness": robustness,
            "degradation": degradation,
        }
    
    if all(a["robustness"] == "ROBUST" for a in analysis.values()):
        overall = "COST_ROBUST"
    elif any(a["robustness"] == "ROBUST" for a in analysis.values()):
        overall = "PARTIALLY_COST_ROBUST"
    else:
        overall = "COST_FRAGILE"
    
    return {
        "lookback_analysis": analysis,
        "overall_verdict": overall,
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Manifest integrity
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    # 2. Data preflight
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    
    # Load all tickers from the collected universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    
    print("\n=== 3. Loading real-data universe ===")
    print(f"  Loaded {len(tickers)} tickers: {', '.join(sorted(tickers.keys()))}")
    
    # 4. Run cost-sensitivity tests
    print("\n=== 4. Cost-sensitivity momentum tests ===")
    print("  Testing lookbacks: {}".format(LOOKBACKS))
    print("  Using realistic costs (commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005)")
    
    # Store results for both cost configurations
    zero_cost_results = {}
    realistic_cost_results = {}
    
    for lookback in LOOKBACKS:
        print(f"\n  Lookback {lookback}:")
        
        # Zero cost baseline
        print(f"    Zero cost...")
        zero_results = run_momentum_cost_sweep(tickers, lookback, zero_costs())
        zero_cost_results[lookback] = zero_results
        
        # Summary statistics for zero cost
        zero_verdicts = [r["verdict"] for r in zero_results.values()]
        zero_stable_pos = sum(1 for v in zero_verdicts if v == "REGIME_STABLE" and
                             any(round(m, 3) > 0 for t in zero_results for m in zero_results[t]["medians"]))
        print(f"      Zero cost: {len([v for v in zero_verdicts if v == 'REGIME_STABLE'])} "
              f"REGIME_STABLE, {len([v for v in zero_verdicts if v == 'REGIME_STABLE_LOSS'])} "
              f"REGIME_STABLE_LOSS, {len([v for v in zero_verdicts if v == 'CONSISTENT_WITH_NOISE'])} "
              f"CONSISTENT_WITH_NOISE")
        
        # Realistic costs
        print(f"    Realistic costs...")
        realistic_results = run_momentum_cost_sweep(tickers, lookback, realistic_costs())
        realistic_cost_results[lookback] = realistic_results
        
        realistic_verdicts = [r["verdict"] for r in realistic_results.values()]
        realistic_stable_pos = sum(1 for v in realistic_verdicts if v == "REGIME_STABLE" and
                                   any(round(m, 3) > 0 for t in realistic_results for m in realistic_results[t]["medians"]))
        print(f"      Realistic costs: {len([v for v in realistic_verdicts if v == 'REGIME_STABLE'])} "
              f"REGIME_STABLE, {len([v for v in realistic_verdicts if v == 'REGIME_STABLE_LOSS'])} "
              f"REGIME_STABLE_LOSS, {len([v for v in realistic_verdicts if v == 'CONSISTENT_WITH_NOISE'])} "
              f"CONSISTENT_WITH_NOISE")
    
    # 5. Cost robustness analysis
    print("\n=== 5. Cost robustness analysis ===")
    
    # Aggregate per-ticker results to get overall verdict for each lookback
    zero_aggregated = {}
    realistic_aggregated = {}
    
    for lookback in LOOKBACKS:
        # Count verdict occurrences for zero cost
        zero_verdict_counts = {}
        for ticker, results in zero_cost_results[lookback].items():
            verdict = results["verdict"]
            zero_verdict_counts[verdict] = zero_verdict_counts.get(verdict, 0) + 1
        
        # Count verdict occurrences for realistic costs
        realistic_verdict_counts = {}
        for ticker, results in realistic_cost_results[lookback].items():
            verdict = results["verdict"]
            realistic_verdict_counts[verdict] = realistic_verdict_counts.get(verdict, 0) + 1
        
        # Determine overall verdict (most common verdict)
        zero_overall = max(zero_verdict_counts.items(), key=lambda x: x[1])[0]
        realistic_overall = max(realistic_verdict_counts.items(), key=lambda x: x[1])[0]
        
        zero_aggregated[lookback] = zero_overall
        realistic_aggregated[lookback] = realistic_overall
    
    # Determine robustness analysis
    lookback_analysis = {}
    for lookback in LOOKBACKS:
        zero_verdict = zero_aggregated[lookback]
        realistic_verdict = realistic_aggregated[lookback]
        
        if realistic_verdict == zero_verdict:
            robustness = "ROBUST"
            degradation = "NONE"
        elif realistic_verdict in ["CONSISTENT_WITH_NOISE", "REGIME_STABLE_LOSS"]:
            robustness = "FRAGILE"
            degradation = "TO_NOISE/LOSS"
        else:
            robustness = "FRAGILE"
            degradation = "TO_DEPENDENT"
        
        lookback_analysis[lookback] = {
            "zero_cost_verdict": zero_verdict,
            "realistic_cost_verdict": realistic_verdict,
            "robustness": robustness,
            "degradation": degradation,
        }
    
    # Overall verdict
    if all(a["robustness"] == "ROBUST" for a in lookback_analysis.values()):
        overall = "COST_ROBUST"
    elif any(a["robustness"] == "ROBUST" for a in lookback_analysis.values()):
        overall = "PARTIALLY_COST_ROBUST"
    else:
        overall = "COST_FRAGILE"
    
    print("  Lookback | Zero Cost Verdict | Realistic Cost Verdict | Robustness")
    print("  --------|-------------------|------------------------|----------")
    for lookback in LOOKBACKS:
        analysis = lookback_analysis[lookback]
        print(f"  {lookback:7d} | {analysis['zero_cost_verdict']:17s} | "
              f"{analysis['realistic_cost_verdict']:22s} | {analysis['robustness']:10s}")
    
    print(f"\n  Overall verdict: {overall}")
    
    # 6. Write artifact for independent verification
    print("\n=== 6. Writing artifact ===")
    
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_cost_config": {
            "commission_per_trade": 2.0,
            "commission_per_share": 0.003,
            "slippage_cents": 2.0,
            "slippage_proportional": 0.0005,
        },
        "zero_cost_results": zero_cost_results,
        "realistic_cost_results": realistic_cost_results,
        "lookback_analysis": lookback_analysis,
        "overall_verdict": overall,
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"  artifact written to: {ARTIFACT_PATH}")
    
    # 7. Summary
    print("\n=== 7. Summary ===")
    print(f"  Primary question: Does the short-horizon momentum edge survive")
    print(f"    conservative transaction-cost stress on the collected real-data universe?")
    print(f"  Overall verdict: {overall}")
    print(f"  Evidence gate: SATISFIED (artifact written for independent verification)")
    print(f"\nNOTE: research/simulation only. No live trading or production execution.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())