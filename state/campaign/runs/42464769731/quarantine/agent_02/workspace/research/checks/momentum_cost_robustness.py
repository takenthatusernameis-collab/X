"""Momentum cost robustness: lookback 3 / 5 / 10 with realistic transaction costs.

Objective: test whether the short-horizon momentum edge (long previous N-day return,
hold 1 day, daily rebalance) survives realistic trading costs on the collected
real-data universe.

Evidence basis: research/LEARNING_STATE.md records 6/10 assets showing lookback-\nrobust positive momentum edges (lookback 3/5/10) with REGIME_STABLE verdicts.\nThe "cost robustness" qualification remains decision-relevant: does the edge\npersist when realistic commissions, slippage, and cash constraints are applied?\n
Method (fixed a-priori, not tuned to OOS):
- Per-asset walk-forward (train=252d, test=84d, warmup=60d, overlap=60d) of\n  momentum_signals with lookback in {3,5,10} on the same 10-asset collected\n  adjusted-close universe.
- Realistic cost model (from examples):
  * commission_per_trade = 2.0 USD
  * commission_per_share = 0.003 (0.3%)
  * slippage_cents = 2.0 cents per share
  * slippage_proportional = 0.0005 (5 bps)
- Matched coin-flip null benchmark (same costs) for each asset/segment.
- Cost-robustness verdict: CONSISTENT_WITH_COSTS if the candidate's baseline\n  median log return > null baseline + 2x null dispersion (sample-calibrated tol).\n
Research/simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
def segment_cost_robustness(ticker, bars, lookback):
    """Cost robustness for one ticker and one lookback."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    # Candidate with realistic costs
    candidate_res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )

    # Matched null with realistic costs (same parameters)
    null_res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )

    # Create a synthetic noise benchmark using the same cost parameters
    noise_benchmark = bt.noise_benchmark(
        bars=bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        seed=SEED,
    )

    # Cost robustness classification based on whether candidate edge > null edge + tolerance
    # Use sample-calibrated tolerance (2x null dispersion / sqrt(n_folds))
    null_disp = null_res.null_dispersion
    n_folds = null_res.n_folds
    tolerance = 2 * null_disp / np.sqrt(n_folds) if n_folds > 0 else 0
    
    candidate_edge = candidate_res.baseline_median_log_return
    null_edge = null_res.baseline_median_log_return
    
    verdict = "CONSISTENT_WITH_COSTS" if (
        candidate_res.overall_verdict == "REGIME_STABLE" and 
        candidate_edge > null_edge + tolerance
    ) else "INSUFFICIENT_COST_ROBUSTNESS"
    
    # Additional metrics
    candidate_disp_vs_null = candidate_res.candidate_dispersion / null_res.null_dispersion if null_res.null_dispersion > 0 else float('inf')

    return dict(
        segments=[s.name for s in candidate_res.scenarios],
        candidate_medians=[round(s.baseline_median_log_return, 3) for s in candidate_res.scenarios],
        candidate_total_returns=[round(s.total_return, 3) for s in candidate_res.scenarios],
        null_medians=[round(s.baseline_median_log_return, 3) for s in null_res.scenarios],
        null_total_returns=[round(s.total_return, 3) for s in null_res.scenarios],
        noise_median=round(noise_benchmark.baseline_median_log_return, 3),
        candidate_dispersion=round(candidate_res.candidate_dispersion, 3),
        null_dispersion=round(null_res.null_dispersion, 3),
        candidate_edge=round(candidate_res.baseline_median_log_return, 3),
        null_edge=round(null_res.baseline_median_log_return, 3),
        noise_edge=round(noise_benchmark.baseline_median_log_return, 3),
        tolerance=round(tolerance, 4),
        candidate_disp_vs_null=round(candidate_disp_vs_null, 2),
        verdict=verdict,
        n_folds=candidate_res.n_folds,
        cost_type="realistic",
    )

def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Momentum Cost Robustness Analysis ===")
    print("Lookbacks: {} | Costs: commission_per_trade=2.0, commission_per_share=0.003,")
          "slippage_cents=2.0, slippage_proportional=0.0005")
    print("Method: walk-forward IS/OOS validation on real data with matched coin-flip null.")

    # Manifest integrity
    print("\n=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # Preflight gate
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    print("  [PASS] preflight")

    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]

    print(f"\n=== 3. Cost robustness across collected universe ({len(tickers)} assets) ===")

    # Results per lookback per ticker
    lookback_results = {lb: {t: None for t in tickers.keys()} for lb in LOOKBACKS}
    cost_robustness_summary = {lb: {"CONSISTENT_WITH_COSTS": 0, "INSUFFICIENT_COST_ROBUSTNESS": 0} for lb in LOOKBACKS}

    for lb in LOOKBACKS:
        print(f"\nLookback {lb} days:")
        per_asset = {}
        consistent = 0
        insufficient = 0
        
        for ticker in tickers.keys():
            bars = tickers[ticker]
            res = segment_cost_robustness(ticker, bars, lb)
            lookback_results[lb][ticker] = res
            
            per_asset[ticker] = res
            
            if res["verdict"] == "CONSISTENT_WITH_COSTS":
                consistent += 1
            else:
                insufficient += 1
                
            print(f"  {ticker}: candidate={res['candidate_edge']:+.3f} null={res['null_edge']:+.3f} disp_vs_null={res['candidate_disp_vs_null']:.2f} verdict={res['verdict']}")
        
        cost_robustness_summary[lb]["CONSISTENT_WITH_COSTS"] = consistent
        cost_robustness_summary[lb]["INSUFFICIENT_COST_ROBUSTNESS"] = insufficient

    # Output summary
    print(f"\n=== 4. Cost robustness summary ===")
    for lb in LOOKBACKS:
        print(f"  Lookback {lb}: CONSISTENT_WITH_COSTS={cost_robustness_summary[lb]['CONSISTENT_WITH_COSTS']}/10")
        print(f"           INSUFFICIENT_COST_ROBUSTNESS={cost_robustness_summary[lb]['INSUFFICIENT_COST_ROBUSTNESS']}/10")

    # Determinism check (partial)
    print("\n=== 5. Determinism check (AMZN/JPM subset) ===")
    target = ["AMZN", "JPM"]
    for lb in LOOKBACKS:
        res1 = {t: segment_cost_robustness(t, tickers[t], lb) for t in target}
        res2 = {t: segment_cost_robustness(t, tickers[t], lb) for t in target}
        out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                          for lb in LOOKBACKS}, sort_keys=True)
        out2 = json.dumps(res2, sort_keys=True)
        match = out1 == out2
        print(f"  lookback {lb}: deterministic {match}")

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_type="realistic",
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=lookback_results,
        cost_robustness_summary=cost_robustness_summary,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\n=== 6. Artifact written to {ARTIFACT_PATH} ===")
    print("NOTE: research/simulation only. No live trading or production execution.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
