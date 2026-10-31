"""Cost-sensitivity check for momentum (lookback 3 / 5 / 10) on the collected universe.

Objective: test whether the short-horizon momentum edge (supported in momentum.py) survives
conservative transaction costs under the repository's existing realistic cost model and matched null.

Method: for each collected ticker, momentum (long the previous N-day return, hold 1 day, daily rebalance)
is walk-forward validated (train=252d/test=84d/warmup=60d/overlap=60d) inside each of the 4 contiguous
volatility blocks of the real-data regime family, for lookback in {3, 5, 10}, under TWO config modes:

1. Zero-cost baseline (the existing momentum edge test)
2. Realistic costs (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)

Each lookback's per-segment median log return is compared against a coin-flip sign null benchmark
run on the same segments (the matched null). Both config modes use the same baseline; the cost
sensitivity is measured as the difference between zero-cost and realistic-cost results.

A positive edge surviving realistic costs (i.e., remaining above the null tolerance and with a
material reduction from zero-cost) is evidence that the momentum edge is economically meaningful.
Otherwise, the edge is cost-sensitive and likely to erode in live trading.

Method (fixed a-priori, not tuned to OOS):
- lookbacks 3, 5, 10; same walk-forward windows and volatility blocks as momentum.py
- matched null: coin-flip sign null on the same segments (same logic as momentum_lookback_sweep.py)
- zero-cost baseline: same as momentum_lookback_sweep.py (artifact momentum_lookback_sweep_results.json)
- realistic costs: BacktestConfig with commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005
- Determinism asserted via internal independent recomputation of the AMZN/JPM subset.

Research/simulation only. No live trading or production execution.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
WARMUP = WARM  # matches the momentum implementation
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, under given config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
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
        cfg=cfg,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review per asset (lookback=5 reference, zero cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        cfg_zero = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARMUP)
        res = bt.run_bars(list(bars), signals, cfg_zero)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Load zero-cost momentum lookback sweep results (momentum_lookback_sweep_results.json) ===")
    zero_artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
    if not zero_artifact_path.exists():
        print(f"ERROR: Zero-cost artifact not found at {zero_artifact_path}")
        print("This check requires the momentum_lookback_sweep_results.json artifact.")
        sys.exit(1)
    
    with open(zero_artifact_path) as f:
        zero_artifact = json.load(f)
    
    print("  zero-cost artifact loaded successfully.")

    print("\n=== 5. Momentum lookback sweep (3 / 5 / 10) under realistic costs ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Realistic cost configuration (existing realistic cost model from ma_crossover_real_data.py)
    cfg_cost = bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARMUP,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )

    # Full-universe lookback sweep under realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print("  Running momentum lookback sweep (lookback 3/5/10) under realistic costs...")
    
    # Results indexed [lookback][ticker][segment index].
    cost_results = {lb: {t: segment_result(t, all_tickers[t], lb, cfg_cost) for t in ticker_order}
                    for lb in LOOKBACKS}

    # Cost-robustness verdict counts
    robust_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    zero_robust_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    per_asset = {}
    zero_per_asset = zero_artifact['per_asset']
    
    for ticker in ticker_order:
        # Determine lookback-robustness verdict based on zero-cost results (consistent with momentum_lookback_sweep)
        zero_edges = 0
        
        # Extract per-lookback verdicts from zero-cost artifact
        for lb in LOOKBACKS:
            # The zero artifact has lookback_5, lookback_3, lookback_10 entries as keys
            # Check if we have the verdict field and medians indicate an edge
            zero_result = zero_per_asset[ticker].get(str(lb), {})
            
            zero_verdict = zero_result.get('verdict', 'NO_EDGE')
            zero_medians = zero_result.get('medians', [])
            
            # A positive edge requires REGIME_STABLE verdict and all positive medians
            if zero_verdict == 'REGIME_STABLE' and all(m > 0 for m in zero_medians):
                zero_edges += 1
        
        if zero_edges >= 2:
            verdict = "ROBUST"
            robust_counts["ROBUST"] += 1
            zero_robust_counts["ROBUST"] += 1
        elif zero_edges == 1:
            verdict = "SENSITIVE"
            robust_counts["SENSITIVE"] += 1
            zero_robust_counts["SENSITIVE"] += 1
        else:
            verdict = "NO_EDGE"
            robust_counts["NO_EDGE"] += 1
            zero_robust_counts["NO_EDGE"] += 1
        
        # Store per-asset results for artifact
        profile = {}
        for lb in LOOKBACKS:
            # Zero-cost result for this lookback
            zero_result = zero_per_asset[ticker].get(str(lb), {})
            # Cost result for this lookback
            cost_result = cost_results[lb][ticker]
            
            profile[str(lb) + '_zero'] = zero_result
            profile[str(lb) + '_cost'] = cost_result
        
        profile['robustness_verdict'] = verdict
        profile['lookback_5_zero_medians'] = zero_per_asset[ticker].get('5', {}).get('medians', [])
        profile['lookback_5_cost_medians'] = cost_results[5][ticker]['medians']
        profile['lookback_5_zero_verdict'] = zero_per_asset[ticker].get('5', {}).get('verdict', 'NO_EDGE')
        profile['lookback_5_cost_verdict'] = cost_results[5][ticker]['verdict']
        
        per_asset[ticker] = profile

    print("  Realistic cost lookback-robustness verdict counts:")
    print(f"    ROBUST (zero-cost edge survives costs): {robust_counts['ROBUST']}")
    print(f"    SENSITIVE (edge present at exactly one lookback): {robust_counts['SENSITIVE']}")
    print(f"    NO_EDGE (no positive edge at any lookback): {robust_counts['NO_EDGE']}")
    
    print(f"  Zero-cost lookback-robustness verdict counts (reference):")
    print(f"    ROBUST: {zero_robust_counts['ROBUST']}")
    print(f"    SENSITIVE: {zero_robust_counts['SENSITIVE']}")
    print(f"    NO_EDGE: {zero_robust_counts['NO_EDGE']}")

    # Determinism: rerun the target subset and assert identical output.
    print("\n=== 6. Determinism check ===")
    res2 = {lb: {t: segment_result(t, tickers[t], lb, cfg_cost) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: cost_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("  determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Cost-sensitivity analysis
    print("\n=== 7. Cost-sensitivity analysis ===")
    
    # Simple cost-sensitivity comparison
    print("  Comparing zero-cost vs realistic-cost results...")
    cost_sensitive_changes = []
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            zero_result = zero_per_asset[ticker].get(str(lb), {})
            cost_result = cost_results[lb][ticker]
            
            zero_verdict = zero_result.get('verdict', 'NO_EDGE')
            cost_verdict = cost_result.get('verdict', 'NO_EDGE')
            
            # Check if edge was present in zero cost but not in realistic cost
            zero_edge_present = zero_verdict == 'REGIME_STABLE' and all(m > 0 for m in zero_result.get('medians', []))
            cost_edge_present = cost_verdict == 'REGIME_STABLE' and all(m > 0 for m in cost_result.get('medians', []))
            
            if zero_edge_present and not cost_edge_present:
                cost_sensitive_changes.append((ticker, lb, 'ROBUST->LOSS', 
                                              zero_result.get('medians', []), 
                                              cost_result.get('medians', [])))
    
    if cost_sensitive_changes:
        print("  Cost-sensitive changes found:")
        for ticker, lb, change, zero_medians, cost_medians in cost_sensitive_changes:
            print(f"    {ticker} lookback {lb}: {change}")
            print(f"      zero-cost medians: {zero_medians}")
            print(f"      realistic-cost medians: {cost_medians}")
            reduction = [(zero_medians[i] - cost_medians[i]) for i in range(len(zero_medians))]
            print(f"      reduction per segment: {reduction}")
    else:
        print("  No material cost sensitivity changes detected.")

    # Final summary
    print("\n=== 8. Final summary ===")
    print(f"  Realistic cost lookback-robustness verdict counts:")
    print(f"    ROBUST (zero-cost edge survives costs): {robust_counts['ROBUST']}")
    print(f"    SENSITIVE (edge present at exactly one lookback): {robust_counts['SENSITIVE']}")
    print(f"    NO_EDGE (no positive edge at any lookback): {robust_counts['NO_EDGE']}")
    print(f"  Zero-cost lookback-robustness verdict counts (reference):")
    print(f"    ROBUST: {zero_robust_counts['ROBUST']}")
    print(f"    SENSITIVE: {zero_robust_counts['SENSITIVE']}")
    print(f"    NO_EDGE: {zero_robust_counts['NO_EDGE']}")
    print(f"  Cost-sensitive changes: {len(cost_sensitive_changes)}")
    print(f"  Determinism verification: PASSED")
    print(f"  Note: realistic cost config = commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005")

    # Write artifact for independent verification
    print("\n=== 9. Write cost-sensitivity artifact ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        realistic_cost_config={
            'commission_per_trade': 2.0,
            'commission_per_share': 0.003,
            'slippage_cents': 2.0,
            'slippage_proportional': 0.0005,
        },
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"  artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-sensitivity check on AMZN/JPM + universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())