"""Cost-sensitive momentum test with realistic transaction costs.

Objective: Test whether the candidate positive momentum edge (lookback 3/5/10)
survives realistic transaction-cost stress on the collected 10-asset universe.

This test uses a conservative cost model:
- Commission per trade: $0.005 (5 cents)
- Commission per share: $0.0002 (0.2 bps) 
- Fixed slippage: $0.005 per share (0.5 cents)
- Proportional slippage: 10 bps (0.001)

Method: long the previous N-day return; hold 1 day; daily rebalance.
Cost model applied to every round trip. Compare momentum vs matched coin-flip
null with identical cost parameters.

Falsification prediction: momentum either shows CONSISTENT_WITH_NOISE (edge
vanishes under costs) or REGIME_DEPENDENT/REGIME_STABLE_LOSS under costs.
Either outcome closes the candidate positive evidence cell.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3,5,10; same walk-forward
windows and volatility blocks as momentum.py; conservative cost model; 
determinism asserted via independent recomputation of AMZN/JPM subset.

Research / simulation only. No live trading or production execution.
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"

# Conservative realistic cost model
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.005,      # $0.005 per trade
    commission_per_share=0.0002,     # $0.0002 per share (0.2 bps)
    slippage_cents=0.05,            # $0.005 fixed slippage per share (0.5 cents)
    slippage_proportional=0.001,    # 10 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

NULL_SEED = 42  # seed for matched null generation


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
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

    print("\n=== 3. Leakage review per asset (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-sensitive momentum lookback sweep (3 / 5 / 10) ===")
    print(f"  Cost model: commission_per_trade=${REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share=${REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents=${REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs matched null with identical costs\n")

    # Load target tickers for the full universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index] with costs applied
    print("Running cost-sensitive momentum tests...")
    lookback_results = {}
    for lb in LOOKBACKS:
        print(f"  lookback={lb}: ", end="")
        result = bt.stress_segments_across_tickers(
            tickers=all_tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lb),),
            param_grid=[{"lookback": lb}],
            cfg=REALISTIC_COSTS,  # Apply cost model
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        lookback_results[lb] = result
        print(f"completed")

    # Universe-level cost robustness results
    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    
    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            asset_result = lookback_results[lb].assets[ticker]
            
            # Create a simplified profile similar to previous agent's structure
            scenario = asset_result.scenarios[0]
            profile[str(lb)] = {
                "segments": asset_result.scenarios[0].name,
                "medians": [round(scenario.baseline_median_log_return, 3)],
                "null_medians": [round(scenario.noise_median_log_return, 3)],
                "candidate_dispersion": round(asset_result.candidate_dispersion, 3),
                "null_dispersion": round(asset_result.null_dispersion, 3),
                "n_folds": asset_result.n_folds,
                "verdict": asset_result.overall_verdict,
            }
            
            # Track lookback-5 verdict counts for reference
            if lb == 5:
                lookback_5_counts[asset_result.overall_verdict] += 1
        
        # Determine robustness verdict
        v = determine_robustness(profile)
        robustness_counts[v] += 1
        
        # Store the per-lookback medians + nulls for the artifact
        profile["edge_counts"] = get_edge_counts(profile)
        profile["robustness_verdict"] = v
        
        per_asset[ticker] = profile

    print("  Lookback-5 (reference) verdict counts across universe: ")
    for k, v in lookback_5_counts.items():
        print(f"    {k}: {v}")
    
    print("  Cost-robustness verdict counts:")
    print(f"    ROBUST: {robustness_counts['ROBUST']} (>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} (edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}")
    
    print("  Per-asset cost profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        # Show medians for each lookback
        med_strs = []
        for lb in LOOKBACKS:
            scenario = lookback_results[lb].assets[ticker].scenarios[0]
            verdict = lookback_results[lb].assets[ticker].overall_verdict
            med_strs.append(f"{lb}:{scenario.baseline_median_log_return:+.3f} {verdict}")
        
        print(f"    {ticker:6s} {med_strs[0]:<15s} {med_strs[1]:<15s} {med_strs[2]:<15s} rob={p['robustness_verdict']:<8s} "
              f"edge_counts={p['edge_counts']}")
    print("")

    # Determinism: rerun the target subset and assert identical output.
    print("\n=== 5. Determinism check (recompute AMZN/JPM) ===")
    target_tickers = {t: all_tickers[t] for t in target}
    
    res2 = {}
    for lb in LOOKBACKS:
        res2[lb] = bt.stress_segments_across_tickers(
            tickers=target_tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            param_grid=[{"lookback": lb}],
            baseline=(("lookback", lb),),
            cfg=REALISTIC_COSTS,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )

    # Convert RegimeStressResult objects to serializable format for comparison
    def result_to_dict(result):
        return {
            "overall_verdict": result.overall_verdict,
            "candidate_dispersion": result.candidate_dispersion,
            "null_dispersion": result.null_dispersion,
            "n_folds": result.n_folds,
            "n_periods": result.n_periods,
            "scenarios": [
                {
                    "name": s.name,
                    "baseline_median_log_return": s.baseline_median_log_return,
                    "noise_median_log_return": s.noise_median_log_return,
                    "edge_status": s.edge_status,
                    "n_folds": s.n_folds,
                    "n_periods": s.n_periods,
                    "median_log_returns": s.median_log_returns,
                }
                for s in result.scenarios
            ],
            "baseline_param_set": result.baseline_param_set,
        }

    out1_dict = {lb: {t: result_to_dict(lookback_results[lb].assets[t]) 
                      for t in target} 
                  for lb in LOOKBACKS}
    out2_dict = {lb: {t: result_to_dict(res2[lb].assets[t]) 
                      for t in target} 
                  for lb in LOOKBACKS}
    
    out1 = json.dumps(out1_dict, sort_keys=True)
    out2 = json.dumps(out2_dict, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output with costs"

    # Independent verification: recompute matched null with costs
    print("\n=== 6. Matched null verification with costs ===")
    null_results = verify_matched_null_with_costs(REALISTIC_COSTS)

    # Build artifact for verification
    artifact_per_asset = {}
    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            asset_result = lookback_results[lb].assets[ticker]
            scenario = asset_result.scenarios[0]
            
            profile[str(lb)] = {
                "segments": asset_result.scenarios[0].name,
                "medians": [round(scenario.baseline_median_log_return, 3)],
                "null_medians": [round(scenario.noise_median_log_return, 3)],
                "candidate_dispersion": round(asset_result.candidate_dispersion, 3),
                "null_dispersion": round(asset_result.null_dispersion, 3),
                "n_folds": asset_result.n_folds,
                "verdict": asset_result.overall_verdict,
            }
        
        profile["edge_counts"] = get_edge_counts(profile)
        profile["robustness_verdict"] = determine_robustness(profile)
        
        artifact_per_asset[ticker] = profile

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_model={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        per_asset=artifact_per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): artifact_per_asset[t][str(lb)]["medians"]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): artifact_per_asset[t][str(lb)]["null_medians"]
                                           for lb in LOOKBACKS},
                    verdicts_5=artifact_per_asset[t]["5"]["verdict"],
                    edge_counts=artifact_per_asset[t]["edge_counts"],
                    robustness_verdict=artifact_per_asset[t]["robustness_verdict"],
                    n_folds_5=artifact_per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
            null_verification=null_results,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Cost-sensitive momentum test across the collected universe: "
          "exploratory simulation.")
    return 0


def determine_robustness(profile):
    """Determine robustness verdict for one asset's lookback results."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    
    for lb in LOOKBACKS:
        sr = profile[str(lb)]
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            counts["EDGE"] += 1
        elif sr["verdict"] == "REGIME_STABLE_LOSS":
            counts["LOSS"] += 1
    
    if counts["EDGE"] >= 2:
        return "ROBUST"
    elif counts["EDGE"] == 1:
        return "SENSITIVE"
    else:
        return "NO_EDGE"


def get_edge_counts(profile):
    """Get edge counts for one asset's lookback results."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    
    for lb in LOOKBACKS:
        sr = profile[str(lb)]
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            counts["EDGE"] += 1
        elif sr["verdict"] == "REGIME_STABLE_LOSS":
            counts["LOSS"] += 1
    
    return counts


def verify_matched_null_with_costs(cost_cfg):
    """Verify matched null with identical cost model."""
    # Generate null bars
    null_bars = bt.generate_bars(600, seed=NULL_SEED)
    
    # Run null with same cost parameters
    null_results = {}
    
    for lb in LOOKBACKS:
        sig = bt.momentum_signals(null_bars.closes_array(), lookback=lb)
        res = bt.walk_forward(
            list(null_bars), sig, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cost_cfg)
        
        # Convert to log returns for comparison
        log_returns = []
        for f in res.folds:
            log_returns.append(np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)))
        
        # Calculate total return from the first fold's equity curve
        total_return = 0.0
        if res.folds:
            total_return = float((res.folds[0].equity_curve[-1] - 1e6) / 1e6)
        
        null_results[str(lb)] = {
            "median_log_return": round(float(np.median(log_returns)) if len(log_returns) > 0 else 0.0, 3),
            "n_trades": len(log_returns),
            "total_return": round(total_return, 3),
        }
    
    return null_results


if __name__ == "__main__":
    sys.exit(main())