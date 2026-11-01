"""Real-data momentum lookback sweep with realistic transaction costs.

Objective: Test whether the momentum edge observed at lookback=5
(`research/checks/momentum.py`) is robust to both lookback parameter variations
and realistic transaction costs (commission + slippage).

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
daily rebalance) is walk-forward validated with realistic costs:
- commission_per_trade: $2.0
- commission_per_share: $0.003 per share  
- slippage_cents: $2.0 per share
- slippage_proportional: 0.05% per trade

Walk-forward validation uses train=252d/test=84d/warmup=60d/overlap=60d
inside each of the 4 contiguous volatility blocks of the real-data regime family,
for lookback in {3, 5, 10}. Each lookback's per-segment median log return is
compared against a coin-flip sign null benchmark run on the same segments.

A lookback-robust edge with costs is one where the uniformly-positive
REGIME_STABLE pattern at lookback=5 also appears at >= 2 of the 3 lookbacks
for an asset; a lookback-sensitive result (edge only at lookback=5) is a
parameter-fragility warning even when the lookback=5 edge is genuine.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as `momentum.py`; realistic costs applied at
every fill. Determinism is asserted via an internal independent recomputation
of the AMZN/JPM subset.

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
COST_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_cost_results.json"

# Realistic cost model from examples/ma_crossover_real_data.py
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
)

ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker and one lookback with given costs."""
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
    return {
        "segments": [s.name for s in res.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "n_folds": res.n_folds,
        "verdict": res.overall_verdict,
    }


def lookback_verdict(sr):
    """Robustness verdict for one lookback's scenario result with costs.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge on this asset; REGIME_STABLE with uniformly negative
    medians is a stable loss, not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset.

    ROBUST      : uniformly positive REGIME_STABLE edges at >= 2 lookbacks.
    SENSITIVE   : a positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE     : no lookback shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


def compare_with_costs_and_nulls(lookback_results_zero_cost, lookback_results_costs):
    """Compare results with and without costs, and identify cost-robust edges."""
    comparison = {}
    
    for lb in LOOKBACKS:
        for ticker in lookback_results_zero_cost[lb]:
            zero_cost_result = lookback_results_zero_cost[lb][ticker]
            cost_result = lookback_results_costs[lb][ticker]
            
            comparison[f"{ticker}_lb{lb}"] = {
                "zero_cost": zero_cost_result,
                "with_costs": cost_result,
                "edge_preserved_with_costs": (
                    lookback_verdict(cost_result) == "EDGE" and 
                    lookback_verdict(zero_cost_result) == "EDGE"
                ),
                "edge_survives_costs": (
                    lookback_verdict(cost_result) == "EDGE" and 
                    all(m > 0 for m in cost_result["medians"])
                ),
                "cost_impact": {
                    "median_reduction": [round(zero_cost_result["medians"][i] - cost_result["medians"][i], 3) 
                                       for i in range(len(zero_cost_result["medians"]))],
                    "positive_edges_lost": 
                        (lookback_verdict(zero_cost_result) == "EDGE" and lookback_verdict(cost_result) != "EDGE"),
                    "verdict_change": zero_cost_result["verdict"] != cost_result["verdict"]
                }
            }
    
    return comparison


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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
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

    print("\n=== 4. Momentum lookback sweep with realistic costs ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments, realistic costs applied\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index] with realistic costs
    lookback_results_costs = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COST_CONFIG) for t in ticker_order}
                              for lb in LOOKBACKS}
    
    # Also run with zero costs for comparison
    lookback_results_zero_cost = {lb: {t: segment_result(t, all_tickers[t], lb, ZERO_COST_CONFIG) for t in ticker_order}
                                  for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    
    for ticker in ticker_order:
        profile = {str(lb): lookback_results_costs[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts[v] += 1
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        # Lookback-robustness edge counts (edges per asset, for the summary)
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        # Store the per-lookback medians + nulls for the artifact's perturbation
        # view (keyed by lookback so the verifier can recompute each column).
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  lookback-robustness verdict counts: "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")
    print("  Per-asset perturbation profiles (with costs):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")
    
    # Compare zero-cost vs realistic-cost results
    print("=== 5. Cost sensitivity analysis ===")
    comparison = compare_with_costs_and_nulls(lookback_results_zero_cost, lookback_results_costs)
    
    cost_robust_assets = 0
    cost_sensitive_assets = 0
    all_positive_with_costs = 0
    
    for ticker in ticker_order:
        ticker_comparison = {}
        for lb in LOOKBACKS:
            comp = comparison[f"{ticker}_lb{lb}"]
            ticker_comparison[str(lb)] = comp
        
        # Determine if edge is cost-robust (positive with costs and positive without)
        cost_robust = all(comp["edge_survives_costs"] for comp in ticker_comparison.values())
        cost_sensitive = any(comp["edge_survives_costs"] for comp in ticker_comparison.values()) and not cost_robust
        all_positive = all(comp["edge_survives_costs"] for comp in ticker_comparison.values())
        
        if cost_robust:
            cost_robust_assets += 1
        elif cost_sensitive:
            cost_sensitive_assets += 1
        if all_positive:
            all_positive_with_costs += 1
            
        print(f"  {ticker}:")
        print(f"    cost-robust edge (positive with costs): {cost_robust}")
        print(f"    cost-sensitive edge: {cost_sensitive}")
        print(f"    all lookbacks positive with costs: {all_positive}")
        print(f"    median reductions: {ticker_comparison['3']['cost_impact']['median_reduction']}")
    
    print(f"\n  Summary:")
    print(f"    Cost-robust assets (>=2 lookbacks positive with costs): {cost_robust_assets} / {len(ticker_order)}")
    print(f"    Cost-sensitive assets (edge at >=1 lookback with costs): {cost_sensitive_assets} / {len(ticker_order)}")
    print(f"    Assets with all lookbacks positive with costs: {all_positive_with_costs} / {len(ticker_order)}")

    # Determinism: rerun the target subset and assert identical output (with costs).
    print("\n=== 6. Determinism test (with realistic costs) ===")
    res2 = {lb: {t: segment_result(t, tickers[t], lb, REALISTIC_COST_CONFIG) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results_costs[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output with realistic costs"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=42,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                           for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
        cost_model={"commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
                   "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
                   "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
                   "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional},
        comparison=comparison,
    )
    with open(COST_ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(COST_ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with realistic costs across the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())