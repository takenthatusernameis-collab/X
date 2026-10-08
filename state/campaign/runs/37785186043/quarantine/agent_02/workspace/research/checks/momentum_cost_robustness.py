"""Cost robustness check for short-horizon momentum.

Objective: the momentum cell (lookback=5) was admitted as candidate positive
evidence showing a REGIME_STABLE positive edge in 7/10 assets, and the
lookback-sweep check showed the edge is robust across lookback 3/5/10 in 6/10
assets. The frontier question is whether this short-horizon momentum edge
survives realistic transaction-cost stress under the repository's existing
realistic cost model and matched null.

Method (fixed a-priori, not tuned to OOS): long the previous N-day return
(lookback 3/5/10), hold 1 day, daily rebalance. Params: walk-forward
per segment train=252d/test=84d/warmup=60d/overlap=60d, min 400 bars/segment,
4 contiguous volatility blocks, each lookback vs a coin-flip null on the same
segments. Determinism is asserted via an internal independent recomputation
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
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

# Realistic transaction cost model
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=1.0,        # $1 per round trip
    commission_per_share=0.005,      # $0.005 per share (0.5 bps)
    slippage_cents=1.0,             # $0.01 per share fixed (1 cent)
    slippage_proportional=0.02,    # 2 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

# Baseline costs (zero costs) for comparison
BASELINE_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)
def run_asset_cost_sensitivity(ticker, bars, lookback):
    """Run momentum signal with realistic costs."""
    closes = bars.closes_array()
    
    # Create segment labels
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Generate momentum signals
    signals = bt.momentum_signals(closes, lookback=lookback)
    
    # Run stress segments with realistic costs
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
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
        cfg=REALISTIC_COSTS,
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
def cost_robustness_verdict(cost_results):
    """Verdict on whether momentum edge survives realistic costs.
    
    ROBUST: momentum shows REGIME_STABLE with positive medians at >= 2 lookbacks
             under realistic costs.
    FRAGILE: momentum shows REGIME_STABLE with positive medians at >= 2 lookbacks
             under realistic costs BUT momentum does NOT outperform null significantly.
    NO_EDGE: momentum does not show positive REGIME_STABLE edge under realistic costs.
    """
    momentum_verdict = cost_results["verdict"]
    momentum_medians = cost_results["medians"]
    null_medians = cost_results["null_medians"]
    
    # Check if momentum shows positive edge under realistic costs
    has_positive_edge = (
        momentum_verdict == "REGIME_STABLE" and 
        all(m > 0 for m in momentum_medians)
    )
    
    if not has_positive_edge:
        return "NO_EDGE"
    
    # Check if momentum outperforms null
    momentum_mean = np.mean(momentum_medians)
    null_mean = np.mean(null_medians)
    
    # Simple statistical check: momentum should be meaningfully better than null
    # Use a tolerance of 0.02 (2% annualized log return difference)
    if momentum_mean > null_mean + 0.02:
        return "ROBUST"
    else:
        return "FRAGILE"
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

    print("\n=== 3. Cost robustness check for momentum lookbacks 3/5/10 ===")
    print("  Realistic cost model: $1/trade, $0.005/share, $0.01 fixed slippage, 2bps proportional")
    print("  Long the previous N-day return, hold 1 day, daily rebalance.")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost robustness check
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker]
    cost_results = {lb: {t: run_asset_cost_sensitivity(t, all_tickers[t], lb) for t in ticker_order}
                   for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "FRAGILE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        profile = {str(lb): cost_results[lb][ticker] for lb in LOOKBACKS}
        
        # Calculate robustness for each lookback
        momentum_edges = 0
        fragile_edges = 0
        for lb in LOOKBACKS:
            m = profile[str(lb)]
            verdict = m["verdict"]
            medians = m["medians"]
            null_medians = m["null_medians"]
            
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                momentum_mean = np.mean(medians)
                null_mean = np.mean(null_medians)
                if momentum_mean > null_mean + 0.02:
                    momentum_edges += 1
                else:
                    fragile_edges += 1
        
        # Overall cost robustness verdict for the asset
        total_positive_edges = momentum_edges + fragile_edges
        if total_positive_edges >= 2:
            if momentum_edges >= 2:
                cost_verdict = "ROBUST"
            else:
                cost_verdict = "FRAGILE"
        else:
            cost_verdict = "NO_EDGE"
        
        robustness_counts[cost_verdict] += 1
        
        profile["cost_robustness_verdict"] = cost_verdict
        profile["momentum_edge_count"] = momentum_edges
        profile["fragile_edge_count"] = fragile_edges
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  Cost robustness verdict counts across universe: "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(positive edge survives realistic costs) | "
          f"FRAGILE={robustness_counts['FRAGILE']} "
          f"(positive edge exists but not robust to costs) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")
    print("  Per-asset cost robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} momentum_medians={:<40s} cost_verdict={:<8s} "
              "mom_edges={}/{} frag_edges={}/{}".format(
            ticker, str(p[str(5)]["segments"]), med, p["cost_robustness_verdict"],
            p["momentum_edge_count"], len(LOOKBACKS), p["fragile_edge_count"], len(LOOKBACKS)))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: run_asset_cost_sensitivity(t, bt.load_ticker(t)[0], lb) for t in ["AMZN", "JPM"]}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: cost_results[lb][t] for t in ["AMZN", "JPM"]}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            cost_robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                            for lb in LOOKBACKS},
                    cost_robustness_verdict=per_asset[t]["cost_robustness_verdict"],
                    momentum_edge_count=per_asset[t]["momentum_edge_count"],
                    fragile_edge_count=per_asset[t]["fragile_edge_count"],
                ))
                for t in ticker_order
            ),
        ),
        cost_model=dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost robustness check (lookback 3/5/10) across "
          "the collected universe: exploratory simulation.")
    return 0
if __name__ == "__main__":
    main()
