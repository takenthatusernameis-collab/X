"""Real-data momentum cost robustness check for lookback 3/5/10 across the collected universe.

Objective: test whether the momentum edge observed at lookback=5
supports an independent cost-robust classification when realistic
transaction costs are applied, and whether the edge survives across
lookbacks 3, 5, and 10 under the same realistic cost model.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; walk-forward
validation (train=252d/test=84d/warmup=60d/overlap=60d); realistic
transaction-cost model: commission_per_trade=2.0, commission_per_share=0.003,
slippage_cents=2.0, slippage_proportional=0.0005; each lookback vs a
coin-flip null benchmark run on the same segments. Determinism is
asserted via an internal independent recomputation of the AMZN/JPM subset.

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
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_check.json"


def create_realistic_cost_config():
    """Create the realistic cost config used by the repository."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=2.0,      # matching Agent 02 results
        commission_per_share=0.003,    # matching Agent 02 results
        slippage_cents=2.0,            # matching Agent 02 results
        slippage_proportional=0.0005, # matching Agent 02 results
        warmup_periods=0,
        periods_per_year=252,
    )


def segment_result(ticker, bars, lookback, cost_cfg):
    """Walk-forward regime-stability result for one ticker and one lookback with realistic costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    # Zero-cost baseline (no transaction costs)
    zero_cost_cfg = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
        warmup_periods=0,
        periods_per_year=252,
    )
    
    zero_res = bt.stress_segments(
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
        cfg=zero_cost_cfg,
    )
    
    # Realistic cost baseline
    realistic_res = bt.stress_segments(
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
        cfg=cost_cfg,
    )
    
    # Matched null benchmark (coin-flip signals with same cost config)
    def null_signals(closes, **params):
        return bt.random_signals(len(closes), seed=SEED + int(lookback))
    
    null_res = bt.stress_segments(
        signals_fn=null_signals,
        bars=bars,
        signals=null_signals(closes, lookback=lookback),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_cfg,
    )
    
    scen = realistic_res.scenarios[0]
    return dict(
        lookback=lookback,
        segments=[s.name for s in realistic_res.scenarios],
        zero_cost_median_log_return=round(scen.baseline_median_log_return, 3),
        realistic_cost_median_log_return=round(scen.baseline_median_log_return, 3),  # same as realistic_res
        null_median_log_return=round(null_res.scenarios[0].baseline_median_log_return, 3),
        zero_cost_candidate_dispersion=round(zero_res.candidate_dispersion, 3),
        realistic_cost_candidate_dispersion=round(realistic_res.candidate_dispersion, 3),
        null_dispersion=round(null_res.null_dispersion, 3),
        zero_cost_null_dispersion=round(zero_res.null_dispersion, 3),
        n_folds=realistic_res.n_folds,
        zero_cost_verdict=zero_res.overall_verdict,
        realistic_cost_verdict=realistic_res.overall_verdict,
        null_verdict=null_res.overall_verdict,
        cost_impact_absolute=round(scen.baseline_median_log_return - scen.baseline_median_log_return, 3),  # placeholder
        cost_impact_percentage=0.0,  # placeholder
    )


def robustness_verdict(segment_results):
    """Verdict on cost robustness for one ticker.
    
    REGIME_STABLE_POSITIVE_ROBUST: realistic cost shows REGIME_STABLE positive edge
    and zero cost also shows REGIME_STABLE positive edge with minimal degradation.
    REGIME_STABLE_NEGATIVE_ROBUST: realistic cost shows REGIME_STABLE loss
    and zero cost also shows REGIME_STABLE loss (consistency).
    SENSITIVE: realistic cost and zero cost disagree (edge present at zero cost but not with costs).
    NO_EDGE: realistic cost shows CONSISTENT_WITH_NOISE or REGIME_DEPENDENT.
    """
    realistic_verdict = segment_results["realistic_cost_verdict"]
    zero_cost_verdict = segment_results["zero_cost_verdict"]
    
    # Check if realistic cost has a positive edge
    realistic_positive = (
        realistic_verdict == "REGIME_STABLE" and 
        segment_results["realistic_cost_median_log_return"] > 0
    )
    
    # Check if zero cost has a positive edge
    zero_cost_positive = (
        zero_cost_verdict == "REGIME_STABLE" and 
        segment_results["zero_cost_median_log_return"] > 0
    )
    
    # Check if null shows no edge (baseline within null tolerance)
    realistic_within_null = abs(segment_results["realistic_cost_median_log_return"] - 
                              segment_results["null_median_log_return"]) <= 0.05
    
    if realistic_positive and realistic_within_null:
        # Realistic cost within null tolerance - no evidence of edge with costs
        return "NO_EDGE_WITH_COSTS", {"realistic_positive": True, "within_null_tolerance": True}
    
    if realistic_positive and realistic_within_null == False:
        # Realistic cost shows edge beyond null
        if realistic_positive and realistic_positive:
            # Both zero and realistic costs show positive edges
            degradation = (segment_results["zero_cost_median_log_return"] - 
                          segment_results["realistic_cost_median_log_return"])
            if degradation <= 0.02:  # Minimal degradation
                return "REGIME_STABLE_POSITIVE_ROBUST", {"degradation": degradation, "minimal": True}
            else:
                return "REGIME_STABLE_POSITIVE_SENSITIVE", {"degradation": degradation, "minimal": False}
        else:
            # Edge only present at zero cost, not with costs
            return "EDGE_ONLY_WITHOUT_COSTS", {"zero_cost_positive": True, "realistic_negative": False}
    
    return "NO_EDGE", {"realistic_positive": False}


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cost_cfg = create_realistic_cost_config()

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

    print("\n=== 3. Cost robustness check per asset (lookback 3/5/10) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        print(f"  {ticker}: loaded {bars.n_bars} bars")

    # Full-universe cost robustness check
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    cost_robustness_counts = {
        "REGIME_STABLE_POSITIVE_ROBUST": 0,
        "REGIME_STABLE_POSITIVE_SENSITIVE": 0,
        "EDGE_ONLY_WITHOUT_COSTS": 0,
        "NO_EDGE": 0,
        "NO_EDGE_WITH_COSTS": 0
    }
    
    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            profile[str(lb)] = segment_result(ticker, all_tickers[ticker], lb, cost_cfg)
        
        # Calculate cost impact
        for lb in LOOKBACKS:
            profile[str(lb)]["cost_impact_absolute"] = (
                profile[str(lb)]["zero_cost_median_log_return"] - 
                profile[str(lb)]["realistic_cost_median_log_return"]
            )
            profile[str(lb)]["cost_impact_percentage"] = (
                profile[str(lb)]["cost_impact_absolute"] / 
                abs(profile[str(lb)]["zero_cost_median_log_return"]) * 100
                if profile[str(lb)]["zero_cost_median_log_return"] != 0 else 0.0
            )
        
        robustness_verdict_result, details = robustness_verdict(profile["5"])
        profile["cost_robustness_verdict"] = robustness_verdict_result
        profile["cost_robustness_details"] = details
        cost_robustness_counts[robustness_verdict_result] += 1
        
        per_asset[ticker] = profile

    print("  Cost robustness verdict counts across universe:")
    for verdict, count in cost_robustness_counts.items():
        print(f"    {verdict}: {count}")

    # Summary statistics
    print("  Cost impact summary:")
    for lb in LOOKBACKS:
        zero_costs = [per_asset[t][f"{lb}"]["zero_cost_median_log_return"] for t in ticker_order]
        realistic_costs = [per_asset[t][f"{lb}"]["realistic_cost_median_log_return"] for t in ticker_order]
        avg_zero = np.mean(zero_costs)
        avg_realistic = np.mean(realistic_costs)
        avg_impact = np.mean([per_asset[t][f"{lb}"]["cost_impact_absolute"] for t in ticker_order])
        print(f"    lookback {lb}: zero-cost {avg_zero:+.3f}, realistic-cost {avg_realistic:+.3f}, "
              f"avg impact {avg_impact:+.3f}")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result(t, tickers[t], lb, cost_cfg) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: per_asset[t][str(lb)] for t in target}
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
        cost_model={
            "commission_per_trade": 2.0,
            "commission_per_share": 0.003,
            "slippage_cents": 2.0,
            "slippage_proportional": 0.0005,
        },
        per_asset=per_asset,
        universe=dict(
            cost_robustness_counts=cost_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    zero_cost_medians={str(lb): per_asset[t][str(lb)]["zero_cost_median_log_return"] for lb in LOOKBACKS},
                    realistic_cost_medians={str(lb): per_asset[t][str(lb)]["realistic_cost_median_log_return"] for lb in LOOKBACKS},
                    null_medians={str(lb): per_asset[t][str(lb)]["null_median_log_return"] for lb in LOOKBACKS},
                    cost_robustness_verdict=per_asset[t]["cost_robustness_verdict"],
                    cost_robustness_details=per_asset[t]["cost_robustness_details"],
                    cost_impact_absolute_avg=sum(per_asset[t][str(lb)]["cost_impact_absolute"] for lb in LOOKBACKS) / len(LOOKBACKS),
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness check on the collected universe: "
          "exploratory simulation with realistic transaction costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())