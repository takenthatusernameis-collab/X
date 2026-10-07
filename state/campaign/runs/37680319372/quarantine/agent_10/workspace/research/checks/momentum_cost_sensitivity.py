"""Cost-sensitive momentum lookback sweep (Zero-cost vs Realistic cost comparison).

Objective: test whether the momentum edge observed at lookback=5 under zero costs
survives realistic transaction-cost stress on the collected 10-asset universe,
for lookback=3,5,10. Compare zero-cost vs realistic-cost results.

Method: long the previous N-day return; hold 1 day; daily rebalance.
Walk-forward per segment: train=252d/test=84d/warmup=60d/overlap=60d,
4 contiguous volatility blocks, min 400 bars/segment. Matched coin-flip null.
Uses realistic transaction-cost model from examples/ma_crossover_real_data.py.

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

# Zero-cost configuration (baseline)
CFG_ZERO = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)

# Realistic cost configuration (from examples/ma_crossover_real_data.py)
CFG_REAL = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and config."""
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
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
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


def lookback_verdict(sr):
    """Robustness verdict for one lookback's scenario result.
    
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


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Cost-Sensitive Momentum Lookback Sweep (3 / 5 / 10) ===")
    print("Zero-cost baseline vs realistic transaction costs (from examples/ma_crossover_real_data.py):")
    print("  commission_per_trade=2.0, commission_per_share=0.003")
    print("  slippage_cents=2.0, slippage_proportional=0.0005")
    print(f"Params: walk-forward per segment train={TRAIN}d/test={TEST}d/warmup={WARM}d/"
          f"overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, 4 volatility blocks,")
    print("matched coin-flip null on same segments.")

    # Load full universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Run both zero-cost and realistic cost sweeps
    print("\n=== Running zero-cost sweep ===")
    lookback_results_zero = {lb: {t: segment_result(t, all_tickers[t], lb, CFG_ZERO)
                                  for t in ticker_order}
                             for lb in LOOKBACKS}

    print("=== Running realistic-cost sweep ===")
    lookback_results_real = {lb: {t: segment_result(t, all_tickers[t], lb, CFG_REAL)
                                 for t in ticker_order}
                             for lb in LOOKBACKS}

    # Analyze zero cost results
    print("\n=== Zero-Cost Results ===")
    per_asset_zero = {}
    robustness_counts_zero = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts_zero = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                             "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile = {str(lb): lookback_results_zero[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts_zero[v] += 1
        r5 = profile["5"]
        lookback_5_counts_zero[r5["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_zero[ticker] = profile

    print(f"  lookback 5 verdict counts: {lookback_5_counts_zero}")
    print(f"  lookback-robustness verdict counts: ROBUST={robustness_counts_zero['ROBUST']} | "
          f"SENSITIVE={robustness_counts_zero['SENSITIVE']} | "
          f"NO_EDGE={robustness_counts_zero['NO_EDGE']}")

    # Analyze realistic cost results
    print("\n=== Realistic Cost Results ===")
    per_asset_real = {}
    robustness_counts_real = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts_real = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                             "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile = {str(lb): lookback_results_real[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts_real[v] += 1
        r5 = profile["5"]
        lookback_5_counts_real[r5["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_real[ticker] = profile

    print(f"  lookback 5 verdict counts: {lookback_5_counts_real}")
    print(f"  lookback-robustness verdict counts: ROBUST={robustness_counts_real['ROBUST']} | "
          f"SENSITIVE={robustness_counts_real['SENSITIVE']} | "
          f"NO_EDGE={robustness_counts_real['NO_EDGE']}")

    # Compare cost impact
    print("\n=== Cost Impact Analysis ===")
    cost_impact = {}
    for ticker in ticker_order:
        zero_5 = lookback_results_zero[5][ticker]
        real_5 = lookback_results_real[5][ticker]
        cost_diff = real_5["medians"][0] - zero_5["medians"][0]
        cost_diff_pct = (cost_diff / zero_5["medians"][0] * 100) if zero_5["medians"][0] != 0 else 0
        verdict_change = "NO_CHANGE"
        if zero_5["verdict"] != "CONSISTENT_WITH_NOISE" and real_5["verdict"] == "CONSISTENT_WITH_NOISE":
            verdict_change = "REGRESS_TO_NOISE"
        elif zero_5["verdict"] == "CONSISTENT_WITH_NOISE" and real_5["verdict"] != "CONSISTENT_WITH_NOISE":
            verdict_change = "GAIN_EDGE"

        cost_impact[ticker] = {
            "zero_cost_median": zero_5["medians"][0],
            "realistic_cost_median": real_5["medians"][0],
            "cost_difference": round(cost_diff, 3),
            "cost_difference_pct": round(cost_diff_pct, 1),
            "verdict_change": verdict_change,
        }

    print(f"  Assets where realistic costs erase the edge: {sum(1 for v in cost_impact.values() if v['verdict_change'] == 'REGRESS_TO_NOISE')}")
    print(f"  Assets where realistic costs still show an edge: {sum(1 for v in cost_impact.values() if v['verdict_change'] == 'GAIN_EDGE')}")
    print(f"  Assets where verdict unchanged: {sum(1 for v in cost_impact.values() if v['verdict_change'] == 'NO_CHANGE')}")
    print(f"  Total lookback-robust assets: {robustness_counts_zero['ROBUST']} (zero-cost) vs {robustness_counts_real['ROBUST']} (realistic-cost)")

    # Determinism checks (target subset)
    print("\n=== Determinism ===")
    target = ["AMZN", "JPM"]

    # Zero-cost determinism
    res2_zero = {lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_ZERO)
                      for t in target}
                 for lb in LOOKBACKS}
    out1_zero = json.dumps({lb: {t: lookback_results_zero[lb][t] for t in target}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps(res2_zero, sort_keys=True)
    zero_deterministic = out1_zero == out2_zero
    print(f"  Zero-cost deterministic: {zero_deterministic}")

    # Realistic-cost determinism
    res2_real = {lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_REAL)
                      for t in target}
                 for lb in LOOKBACKS}
    out1_real = json.dumps({lb: {t: lookback_results_real[lb][t] for t in target}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_real = json.dumps(res2_real, sort_keys=True)
    real_deterministic = out1_real == out2_real
    print(f"  Realistic-cost deterministic: {real_deterministic}")

    assert zero_deterministic, "zero-cost non-deterministic output"
    assert real_deterministic, "realistic-cost non-deterministic output"

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost=dict(
            per_asset=per_asset_zero,
            lookback_5_counts=lookback_5_counts_zero,
            robustness_counts=robustness_counts_zero,
        ),
        realistic_cost=dict(
            per_asset=per_asset_real,
            lookback_5_counts=lookback_5_counts_real,
            robustness_counts=robustness_counts_real,
        ),
        cost_impact=cost_impact,
    )
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Cost-sensitive momentum lookback sweep: exploratory simulation.")
    print("\n=== COST ROBUSTNESS VERIFICATION SUMMARY ===")
    print(f"Zero-cost momentum edge: {robustness_counts_zero['ROBUST']} ROBUST assets, {robustness_counts_zero['SENSITIVE']} SENSITIVE assets, {robustness_counts_zero['NO_EDGE']} NO_EDGE assets")
    print(f"Realistic-cost momentum edge: {robustness_counts_real['ROBUST']} ROBUST assets, {robustness_counts_real['SENSITIVE']} SENSITIVE assets, {robustness_counts_real['NO_EDGE']} NO_EDGE assets")

    # Determine if costs erase the edge
    if robustness_counts_real['ROBUST'] < robustness_counts_zero['ROBUST']:
        print("RESULT: Costs REDUCE the lookback-robust momentum edge.")
    elif robustness_counts_real['ROBUST'] == robustness_counts_zero['ROBUST']:
        print("RESULT: Costs PRESERVE the lookback-robust momentum edge.")
    else:
        print("RESULT: Costs INCREASE the lookback-robust momentum edge.")

    cost_erasure_assets = sum(1 for v in cost_impact.values() if v['verdict_change'] == 'REGRESS_TO_NOISE')
    print(f"Assets where costs erase the edge (to CONSISTENT_WITH_NOISE): {cost_erasure_assets}")

    if robustness_counts_real['ROBUST'] > 0 and all(v['verdict_change'] != 'REGRESS_TO_NOISE' for v in cost_impact.values()):
        print("SUCCESS: The momentum edge survives realistic transaction-cost stress.")
        return 0
    else:
        print("RESULT: The momentum edge does not survive realistic transaction-cost stress.")
        return 0


if __name__ == "__main__":
    sys.exit(main())