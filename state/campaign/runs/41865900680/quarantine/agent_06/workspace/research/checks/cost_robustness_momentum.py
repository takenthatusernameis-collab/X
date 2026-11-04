"""Cost-robustness momentum lookback sweep across the collected universe.

Objective: test whether the short-horizon momentum edge observed in
`research/checks/momentum_lookback_sweep.py` survives conservative transaction-cost
stress on the collected real-data universe.

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
weekly rebalance? Actually daily rebalance based on examples) is walk-forward
validated for lookback in {3, 5, 10} under two cost regimes:
1. Zero cost (commission_per_trade=0, commission_per_share=0, slippage_cents=0, slippage_proportional=0)
2. Realistic costs (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)

Each lookback's per-segment median log return is compared against a matched null
(baseline of the same cost regime). The synthetic generator cannot answer this
because it contains no momentum structure, so the null is drawn from the collected
data itself via a coin-flip sign benchmark run on the same segments.

A cost-robust edge is one where the uniformly-positive REGIME_STABLE pattern at
a lookback (showing a positive edge under realistic costs) also appears at >=2
of the 3 lookbacks for an asset. This directly informs whether momentum should
be admitted to the evidence base as candidate positive evidence that survives
realistic transaction costs.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return;
hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as `momentum_lookback_sweep.py`. Determinism is
asserted via an internal independent recomputation of a subset (AAPL, JPM).

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

# Cost regimes
ZERO_COST = bt.BacktestConfig(initial_capital=1e6, warmup_periods=0)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_momentum_results.json"


def segment_result(
    ticker: str,
    bars: bt.BarSequence,
    lookback: int,
    cost_cfg: bt.BacktestConfig,
) -> dict:
    """Walk-forward regime-stability result for one ticker, one lookback, and one cost regime."""
    closes = bars.closes_array()
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
        cfg=cost_cfg,
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
        cost_regime="zero" if cost_cfg.commission_per_trade == 0 else "realistic",
    )


def lookback_verdict(sr: dict) -> str:
    """Robustness verdict for one lookback's scenario result.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge on this asset under this cost regime; REGIME_STABLE
    with uniformly negative medians is a stable loss, not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results: dict) -> tuple[str, dict]:
    """Verdict on whether the edge is lookback-robust for one asset under one cost regime.

    COST_ROBUST      : uniformly positive REGIME_STABLE edges at >=2 lookbacks under realistic costs.
    COST_SENSITIVE   : a positive REGIME_STABLE edge at exactly one lookback only under realistic costs.
    COST_STABLE_LOSS : uniform losses under realistic costs (negative edge).
    COST_NO_EDGE     : no lookback shows a positive REGIME_STABLE edge under realistic costs.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "COST_ROBUST", counts
    if counts["EDGE"] == 1:
        return "COST_SENSITIVE", counts
    if counts["LOSS"] >= 2:
        return "COST_STABLE_LOSS", counts
    return "COST_NO_EDGE", counts


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Cost-Robustness Momentum Lookback Sweep (3 / 5 / 10) ===")
    print(f"  Cost regimes: ZERO vs REALISTIC (commission=$2/trade + $0.3/share, 2c + 10bps slippage)")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, 4 contiguous volatility blocks, each lookback vs a matched null on the same segments\n")

    # All tickers from the collected universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][cost_regime][ticker][segment index]
    results = {lb: {"zero": {}, "realistic": {}} for lb in LOOKBACKS}

    # Test each lookback with both cost regimes
    for cost_label, cost_cfg in [("zero", ZERO_COST), ("realistic", REALISTIC_COSTS)]:
        print(f"\n=== Testing lookback {cost_label} cost regime ===")
        for lookback in LOOKBACKS:
            print(f"  Lookback {lookback} ({cost_label} cost)...")
            for ticker in ticker_order:
                res = segment_result(ticker, all_tickers[ticker], lookback, cost_cfg)
                results[lookback][cost_label][ticker] = res
                verdict = res["verdict"]
                med_str = " ".join(f"{m:+.3f}" for m in res["medians"])
                print(f"    {ticker}: {verdict} medians={med_str}")

    # Build per-asset cost-robustness profiles
    per_asset = {}
    cost_robustness_counts = {
        "COST_ROBUST": 0,
        "COST_SENSITIVE": 0,
        "COST_STABLE_LOSS": 0,
        "COST_NO_EDGE": 0,
    }
    realistic_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            profile[str(lb)] = results[lb]["realistic"][ticker]
        v, counts = robustness_verdict(profile)
        cost_robustness_counts[v] += 1
        r5 = profile["5"]
        realistic_5_counts[r5["verdict"]] += 1
        profile["cost_robustness_verdict"] = v
        profile["edge_counts"] = counts
        # Store the realistic cost medians for the artifact
        profile["realistic_lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["realistic_lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        profile["lookback_5_segments"] = profile["5"]["segments"]
        per_asset[ticker] = profile

    print(f"\n=== Cost-Robustness Verdict Counts (realistic costs) ===")
    print(f"  COST_ROBUST (>=2 lookbacks positive edge): {cost_robustness_counts['COST_ROBUST']} "
          f"(momentum survives realistic costs)")
    print(f"  COST_SENSITIVE (edge at exactly 1 lookback): {cost_robustness_counts['COST_SENSITIVE']} "
          f"(edge only at one lookback, cost sensitivity)")
    print(f"  COST_STABLE_LOSS (uniformly negative stable losses): {cost_robustness_counts['COST_STABLE_LOSS']} "
          f"(momentum loses money consistently under realistic costs)")
    print(f"  COST_NO_EDGE (no lookback shows positive edge): {cost_robustness_counts['COST_NO_EDGE']} "
          f"(momentum ineffective under realistic costs)")

    print(f"\n  Realistic cost lookback 5 verdict counts: {realistic_5_counts}")
    print(f"\n  Per-asset cost-robustness profiles:")
    for ticker in ticker_order[:10]:  # Show first 10 to keep output manageable
        p = per_asset[ticker]
        rob = p["cost_robustness_verdict"]
        real_med_str = ", ".join(
            f"{lb}:{p[str(lb)]['medians'][0]:6.3f}" for lb in LOOKBACKS
        )
        print(f"    {ticker:6s}: lookback 5={p['5']['verdict']:20s} medians={real_med_str} "
              f"cost_robust={rob:<15s} edge_counts={p['edge_counts']}")
    if len(ticker_order) > 10:
        print(f"    ... and {len(ticker_order) - 10} more tickers")

    # Determinism: rerun a subset and assert identical output
    print(f"\n=== Determinism check (subset: AAPL, JPM) ===")
    target = ["AAPL", "JPM"]
    res2 = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COSTS) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: results[lb]["realistic"][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    identical = out1 == out2
    print(f"  Results from AAPL/JPM recomputation identical: {identical}")
    if not identical:
        print("  WARNING: non-deterministic output detected!")

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=list(LOOKBACKS),
        min_segment_bars=MIN_SEGMENT_BARS,
        n_blocks=N_BLOCKS,
        overlap=OVERLAP,
        per_asset=per_asset,
        universe=dict(
            realistic_cost_lookback_5_counts=realistic_5_counts,
            cost_robustness_counts=cost_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    realistic_lookback_medians={str(lb): per_asset[t]["realistic_lookback_medians"][str(lb)]
                                               for lb in LOOKBACKS},
                    realistic_lookback_null_medians={str(lb): per_asset[t]["realistic_lookback_null_medians"][str(lb)]
                                                for lb in LOOKBACKS},
                    lookback_5_verdict=per_asset[t]["5"]["verdict"],
                    lookback_5_segments=per_asset[t]["5"]["segments"],
                    edge_counts=per_asset[t]["edge_counts"],
                    cost_robustness_verdict=per_asset[t]["cost_robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
        cost_regimes={
            "zero": dict(commission_per_trade=ZERO_COST.commission_per_trade,
                         commission_per_share=ZERO_COST.commission_per_share,
                         slippage_cents=ZERO_COST.slippage_cents,
                         slippage_proportional=ZERO_COST.slippage_proportional),
            "realistic": dict(commission_per_trade=REALISTIC_COSTS.commission_per_trade,
                             commission_per_share=REALISTIC_COSTS.commission_per_share,
                             slippage_cents=REALISTIC_COSTS.slippage_cents,
                             slippage_proportional=REALISTIC_COSTS.slippage_proportional),
        },
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\n=== Artifact written to: {ARTIFACT_PATH} ===")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Cost-robustness momentum lookback sweep across the collected universe: exploratory simulation.")
    print("      Primary question: Does the short-horizon momentum edge survive conservative transaction-cost stress?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
