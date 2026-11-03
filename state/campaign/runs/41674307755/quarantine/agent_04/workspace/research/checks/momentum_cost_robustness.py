"""Cost-robustness test for lookback 3 / 5 / 10 momentum on the collected universe.

Objective: test whether the momentum edge observed at lookback=5
(`research/checks/momentum.py`, activation 37318950814) survives realistic
transaction costs (commission, slippage) and remains distinguishable from
a matched null.

For each collected ticker and each lookback N in {3, 5, 10}, momentum
(long the previous N-day return, hold 1 day, daily rebalance) is walk-forward
validated (train=252d / test=84d / warmup=60d / overlap=60d) inside each
of the 4 contiguous volatility blocks of the real-data regime family.

Two backtest configurations are run:
- cfg0: zero cost (commission_per_trade=0, commission_per_share=0, slippage_cents=0, slippage_proportional=0)
- cfg1: realistic costs (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)

Each configuration is compared against a coin-flip null benchmark run on the same segments. The lookback-robustness verdict is determined by:
- REGIME_STABLE edge at >= 2 lookbacks under cfg0 (no cost) → candidate edge.
- If an edge exists under cfg0, check cfg1:
    - CONSISTENT_WITH_NOISE: indistinguishable from noise under realistic costs.
    - REGIME_STABLE_LOSS: uniform losses under realistic costs.
    - REGIME_DEPENDENT or REGIME_STABLE edge: the edge survives costs.
- NO_EDGE under cfg0: no edge to cost-test.

The cost-robustness classification directly informs whether to retain momentum
in the evidence base: cost-robust edges are qualified as candidate positive evidence.

Method (fixed a-priori, not tuned to OOS): same walk-forward, same volatility blocks, same coin-flip null.

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

# Zero cost configuration (baseline for comparison)
cfg0 = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)

# Realistic costs configuration (conservative transaction costs)
cfg1 = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def segment_result_with_costs(
    ticker: str,
    bars: bt.BarSequence,
    lookback: int,
    cfg: bt.BacktestConfig,
) -> dict:
    """Walk-forward regime-stability result for one ticker, lookback, and cost config."""
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
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def lookback_verdict(sr: dict, cost_config: str) -> str:
    """Cost-robustness verdict for one lookback's scenario result.

    Under zero cost (cfg0):
        REGIME_STABLE with uniformly positive medians → EDGE
        REGIME_STABLE_LOSS with uniformly negative medians → LOSS
        All others → NO_EDGE

    Under realistic costs (cfg1):
        Same verdict logic, but CONSISTENT_WITH_NOISE indicates cost-eroded edge.
    """
    if sr["verdict"] == "REGIME_STABLE":
        if all(m > 0 for m in sr["medians"]):
            return "EDGE"
        if all(m < 0 for m in sr["medians"]):
            return "LOSS"
        return "NO_EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cost_robustness_verdict(lookback_results_no_cost: dict, lookback_results_real_cost: dict) -> tuple[str, dict]:
    """Cost-robustness verdict for one asset.

    Decision logic:
    1. If no edge under zero cost (cfg0): NO_EDGE.
    2. If edge under zero cost (cfg0) but LOSS under real cost: COST_ERODED (edge vanished).
    3. If edge under both: REGIME_STABLE (cost-robust edge).
    4. If edge under zero cost but CONSISTENT_WITH_NOISE/REGIME_DEPENDENT under real cost: COST_SENSITIVE (edge eroded but not eliminated).
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0, "COST_ERODED": 0, "COST_SENSITIVE": 0}

    for lb in LOOKBACKS:
        verdict_no_cost = lookback_verdict(lookback_results_no_cost[str(lb)], "cfg0")
        verdict_real_cost = lookback_verdict(lookback_results_real_cost[str(lb)], "cfg1")

        if verdict_no_cost == "EDGE":
            counts["EDGE"] += 1
            if verdict_real_cost == "EDGE":
                counts["REGIME_STABLE"] = counts.get("REGIME_STABLE", 0) + 1
            elif verdict_real_cost == "LOSS":
                counts["COST_ERODED"] += 1
            else:  # CONSISTENT_WITH_NOISE or REGIME_DEPENDENT
                counts["COST_SENSITIVE"] += 1
        elif verdict_no_cost == "LOSS":
            counts["LOSS"] += 1
        else:
            counts["NO_EDGE"] += 1

    # Final verdict
    if counts["REGIME_STABLE"] >= 2:
        return "COST_ROBUST", counts
    if counts["COST_SENSITIVE"] >= 2:
        return "COST_SENSITIVE", counts
    if counts["COST_ERODED"] >= 2:
        return "COST_ERODED", counts
    return "NO_EDGE", counts


def main() -> int:
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Momentum lookback sweep with cost comparison (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "under two cost regimes (zero vs realistic)\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Run both cost regimes
    print("  Running zero-cost configuration...")
    results_no_cost = {}
    for ticker in ticker_order:
        results_no_cost[ticker] = {str(lb): segment_result_with_costs(ticker, all_tickers[ticker], lb, cfg0)
                                   for lb in LOOKBACKS}

    print("  Running realistic-cost configuration...")
    results_real_cost = {}
    for ticker in ticker_order:
        results_real_cost[ticker] = {str(lb): segment_result_with_costs(ticker, all_tickers[ticker], lb, cfg1)
                                     for lb in LOOKBACKS}

    # Compute cost-robustness verdicts
    per_asset = {}
    cost_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_ERODED": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile_no_cost = {str(lb): results_no_cost[ticker][str(lb)] for lb in LOOKBACKS}
        profile_real_cost = {str(lb): results_real_cost[ticker][str(lb)] for lb in LOOKBACKS}

        verdict, counts = cost_robustness_verdict(profile_no_cost, profile_real_cost)
        cost_counts[verdict] += 1

        # Capture lookback-5 baseline verdicts for reference
        r5_no_cost = profile_no_cost["5"]
        lookback_5_counts[r5_no_cost["verdict"]] += 1

        per_asset[ticker] = {
            "lookback_5_baseline_verdict": r5_no_cost["verdict"],
            "lookback_5_baseline_edge": r5_no_cost["medians"][0],
            "cost_robustness_verdict": verdict,
            "cost_edge_counts": counts,
            "lookback_medians_no_cost": {str(lb): profile_no_cost[str(lb)]["medians"] for lb in LOOKBACKS},
            "lookback_medians_real_cost": {str(lb): profile_real_cost[str(lb)]["medians"] for lb in LOOKBACKS},
        }

    print("  Lookback 5 baseline verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  Cost-robustness verdict counts: "
          f"COST_ROBUST={cost_counts['COST_ROBUST']} "
          f"(edge under both regimes) | "
          f"COST_SENSITIVE={cost_counts['COST_SENSITIVE']} "
          f"(edge vanished under realistic costs) | "
          f"COST_ERODED={cost_counts['COST_ERODED']} "
          f"(uniform losses under realistic costs) | "
          f"NO_EDGE={cost_counts['NO_EDGE']}\n")

    # Determinism check
    print("\n=== 4. Determinism check (rerun target subset) ===")
    target = ["AMZN", "JPM", "AAPL", "TSLA", "GOOGL"]
    res2_no_cost = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, cfg0)
                        for t in target} for lb in LOOKBACKS}
    res2_real_cost = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, cfg1)
                          for t in target} for lb in LOOKBACKS}

    out1_no_cost = json.dumps({lb: {t: results_no_cost[lb][t] for t in target}
                               for lb in LOOKBACKS}, sort_keys=True)
    out2_no_cost = json.dumps(res2_no_cost, sort_keys=True)
    out1_real_cost = json.dumps({lb: {t: results_real_cost[lb][t] for t in target}
                                 for lb in LOOKBACKS}, sort_keys=True)
    out2_real_cost = json.dumps(res2_real_cost, sort_keys=True)

    print(f"determinism zero-cost: r1 == r2: {out1_no_cost == out2_no_cost}")
    print(f"determinism real-cost: r1 == r2: {out1_real_cost == out2_real_cost}")
    assert out1_no_cost == out2_no_cost, "non-deterministic output (zero cost)"
    assert out1_real_cost == out2_real_cost, "non-deterministic output (real cost)"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cfg0=dict(
            commission_per_trade=cfg0.commission_per_trade,
            commission_per_share=cfg0.commission_per_share,
            slippage_cents=cfg0.slippage_cents,
            slippage_proportional=cfg0.slippage_proportional,
        ),
        cfg1=dict(
            commission_per_trade=cfg1.commission_per_trade,
            commission_per_share=cfg1.commission_per_share,
            slippage_cents=cfg1.slippage_cents,
            slippage_proportional=cfg1.slippage_proportional,
        ),
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            cost_robustness_counts=cost_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_5_baseline_verdict=per_asset[t]["lookback_5_baseline_verdict"],
                    lookback_5_baseline_edge=per_asset[t]["lookback_5_baseline_edge"],
                    cost_robustness_verdict=per_asset[t]["cost_robustness_verdict"],
                    lookback_medians_no_cost=per_asset[t]["lookback_medians_no_cost"],
                    lookback_medians_real_cost=per_asset[t]["lookback_medians_real_cost"],
                    edge_counts=per_asset[t]["cost_edge_counts"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness test across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
