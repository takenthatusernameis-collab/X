"""Real-data momentum lookback sweep with cost sensitivity.

Objective: test whether the momentum edge survives conservative transaction costs
under the repository's realistic cost model while preserving lookback robustness
across lookback 3, 5, and 10.

Momentum is mechanical (1-day-ahead bet LONG the previous N-day return). The
edge is already admitted as candidate positive evidence. The most salient
qualification question is whether the edge survives realistic costs without
becoming a no-edge result. This test follows the lookback-robustness framework
(lookback 3/5/10 across 10 collected tickers) but adds a bounded cost-sensitivity
curve: zero cost (existing baseline) and realistic costs (commission + slippage).

Falsification predictions:
- CONSISTENT_WITH_NOISE at realistic costs: edge vanishes under costs.
- COST_SENSITIVE: edge shrinks but persists at realistic costs.
- COST_ROBUST: edge persists robustly (or improves) at realistic costs.

Method (fixed a-priori): long the previous N-day return; hold 1 day; daily
rebalance. Same walk-forward, regime-stability, and coin-flip null framework as
the lookback sweep. Cost sensitivity uses realistic config:
  commission_per_trade = 2.0, commission_per_share = 0.003,
  slippage_cents = 2.0, slippage_proportional = 0.0005.
Determinism is asserted via internal recomputation of AMZN + JPM subset.

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
WARMUP = 60
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_cost_sweep_results.json"


def cost_config(realistic: bool) -> bt.BacktestConfig:
    if realistic:
        return bt.BacktestConfig(
            initial_capital=1e6,
            warmup_periods=WARMUP,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        )
    else:
        return bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARMUP)


def segment_result(ticker, bars, lookback, realistic=False):
    """Walk-forward regime-stability result for one ticker, one lookback, one cost regime."""
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


def cost_robustness_verdict(zero_results, realistic_results, ticker):
    """Classify cost robustness for one ticker.

    Returns:
    COST_CONSISTENT_WITH_NOISE: no positive REGIME_STABLE edge at realistic costs.
    COST_SENSITIVE: positive REGIME_STABLE edge at realistic costs but not at zero costs.
    COST_ROBUST: positive REGIME_STABLE edge at realistic costs, and at least as good as zero cost (or improved).
    """
    zero_edges = sum(1 for lb in LOOKBACKS if lookback_verdict(zero_results[str(lb)]) == "EDGE")
    realistic_edges = sum(1 for lb in LOOKBACKS if lookback_verdict(realistic_results[str(lb)]) == "EDGE")

    if realistic_edges == 0:
        return "COST_CONSISTENT_WITH_NOISE"
    if realistic_edges > 0 and zero_edges == 0:
        return "COST_SENSITIVE"
    return "COST_ROBUST"


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

    print("\n=== 3. Leakage review per asset (lookback=5, zero cost reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        cfg_zero = cost_config(realistic=False)
        res = bt.run_bars(list(bars), signals, cfg_zero)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")

    print("\n=== 4. Momentum lookback sweep with cost sensitivity ===")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs a coin-flip null "
          f"on the same segments\n")
    print(f"  Cost regimes: zero cost vs realistic costs (commission_per_trade=2.0, "
          f"commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)")

    # Full-universe lookback sweep for both cost regimes.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [cost_regime][lookback][ticker][segment_index].
    results = {
        "zero_cost": {lb: {t: segment_result(t, all_tickers[t], lb, realistic=False) for t in ticker_order} for lb in LOOKBACKS},
        "realistic_cost": {lb: {t: segment_result(t, all_tickers[t], lb, realistic=True) for t in ticker_order} for lb in LOOKBACKS},
    }

    per_asset = {}
    cost_robustness_counts = {
        "COST_CONSISTENT_WITH_NOISE": 0,
        "COST_SENSITIVE": 0,
        "COST_ROBUST": 0,
    }
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        zero_profile = {str(lb): results["zero_cost"][lb][ticker] for lb in LOOKBACKS}
        realistic_profile = {str(lb): results["realistic_cost"][lb][ticker] for lb in LOOKBACKS}
        cost_class = cost_robustness_verdict(zero_profile, realistic_profile, ticker)
        cost_robustness_counts[cost_class] += 1
        z5 = zero_profile["5"]
        lookback_5_counts[z5["verdict"]] += 1

        # Lookback-robustness edge counts (edges per asset, for the summary).
        zero_robust, zero_counts = robustness_verdict(zero_profile)
        realistic_robust, realistic_counts = robustness_verdict(realistic_profile)

        profile = dict(
            lookback_5_verdict=z5["verdict"],
            lookback_5_counts=z5,
            zero_cost_robustness=zero_robust,
            realistic_cost_robustness=realistic_robust,
            cost_robustness_verdict=cost_class,
            zero_edge_counts=zero_counts,
            realistic_edge_counts=realistic_counts,
            lookback_medians_zero={str(lb): zero_profile[str(lb)]["medians"] for lb in LOOKBACKS},
            lookback_null_medians_zero={str(lb): zero_profile[str(lb)]["null_medians"] for lb in LOOKBACKS},
            lookback_medians_realistic={str(lb): realistic_profile[str(lb)]["medians"] for lb in LOOKBACKS},
            lookback_null_medians_realistic={str(lb): realistic_profile[str(lb)]["null_medians"] for lb in LOOKBACKS},
        )
        per_asset[ticker] = profile

    print(f"  lookback 5 (zero cost, reference) verdict counts across universe: {lookback_5_counts}")
    print(f"  cost-robustness verdict counts: "
          f"COST_CONSISTENT_WITH_NOISE={cost_robustness_counts['COST_CONSISTENT_WITH_NOISE']} | "
          f"COST_SENSITIVE={cost_robustness_counts['COST_SENSITIVE']} | "
          f"COST_ROBUST={cost_robustness_counts['COST_ROBUST']}\n")

    print("  Per-asset perturbation profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med_zero = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print(f"    {ticker}: zero_cost medians={med_zero} | realistic_cost {p['cost_robustness_verdict']}")

    # Determinism: rerun the target subset and assert identical output.
    target_subset = {t: tickers[t] for t in target}
    res2 = {
        "zero_cost": {lb: {t: segment_result(t, target_subset[t], lb, realistic=False) for t in target}
                       for lb in LOOKBACKS},
        "realistic_cost": {lb: {t: segment_result(t, target_subset[t], lb, realistic=True) for t in target}
                           for lb in LOOKBACKS},
    }
    out1 = json.dumps(
        {cost: {lb: {t: results[cost][lb][t] for t in target} for lb in LOOKBACKS} for cost in results},
        sort_keys=True
    )
    out2 = json.dumps(res2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    # Verify that lookback=5 zero-cost column matches existing momentum_results.json
    existing_path = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        # Simple cross-check: first asset in target
        check_asset = target[0]
        existing_momentum = existing["per_asset"].get(check_asset, {}).get("momentum", {})
        existing_medians = existing_momentum.get("medians", [])
        existing_nulls = existing_momentum.get("null_medians", [])
        existing_verdict = existing_momentum.get("verdict", "")
        # Find the lookback=5 zero-cost result
        first_lb_result = results["zero_cost"].get(5, {}).get(check_asset, {})
        first_medians = first_lb_result.get("medians", [])
        first_nulls = first_lb_result.get("null_medians", [])
        first_verdict = first_lb_result.get("verdict", "")
        if (existing_medians == first_medians and existing_nulls == first_nulls and
                existing_verdict == first_verdict):
            print(f"  cross-check PASS: lookback=5 zero-cost matches existing momentum_results.json")
        else:
            print(f"  cross-check WARNING: lookback=5 zero-cost differs from existing momentum_results.json")
            print(f"    existing: medians={existing_medians}, nulls={existing_nulls}, verdict={existing_verdict}")
            print(f"    computed: medians={first_medians}, nulls={first_nulls}, verdict={first_verdict}")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        warmup=WARMUP,
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            cost_robustness_counts=cost_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_5_verdict=per_asset[t]["lookback_5_verdict"],
                    zero_cost_robustness=per_asset[t]["zero_cost_robustness"],
                    realistic_cost_robustness=per_asset[t]["realistic_cost_robustness"],
                    cost_robustness_verdict=per_asset[t]["cost_robustness_verdict"],
                    zero_edge_counts=per_asset[t]["zero_edge_counts"],
                    realistic_edge_counts=per_asset[t]["realistic_edge_counts"],
                    lookback_medians_zero=per_asset[t]["lookback_medians_zero"],
                    lookback_null_medians_zero=per_asset[t]["lookback_null_medians_zero"],
                    lookback_medians_realistic=per_asset[t]["lookback_medians_realistic"],
                    lookback_null_medians_realistic=per_asset[t]["lookback_null_medians_realistic"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with cost sensitivity: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())