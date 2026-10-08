"""Real-data momentum lookback cost-robustness sweep across the collected universe.

Objective: test whether the momentum edge (lookback 3/5/10) survives realistic
transaction costs (fixed cost model) on the collected 10-asset universe.

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
daily rebalance) is walk-forward validated (train=252d/test=84d/warmup=60d/
overlap=60d) under both zero cost and realistic cost models; each cost variant's
per-segment median log return is compared against a matched coin-flip null benchmark
run on the same segments (separate per-cost variant). The cost-robustness edge is
one where the uniformly-positive REGIME_STABLE pattern appears for both cost models;
for lookback-robustness, the edge must survive across >=2 lookbacks under the
cost model.

Method (fixed a-priori, not tuned to OOS): same walk-forward as momentum_lookback_sweep.
Determinism is asserted via an internal independent recomputation of the AMZN/JPM subset.
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_cost_robustness_results.json"


def zero_cost_config():
    """Zero-cost config (the baseline for comparison)."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )


def realistic_cost_config():
    """Realistic cost config (fixed cost model from repository examples)."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )


def segment_result_with_costs(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, and one cost config."""
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


def lookback_verdict(sr):
    """Verdict for one lookback's scenario result: EDGE if REGIME_STABLE and uniformly positive."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one cost config.

    ROBUST      : uniformly positive REGIME_STABLE edges at >=2 lookbacks.
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


def cost_robustness_verdict(zerocost_results, cost_results):
    """Overall verdict: COST_ROBUST if both configs show the same verdict (>=2 lookbacks positive)."""
    z_verdict, z_counts = robustness_verdict(zerocost_results)
    c_verdict, c_counts = robustness_verdict(cost_results)
    if z_verdict == "ROBUST" and c_verdict == "ROBUST":
        return "COST_ROBUST", z_counts, c_counts
    if z_verdict == "SENSITIVE" and c_verdict == "SENSITIVE":
        return "COST_SENSITIVE", z_counts, c_counts
    if z_verdict == "NO_EDGE" and c_verdict == "NO_EDGE":
        return "COST_NO_EDGE", z_counts, c_counts
    # Mixed verdicts are different outcomes
    return "COST_MIXED", z_counts, c_counts


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
    from preflight import main as pf_main
    rc = pf_main()
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
        cfg0 = zero_cost_config()
        res = bt.run_bars(list(bars), signals, cfg0)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-robustness sweep (lookback 3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks; zero-cost and realistic-cost config; "
          "each cost variant vs a matched coin-flip null.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load the full universe of 10 collected tickers
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results for zero-cost and cost models
    zerocost_results = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, zero_cost_config())
                            for t in ticker_order}
                       for lb in LOOKBACKS}
    cost_results = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, realistic_cost_config())
                       for t in ticker_order}
                   for lb in LOOKBACKS}

    per_asset = {}
    cost_robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_NO_EDGE": 0, "COST_MIXED": 0}
    lookback_5_counts_zerocost = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                   "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    lookback_5_counts_cost = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                              "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        zerocost_profile = {str(lb): zerocost_results[lb][ticker] for lb in LOOKBACKS}
        cost_profile = {str(lb): cost_results[lb][ticker] for lb in LOOKBACKS}

        # Verdicts for each cost config
        z_verdict, z_counts = robustness_verdict(zerocost_profile)
        c_verdict, c_counts = robustness_verdict(cost_profile)
        overall_verdict, _, _ = cost_robustness_verdict(zerocost_profile, cost_profile)
        cost_robustness_counts[overall_verdict] += 1

        # Lookback-5 verdict counts for each cost config
        lookback_5_counts_zerocost[zerocost_profile["5"]["verdict"]] += 1
        lookback_5_counts_cost[cost_profile["5"]["verdict"]] += 1

        # Store for artifact and reporting
        per_asset[ticker] = dict(
            zerocost=dict(
                segments_5=zerocost_profile["5"]["segments"],
                medians_5=zerocost_profile["5"]["medians"],
                null_medians_5=zerocost_profile["5"]["null_medians"],
                verdict_5=zerocost_profile["5"]["verdict"],
                edge_counts_5=z_counts,
                robustness_verdict_5=z_verdict,
                n_folds_5=zerocost_profile["5"]["n_folds"],
            ),
            cost=dict(
                segments_5=cost_profile["5"]["segments"],
                medians_5=cost_profile["5"]["medians"],
                null_medians_5=cost_profile["5"]["null_medians"],
                verdict_5=cost_profile["5"]["verdict"],
                edge_counts_5=c_counts,
                robustness_verdict_5=c_verdict,
                n_folds_5=cost_profile["5"]["n_folds"],
            ),
            overall_cost_robustness_verdict=overall_verdict,
            lookback_medians_5=zerocost_profile["5"]["medians"],
            lookback_null_medians_5=zerocost_profile["5"]["null_medians"],
        )

    print("  lookback 5 (reference) verdict counts across universe: zero-cost {}".format(lookback_5_counts_zerocost))
    print("                lookback 5 (reference) verdict counts across universe: cost {}".format(lookback_5_counts_cost))
    print("  cost-robustness verdict counts: "
          f"COST_ROBUST={cost_robustness_counts['COST_ROBUST']} "
          f"(positive edge for both zero-cost and cost) | "
          f"COST_SENSITIVE={cost_robustness_counts['COST_SENSITIVE']} "
          f"(positive edge for both cost models but not lookback-robust) | "
          f"COST_NO_EDGE={cost_robustness_counts['COST_NO_EDGE']} "
          f"(no positive edge for both) | "
          f"COST_MIXED={cost_robustness_counts['COST_MIXED']}\n")

    print("  Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        z_med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["zerocost"]["medians_5"][0], p["zerocost"]["verdict_5"])
            for lb in LOOKBACKS
        )
        c_med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["cost"]["medians_5"][0], p["cost"]["verdict_5"])
            for lb in LOOKBACKS
        )
        print("    {:6s} zero_cost_rob={:<8s} cost_rob={:<8s} overall={:<12s} z_meds={:<50s} c_meds={:<50s}".format(
            ticker, p["zerocost"]["robustness_verdict_5"], p["cost"]["robustness_verdict_5"],
            p["overall_cost_robustness_verdict"], z_med, c_med))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2_zerocost = {lb: {t: segment_result_with_costs(t, tickers[t], lb, zero_cost_config())
                         for t in target}
                    for lb in LOOKBACKS}
    res2_cost = {lb: {t: segment_result_with_costs(t, tickers[t], lb, realistic_cost_config())
                     for t in target}
                for lb in LOOKBACKS}

    out1_zerocost = json.dumps({lb: {t: zerocost_results[lb][t] for t in target}
                               for lb in LOOKBACKS}, sort_keys=True)
    out2_zerocost = json.dumps(res2_zerocost, sort_keys=True)
    print("determinism (zero-cost): r1 == r2: {}".format(out1_zerocost == out2_zerocost))
    assert out1_zerocost == out2_zerocost, "non-deterministic zero-cost output"

    out1_cost = json.dumps({lb: {t: cost_results[lb][t] for t in target}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_cost = json.dumps(res2_cost, sort_keys=True)
    print("determinism (cost): r1 == r2: {}".format(out1_cost == out2_cost))
    assert out1_cost == out2_cost, "non-deterministic cost output"

    # Create the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts_zerocost=lookback_5_counts_zerocost,
            lookback_5_counts_cost=lookback_5_counts_cost,
            cost_robustness_counts=cost_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    zerocost_robustness_verdict=per_asset[t]["zerocost"]["robustness_verdict_5"],
                    cost_robustness_verdict=per_asset[t]["cost"]["robustness_verdict_5"],
                    overall_cost_robustness_verdict=per_asset[t]["overall_cost_robustness_verdict"],
                    zerocost_segments_5=per_asset[t]["zerocost"]["segments_5"],
                    cost_segments_5=per_asset[t]["cost"]["segments_5"],
                    zerocost_null_medians_5=per_asset[t]["zerocost"]["null_medians_5"],
                    cost_null_medians_5=per_asset[t]["cost"]["null_medians_5"],
                    zerocost_n_folds_5=per_asset[t]["zerocost"]["n_folds_5"],
                    cost_n_folds_5=per_asset[t]["cost"]["n_folds_5"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback cost-robustness across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())