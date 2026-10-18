"""Cost-sensitive momentum lookback sweep across the collected universe.

Objective: Test whether the momentum edge observed at lookback=5
(`research/checks/momentum.py`, activation 37318950814) survives conservative
transaction-cost stress on the real-data universe. Momentum is a very
mechanical 1-day-ahead bet, so the most salient robustness question is whether
the positive REGIME_STABLE pattern at lookback=5 also appears at lookback 3 and
lookback 10 when realistic trading costs are applied.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as `momentum.py`; realistic transaction costs.
Determinism is asserted via an internal independent recomputation.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_cost_results.json"

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
    risk_free=0.0,
    periods_per_year=252,
)

NULL_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
)


def segment_result_with_costs(ticker, bars, lookback, costs):
    """Walk-forward regime-stability result for one ticker, lookback, and costs.

    Uses a separate cost configuration than the null benchmark.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    # Candidate with costs
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
        cfg=costs,
    )

    # Matched null with same cost parameters
    null_res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=0,  # no warmup for null
        overlap_window=0,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=costs,
    )

    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    ), dict(
        segments=[s.name for s in null_res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in null_res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in null_res.scenarios],
        candidate_dispersion=round(null_res.candidate_dispersion, 3),
        null_dispersion=round(null_res.null_dispersion, 3),
        n_folds=null_res.n_folds,
        verdict=null_res.overall_verdict,
    )


def cost_verdict(sr):
    """Cost-robustness verdict for one lookback's scenario result.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge despite costs; REGIME_STABLE with uniformly negative
    medians is a stable loss despite costs, not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "COST_ROBUST_EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "COST_ROBUST_LOSS"
    return "COST_ROBUST_NO_EDGE"


def robustness_verdict(lookback_results):
    """Cost-robustness verdict for one asset across lookbacks.

    ROBUST      : uniformly positive REGIME_STABLE edges at >= 2 lookbacks despite costs.
    SENSITIVE   : a positive REGIME_STABLE edge at exactly one lookback despite costs.
    NO_EDGE     : no lookback shows a positive REGIME_STABLE edge despite costs.
    """
    counts = {"COST_ROBUST_EDGE": 0, "COST_ROBUST_LOSS": 0, "COST_ROBUST_NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[cost_verdict(lookback_results[str(lb)])] += 1
    if counts["COST_ROBUST_EDGE"] >= 2:
        return "COST_ROBUST", counts
    if counts["COST_ROBUST_EDGE"] == 1:
        return "COST_SENSITIVE", counts
    return "COST_NO_EDGE", counts


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

    print("\n=== 4. Cost-sensitive momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a matched null "
          "with the same costs\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost-sensitive lookback sweep.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][candidate/null]
    lookback_results = {}

    for ticker in ticker_order:
        bars = all_tickers[ticker]
        for lb in LOOKBACKS:
            if lb not in lookback_results:
                lookback_results[lb] = {}
            cand, null = segment_result_with_costs(ticker, bars, lb, REALISTIC_COSTS)
            lookback_results[lb][ticker] = {"candidate": cand, "null": null}

    print("  cost-sensitive lookback sweep completed.")

    # Summarize per-asset cost-robustness
    per_asset = {}
    cost_robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker]["candidate"] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        cost_robustness_counts[v] += 1
        l5 = profile["5"]
        lookback_5_counts[l5["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["cost_robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        profile["lookback_candidate_dispersions"] = {str(lb): profile[str(lb)]["candidate_dispersion"] for lb in LOOKBACKS}
        profile["lookback_null_dispersions"] = {str(lb): profile[str(lb)]["null_dispersion"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  cost-robustness verdict counts: "
          f"COST_ROBUST={cost_robustness_counts['COST_ROBUST']} "
          f"(positive edge at >=2 lookbacks despite costs) | "
          f"COST_SENSITIVE={cost_robustness_counts['COST_SENSITIVE']} "
          f"(edge at exactly one lookback despite costs) | "
          f"COST_NO_EDGE={cost_robustness_counts['COST_NO_EDGE']}\n")
    print("  Per-asset perturbation profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} cost_rob={:<15s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["cost_robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result_with_costs(t, tickers[t], lb, REALISTIC_COSTS)[0]
                 for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t]["candidate"] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Build artifact for independent verification.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs=dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            cost_robustness_counts=cost_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                         for lb in LOOKBACKS},
                    lookback_candidate_dispersions={str(lb): per_asset[t]["lookback_candidate_dispersions"][str(lb)]
                                                     for lb in LOOKBACKS},
                    lookback_null_dispersions={str(lb): per_asset[t]["lookback_null_dispersions"][str(lb)]
                                                for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    cost_robustness_verdict=per_asset[t]["cost_robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost lookback sweep across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())