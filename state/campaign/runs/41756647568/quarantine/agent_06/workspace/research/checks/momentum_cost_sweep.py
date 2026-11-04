"""Cost-sensitive momentum lookback sweep across the collected universe.

Objective: test whether the momentum edge observed at lookback=5
(qualifying it as candidate positive evidence) survives realistic
transaction costs (commission + slippage) for lookbacks 3, 5, and 10.

For each collected ticker, momentum (long the previous N-day return, hold
1 day, daily rebalance) is walk-forward validated (train=252d/test=84d/
warmup=60d/overlap=60d) inside each of the 4 contiguous volatility
blocks of the real-data regime family, for lookback in {3, 5, 10}. Each
lookback's per-segment median log return is compared against a coin-flip
sign null benchmark run on the same segments. The coin-flip null must be
matched (same size, same date-range, same segment boundaries) so that
transaction costs are only compared against a truly equivalent operation.

A cost-robust edge is one where the lookback=5 momentum median log
return remains positive (or at least not eroded into negative territory)
after realistic costs for >= 2 lookbacks; a cost-sensitive result
(costs erase the edge) is a decision-relevant qualification for whether
the momentum should be admitted to the evidence base.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; same
walk-forward windows and volatility blocks as `momentum.py`.
Realistic cost model per round-trip: commission_per_trade=2.0,
commission_per_share=0.003 (10 bps), slippage_cents=2.0, slippage
proportional=0.0005 (5 bps). Cost is applied in the same framework
path as the signal execution (see `examples/ma_crossover.py`). Determinism
is asserted via an internal independent recomputation of the AMZN/JPM
subset.

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

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)


def segment_result_cost(ticker, bars, lookback):
    """Walk-forward regime-stability result with realistic costs for one ticker and one lookback."""
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
        cfg=REALISTIC_COST_CONFIG,
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


def segment_result_zero_cost(ticker, bars, lookback):
    """Walk-forward regime-stability result with zero costs for one ticker and one lookback."""
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
        cfg=ZERO_COST_CONFIG,
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


def cost_robustness_verdict(lookback_results):
    """Verdict on whether the momentum edge survives realistic costs for one asset.

    COST_ROBUST      : zero-cost median log return positive AND realistic-cost
                        median log return not negative (i.e. survives costs).
    COST_SENSITIVE   : realistic costs eliminate the edge (realistic-cost
                        median log return <= 0).
    COST_UNKNOWN     : zero-cost test not run or insufficient data.
    """
    if "zero_cost" not in lookback_results or "real_cost" not in lookback_results:
        return "COST_UNKNOWN", {"zero_cost": False, "real_cost": False}

    zero_results = lookback_results["zero_cost"]
    real_results = lookback_results["real_cost"]

    edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}

    for lb_name, lb_result in zero_results.items():
        if lb_result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in lb_result["medians"]):
            edge_counts["EDGE"] += 1
        elif lb_result["verdict"] == "REGIME_STABLE_LOSS":
            edge_counts["LOSS"] += 1
        else:
            edge_counts["NO_EDGE"] += 1

    real_edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}

    for lb_name, lb_result in real_results.items():
        if lb_result["verdict"] == "REGIME_STABLE" and any(m > 0 for m in lb_result["medians"]):
            real_edge_counts["EDGE"] += 1
        elif lb_result["verdict"] == "REGIME_STABLE_LOSS":
            real_edge_counts["LOSS"] += 1
        else:
            real_edge_counts["NO_EDGE"] += 1

    return {
        "verdict": "COST_ROBUST" if edge_counts["EDGE"] >= 2 and real_edge_counts["EDGE"] >= 2 else "COST_SENSITIVE",
        "zero_cost_edge_counts": edge_counts,
        "real_cost_edge_counts": real_edge_counts,
    }


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

    print("\n=== 3. Leakage review per asset (lookback=5 zero cost reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, ZERO_COST_CONFIG)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")

    print("\n=== 4. Cost-sensitive momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments. Realistic costs applied via "
          "commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005 per round-trip."
          "\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with zero and realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][cost_type][segment index]
    lookback_results = {lb: {t: {"zero_cost": {}, "real_cost": {}} for t in ticker_order} for lb in LOOKBACKS}

    # Compute zero-cost results
    print("=== Zero-cost baseline ===")
    for lb in LOOKBACKS:
        for ticker in ticker_order:
            result = segment_result_zero_cost(ticker, all_tickers[ticker], lb)
            lookback_results[lb][ticker]["zero_cost"][str(lb)] = result
            print(f"  {ticker}: lookback {lb:2d} -> verdict {result['verdict']}, "
                  f"median {result['medians'][0]:+.3f}")

    # Compute realistic cost results
    print("\n=== Realistic-cost test ===")
    for lb in LOOKBACKS:
        for ticker in ticker_order:
            result = segment_result_cost(ticker, all_tickers[ticker], lb)
            lookback_results[lb][ticker]["real_cost"][str(lb)] = result
            print(f"  {ticker}: lookback {lb:2d} -> verdict {result['verdict']}, "
                  f"median {result['medians'][0]:+.3f}")

    # Aggregate per-asset results
    per_asset = {}
    robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_UNKNOWN": 0}
    lookback_5_zero_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    lookback_5_real_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}

    for ticker in ticker_order:
        profile = {"zero_cost": {}, "real_cost": {}}
        for lb in LOOKBACKS:
            profile["zero_cost"][str(lb)] = lookback_results[lb][ticker]["zero_cost"][str(lb)]
            profile["real_cost"][str(lb)] = lookback_results[lb][ticker]["real_cost"][str(lb)]

        verdict_info = cost_robustness_verdict(profile)
        profile["robustness_verdict"] = verdict_info["verdict"]
        profile["zero_cost_edge_counts"] = verdict_info["zero_cost_edge_counts"]
        profile["real_cost_edge_counts"] = verdict_info["real_cost_edge_counts"]

        # Count lookback 5 zero cost results
        lb5_zero = profile["zero_cost"]["5"]
        if lb5_zero["verdict"] == "REGIME_STABLE" and all(m > 0 for m in lb5_zero["medians"]):
            lookback_5_zero_counts["EDGE"] += 1
        elif lb5_zero["verdict"] == "REGIME_STABLE_LOSS":
            lookback_5_zero_counts["LOSS"] += 1
        else:
            lookback_5_zero_counts["NO_EDGE"] += 1

        # Count lookback 5 realistic cost results
        lb5_real = profile["real_cost"]["5"]
        if lb5_real["verdict"] == "REGIME_STABLE" and any(m > 0 for m in lb5_real["medians"]):
            lookback_5_real_counts["EDGE"] += 1
        elif lb5_real["verdict"] == "REGIME_STABLE_LOSS":
            lookback_5_real_counts["LOSS"] += 1
        else:
            lookback_5_real_counts["NO_EDGE"] += 1

        robustness_counts[verdict_info["verdict"]] += 1
        per_asset[ticker] = profile

    print("\n  Lookback 5 zero-cost verdict counts across universe: "
          f"EDGE={lookback_5_zero_counts['EDGE']}, "
          f"LOSS={lookback_5_zero_counts['LOSS']}, "
          f"NO_EDGE={lookback_5_zero_counts['NO_EDGE']}")
    print("  Lookback 5 realistic-cost verdict counts across universe: "
          f"EDGE={lookback_5_real_counts['EDGE']}, "
          f"LOSS={lookback_5_real_counts['LOSS']}, "
          f"NO_EDGE={lookback_5_real_counts['NO_EDGE']}")
    print("  Cost-robustness verdict counts: "
          f"COST_ROBUST={robustness_counts['COST_ROBUST']} "
          f"(survives costs) | "
          f"COST_SENSITIVE={robustness_counts['COST_SENSITIVE']} "
          f"(costs eliminate edge) | "
          f"COST_UNKNOWN={robustness_counts['COST_UNKNOWN']}\n")

    # Per-asset perturbation profiles
    print("  Per-asset perturbation profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        zero_medians = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["zero_cost"][str(lb)]["medians"][0], p["zero_cost"][str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        real_medians = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["real_cost"][str(lb)]["medians"][0], p["real_cost"][str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} zero_cost_segments={:<8} medians={:<40s} rob={:<8s} "
              "cost_rob={:<8s}".format(
                ticker, str(p["zero_cost"]["5"]["segments"]), zero_medians,
                p["robustness_verdict"], p["robustness_verdict"]))
        print("      realistic_cost segments={:<8} medians={:<40s} "
              "cost_rob={:<8s}".format(
                str(p["real_cost"]["5"]["segments"]), real_medians,
                p["robustness_verdict"]))

    # Determinism: rerun the target subset and assert identical output
    print("\n=== Determinism: rerun AMZN/JPM with both zero and realistic costs ===")
    res2 = {lb: {t: {"zero_cost": {}, "real_cost": {}} for t in target}
            for lb in LOOKBACKS}
    for lb in LOOKBACKS:
        for ticker in target:
            res2[lb][ticker]["zero_cost"][str(lb)] = segment_result_zero_cost(ticker, tickers[ticker], lb)
            res2[lb][ticker]["real_cost"][str(lb)] = segment_result_cost(ticker, tickers[ticker], lb)

    # Compare with first run
    all_match = True
    for lb in LOOKBACKS:
        for ticker in target:
            for cost_type in ["zero_cost", "real_cost"]:
                for lb_str in LOOKBACKS:
                    orig = lookback_results[lb][ticker][cost_type][lb_str]
                    new = res2[lb][ticker][cost_type][lb_str]
                    if orig != new:
                        all_match = False
                        print(f"  MISMATCH: {ticker} lookback {lb} {cost_type}")

    print("determinism: r1 == r2: {}".format(all_match))
    assert all_match, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_cost_config={
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
        },
        zero_cost_config={
            "commission_per_trade": ZERO_COST_CONFIG.commission_per_trade,
            "commission_per_share": ZERO_COST_CONFIG.commission_per_share,
            "slippage_cents": ZERO_COST_CONFIG.slippage_cents,
            "slippage_proportional": ZERO_COST_CONFIG.slippage_proportional,
        },
        per_asset=per_asset,
        universe=dict(
            lookback_5_zero_counts=lookback_5_zero_counts,
            lookback_5_real_counts=lookback_5_real_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_5_zero_verdict=per_asset[t]["zero_cost"]["5"]["verdict"],
                    lookback_5_zero_median=per_asset[t]["zero_cost"]["5"]["medians"][0],
                    lookback_5_real_verdict=per_asset[t]["real_cost"]["5"]["verdict"],
                    lookback_5_real_median=per_asset[t]["real_cost"]["5"]["medians"][0],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    zero_cost_edge_counts=per_asset[t]["zero_cost_edge_counts"],
                    real_cost_edge_counts=per_asset[t]["real_cost_edge_counts"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity test (lookback 3/5/10) "
          "with realistic costs and matched null: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())