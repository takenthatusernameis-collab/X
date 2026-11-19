"""Cost-sensitivity check for short-horizon momentum on the collected universe.

Objective: classify whether the candidate positive short-horizon momentum edge
(`research/checks/momentum.py`, `momentum_lookback_sweep.py`) survives the
repository's existing realistic transaction-cost model and a conservative cost
stress, using a matched coin-flip null at every cost level.

A-priori prediction (declared before execution, not tuned to results): the
daily-rebalanced short-horizon momentum edge will NOT remain a positive
REGIME_STABLE edge at base realistic costs for a majority of the 10-asset
universe. The prediction is COST_ROBUST <= 3/10 at lookback 5 (and <= 3/10 at
lookbacks 3 and 10). A result with >= 6/10 COST_ROBUST at any lookback
falsifies this prediction.

Cost model (the repository's existing "realistic costs" configuration from
`examples/ma_crossover.py` and `examples/ma_crossover_real_data.py`):
commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
slippage_proportional=0.0005. The predeclared grid scales every component by
{0.0, 0.5, 1.0, 2.0}; 2.0x is the conservative stress.

Classification rule (predeclared, per asset and lookback):
- edge(m): the framework's stress_segments verdict at cost multiplier m is
  REGIME_STABLE and every segment median is > 0 (the same positive-edge
  definition used by the prior lookback-sweep check).
- COST_ROBUST: edge(1.0) and edge(2.0).
- COST_RESILIENT_BASE_ONLY: edge(1.0) and not edge(2.0).
- COST_FRAGILE: edge(0.0) and not edge(1.0).
- NO_PRE_COST_EDGE: not edge(0.0).

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return, hold 1 day, daily rebalance, lookbacks 3/5/10, the same
train=252d/test=84d/warmup=60d/overlap=60d walk-forward and the same 4
contiguous volatility blocks as the prior momentum checks, with the matched
coin-flip null from the framework (`noise_benchmark`) evaluated under the same
cost configuration and the same segments. The 0.0x column must reproduce
`momentum_lookback_sweep_results.json` exactly; that cross-consistency check is
an evidence gate.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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
COST_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)

BASE_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

OUTLIER_TOL = bt.OUTLIER_TOL

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
REFERENCE_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

PREDICTION = (
    "COST_ROBUST <= 3/10 at lookback 5 and <= 3/10 at lookbacks 3 and 10; "
    ">= 6/10 COST_ROBUST at any lookback falsifies the prediction."
)
CLASSIFICATION_RULE = (
    "edge(m): stress_segments verdict == REGIME_STABLE and all segment medians "
    "> 0. COST_ROBUST: edge(1.0) and edge(2.0). "
    "COST_RESILIENT_BASE_ONLY: edge(1.0) and not edge(2.0). "
    "COST_FRAGILE: edge(0.0) and not edge(1.0). "
    "NO_PRE_COST_EDGE: not edge(0.0)."
)


def cost_cfg(multiplier):
    """BacktestConfig for a multiplier of the existing realistic cost model."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=BASE_COSTS["commission_per_trade"] * float(multiplier),
        commission_per_share=BASE_COSTS["commission_per_share"] * float(multiplier),
        slippage_cents=BASE_COSTS["slippage_cents"] * float(multiplier),
        slippage_proportional=BASE_COSTS["slippage_proportional"] * float(multiplier),
    )


def segment_result(ticker, bars, segment_fn, lookback, multiplier):
    """Walk-forward regime-stability result for one ticker/lookback/cost level."""
    closes = bars.closes_array()
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=segment_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cost_cfg(multiplier),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    medians = [round(float(s.baseline_median_log_return), 3) for s in res.scenarios]
    null_medians = [round(float(s.noise_median_log_return), 3) for s in res.scenarios]
    median_candidate = round(float(np.median(medians)), 3) if medians else 0.0
    median_null = round(float(np.median(null_medians)), 3) if null_medians else 0.0
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=medians,
        null_medians=null_medians,
        candidate_dispersion=round(float(res.candidate_dispersion), 3),
        null_dispersion=round(float(res.null_dispersion), 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
        median_candidate=median_candidate,
        median_null=median_null,
        edge_vs_null=round(median_candidate - median_null, 3),
        n_positive_net=sum(1 for m in medians if m > 0),
        n_segments_edge_vs_null=sum(
            1 for m, n in zip(medians, null_medians) if (m - n) > OUTLIER_TOL
        ),
    )


def edge_flag(rec):
    """The predeclared positive-edge predicate at one cost level."""
    return rec["verdict"] == "REGIME_STABLE" and all(m > 0 for m in rec["medians"])


def classify(by_cost):
    """Apply the predeclared classification rule to one asset/lookback."""
    e0 = edge_flag(by_cost["0.0"])
    e1 = edge_flag(by_cost["1.0"])
    e2 = edge_flag(by_cost["2.0"])
    if not e0:
        return "NO_PRE_COST_EDGE"
    if e1 and e2:
        return "COST_ROBUST"
    if e1:
        return "COST_RESILIENT_BASE_ONLY"
    return "COST_FRAGILE"


def break_even(by_cost):
    """Largest cost multiplier at which the positive-edge predicate holds."""
    if edge_flag(by_cost["2.0"]):
        return "GE_2.0"
    for multiplier in reversed(COST_MULTIPLIERS[:-1]):
        if edge_flag(by_cost[str(multiplier)]):
            return str(multiplier)
    return "NONE"


def compare_zero_cost_to_reference(per_asset, reference):
    """Cross-consistency gate: the 0.0x column must reproduce the prior sweep."""
    mismatches = []
    ref_universe = reference["per_asset"]
    for ticker in ref_universe:
        for lookback in LOOKBACKS:
            published = ref_universe[ticker][str(lookback)]
            current = per_asset[ticker][f"lookback_{lookback}"]["by_cost"]["0.0"]
            fields = ("segments", "medians", "null_medians", "verdict")
            for field in fields:
                if current[field] != published[field]:
                    mismatches.append(
                        f"{ticker} lookback {lookback} {field}: "
                        f"current={current[field]} published={published[field]}"
                    )
    return mismatches


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

    print("\n=== 3. Leakage review (lookback=5, base realistic costs) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_cfg(1.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost grid (0.0x / 0.5x / 1.0x / 2.0x) x lookback (3 / 5 / 10) ===")
    print("  base cost model: commission_per_trade={}, commission_per_share={}, "
          "slippage_cents={}, slippage_proportional={}".format(
              BASE_COSTS["commission_per_trade"],
              BASE_COSTS["commission_per_share"],
              BASE_COSTS["slippage_cents"],
              BASE_COSTS["slippage_proportional"]))
    print("  params: train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, {} contiguous volatility blocks, "
          "matched coin-flip null per cost level\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, N_BLOCKS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        per_asset[ticker] = {}
        for lookback in LOOKBACKS:
            by_cost = {}
            for multiplier in COST_MULTIPLIERS:
                by_cost[str(multiplier)] = segment_result(
                    ticker, bars, seg_fn, lookback, multiplier
                )
            classification = classify(by_cost)
            per_asset[ticker][f"lookback_{lookback}"] = dict(
                by_cost=by_cost,
                classification=classification,
                break_even_multiplier=break_even(by_cost),
            )
            base = by_cost["1.0"]
            print(
                "  {:6s} lb={:<2d} {:24s} break_even={:<6s} "
                "cand(1x)={:+.3f} null(1x)={:+.3f} edge={:+.3f} "
                "pos_net={}/{} verdict(1x)={}".format(
                    ticker, lookback, classification,
                    per_asset[ticker][f"lookback_{lookback}"]["break_even_multiplier"],
                    base["median_candidate"], base["median_null"], base["edge_vs_null"],
                    base["n_positive_net"], len(base["medians"]), base["verdict"],
                )
            )
    print("")

    print("=== 5. Cross-consistency gate: 0.0x vs momentum_lookback_sweep_results.json ===")
    with open(REFERENCE_PATH) as fh:
        reference = json.load(fh)
    mismatches = compare_zero_cost_to_reference(per_asset, reference)
    if mismatches:
        print("  CROSS-CONSISTENCY MISMATCH:")
        for item in mismatches:
            print("    " + item)
        print("  Aborting before artifact write; investigate the 0.0x path.")
        return 1
    print("  all 10 tickers x 3 lookbacks: segments, medians, null_medians, "
          "verdict MATCH")

    print("\n=== 6. Determinism (AMZN lookback 5 at 1.0x and 2.0x) ===")
    amzn_bars = all_tickers["AMZN"]
    amzn_labels = bt.volatility_blocks(
        amzn_bars.closes_array(), n_blocks=N_BLOCKS, window=WINDOW
    )
    amzn_seg_fn = bt.segment_fn_from_labels(amzn_labels)
    for multiplier in (1.0, 2.0):
        rerun = segment_result("AMZN", amzn_bars, amzn_seg_fn, 5, multiplier)
        published = per_asset["AMZN"]["lookback_5"]["by_cost"][str(multiplier)]
        status = "identical" if rerun == published else "DIFFERENT"
        print(f"  AMZN lb=5 cost={multiplier}: {status}")
        assert rerun == published, "non-deterministic cost-sensitivity output"

    counts_by_lookback = {}
    for lookback in LOOKBACKS:
        counts = Counter(
            per_asset[t][f"lookback_{lookback}"]["classification"]
            for t in ticker_order
        )
        counts_by_lookback[str(lookback)] = dict(counts)
    overall = Counter(
        per_asset[t][f"lookback_{lb}"]["classification"]
        for t in ticker_order
        for lb in LOOKBACKS
    )

    print("\n=== 7. Cost-robustness classification ===")
    for lookback in LOOKBACKS:
        counts = counts_by_lookback[str(lookback)]
        robust = [t for t in ticker_order
                  if per_asset[t][f"lookback_{lookback}"]["classification"] == "COST_ROBUST"]
        print(f"  lookback {lookback}: {counts} | COST_ROBUST={robust}")
    print(f"  overall (30 cells): {dict(overall)}")
    base_medians = [per_asset[t]["lookback_5"]["by_cost"]["1.0"]["median_candidate"]
                    for t in ticker_order]
    base_nulls = [per_asset[t]["lookback_5"]["by_cost"]["1.0"]["median_null"]
                  for t in ticker_order]
    print(f"  lookback 5 at 1.0x: median candidate {np.median(base_medians):+.3f}, "
          f"median null {np.median(base_nulls):+.3f}")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_multipliers=list(COST_MULTIPLIERS),
        base_cost=BASE_COSTS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        outlier_tol=OUTLIER_TOL,
        a_priori_prediction=PREDICTION,
        classification_rule=CLASSIFICATION_RULE,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            classification_counts_by_lookback=counts_by_lookback,
            classification_counts_overall=dict(overall),
        ),
    )
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())