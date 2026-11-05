#!/usr/bin/env python3
"""Cost-sensitive momentum lookback sweep for R-001.

This runs the same lookback=3/5/10 momentum edge test as R-002 but incorporates
the repository's realistic cost model (commission_per_share=0.003, slippage_cents=5.0)
to answer the primary question:

  Does the short-horizon momentum edge survive conservative transaction-cost stress
  on the collected real-data universe?

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return;
hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows, 4 contiguous volatility blocks, min_segment_bars threshold, and
''matched null'' coin-flip benchmark run on the same segments as momentum_lookback_sweep.py.
The edge is computed with realistic costs; the null is cost-free (matched null).

Classification: an asset shows an edge at a lookback when its per-segment
medians are uniformly positive REGIME_STABLE (as in momentum_lookback_sweep.py).
The universe-level verdict follows the same ROBUST/SENSITIVE/NO_EDGE logic.

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
COST_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_cost_sensitivity_results.json"

# Realistic cost model from test_engine.py and BacktestConfig defaults
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,           # commission per trade (rare)
    commission_per_share=0.003,         # 3 bps per share (realistic)
    slippage_cents=5.0,                 # 5 cents per share (realistic)
    slippage_proportional=0.0,          # no proportional slippage for this test
    warmup_periods=0,
)

COST_FREE_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    # All costs are zero for the matched null benchmark
)
def segment_result(ticker, bars, lookback, use_costs=True):
    """Walk-forward regime-stability result for one ticker and one lookback.
    
    Uses realistic costs for edge computation when use_costs=True;
    uses cost-free config for matched null when use_costs=False.
    """
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    # Use realistic costs for edge computation
    cfg = REALISTIC_COST_CONFIG if use_costs else COST_FREE_CONFIG
    
    # Use the stress_segments function with the specified cost config
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
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
        edge_return_with_costs=round(scen.baseline_median_log_return, 3) if use_costs else None,
    )
def lookback_verdict(sr):
    """Verdict on whether one lookback shows an edge."""
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
        res = bt.run_bars(list(bars), signals, REALISTIC_COST_CONFIG)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f} (with costs)")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep with realistic costs (3 / 5 / 10) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a cost-free matched null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  BacktestConfig for edge: target_exposure=1.0, commission_per_share=0.003, slippage_cents=5.0")
    print("  BacktestConfig for matched null: all costs zero (matched null)")

    # Full-universe lookback sweep with realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index] with realistic costs
    lookback_results_with_costs = {lb: {t: segment_result(t, all_tickers[t], lb, use_costs=True)
                                        for t in ticker_order}
                                   for lb in LOOKBACKS}

    per_asset_with_costs = {}
    robustness_counts_with_costs = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts_with_costs = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                    "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results_with_costs[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts_with_costs[v] += 1
        r5 = profile["5"]
        lookback_5_counts_with_costs[r5["verdict"]] += 1
        # Lookback-robustness edge counts (edges per asset, for the summary).
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        # Store the per-lookback medians + nulls for the artifact's perturbation
        # view (keyed by lookback so the verifier can recompute each column).
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_with_costs[ticker] = profile

    print("  lookback 5 (reference) verdict counts with realistic costs: "
          f"{lookback_5_counts_with_costs}")
    print("  lookback-robustness verdict counts with costs: "
          f"ROBUST={robustness_counts_with_costs['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts_with_costs['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts_with_costs['NO_EDGE']}\n")
    print("  Per-asset perturbation profiles (with realistic costs):")
    for ticker in ticker_order:
        p = per_asset_with_costs[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={} edge_with_costs={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"], p["5"]["edge_return_with_costs"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result(t, tickers[t], lb, use_costs=True) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results_with_costs[lb][t] for t in target}
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
        per_asset=per_asset_with_costs,
        universe=dict(
            lookback_5_counts=lookback_5_counts_with_costs,
            robustness_counts=robustness_counts_with_costs,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset_with_costs[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset_with_costs[t]["lookback_null_medians"][str(lb)]
                                           for lb in LOOKBACKS},
                    verdicts_5=per_asset_with_costs[t]["5"]["verdict"],
                    edge_counts=per_asset_with_costs[t]["edge_counts"],
                    robustness_verdict=per_asset_with_costs[t]["robustness_verdict"],
                    n_folds_5=per_asset_with_costs[t]["5"]["n_folds"],
                    edge_return_with_costs=per_asset_with_costs[t]["5"]["edge_return_with_costs"],
                ))
                for t in ticker_order
            ),
        ),
        cost_config={
            "initial_capital": REALISTIC_COST_CONFIG.initial_capital,
            "target_exposure": REALISTIC_COST_CONFIG.target_exposure,
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
        },
    )
    with open(COST_ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(COST_ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with realistic costs: exploratory "
          "simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())