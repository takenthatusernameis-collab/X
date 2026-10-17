"""Real-data momentum cost-sensitivity check for lookback 3/5/10.

Objective: test whether the momentum edge observed at lookback=5
(`research/checks/momentum.py`, activation 37318950814) survives conservative
transaction-cost stress on the collected real-data universe (10 assets).

Method (fixed a-priori, not tuned to OOS):
- Long the previous N-day return; hold 1 day; daily rebalance; lookbacks 3, 5, 10.
- Same walk-forward windows and volatility blocks as `momentum.py`.
- Use realistic cost model: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005.
- Matched null: coin-flip null benchmark run on the same segments.
- Determinism via internal independent recomputation of a 2-asset subset.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
REALISTIC_COSTS_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COSTS_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)


def segment_result(ticker, bars, lookback, use_realistic_costs=True):
    """Walk-forward regime-stability result for one ticker, lookback, and cost model."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    cfg = REALISTIC_COSTS_CFG if use_realistic_costs else ZERO_COSTS_CFG

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
        cost_model="realistic" if use_realistic_costs else "zero",
    )


def lookback_verdict(sr):
    """Verdict for one lookback's scenario result."""
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


def cost_robustness_verdict(lookback_results_zero, lookback_results_real):
    """Verdict on whether momentum edge survives cost stress.

    EDGE_SURVIVES : positive edge at >= 2 lookbacks in both zero-cost and realistic-cost models.
    EDGE_VANISHES : positive edge in zero-cost but not in realistic-cost.
    NO_EDGE : no positive edge in zero-cost model.
    """
    zero_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    real_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        zero_counts[lookback_verdict(lookback_results_zero[str(lb)])] += 1
        real_counts[lookback_verdict(lookback_results_real[str(lb)])] += 1

    if zero_counts["EDGE"] >= 2 and real_counts["EDGE"] >= 2:
        return "EDGE_SURVIVES", zero_counts, real_counts
    elif zero_counts["EDGE"] >= 2 and real_counts["EDGE"] < 2:
        return "EDGE_VANISHES", zero_counts, real_counts
    else:
        return "NO_EDGE", zero_counts, real_counts


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

    print("\n=== 4. Momentum cost-sensitivity check (lookback 3 / 5 / 10) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load full universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Cost-sensitivity results indexed [lookback][ticker][cost_model]
    results = {lb: {t: {"zero": None, "real": None} for t in ticker_order} for lb in LOOKBACKS}

    print("  Running cost-sensitivity check across collected universe ===")
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        for lb in LOOKBACKS:
            print(f"    {ticker} lookback {lb}: zero-cost", end="")
            results[lb][ticker]["zero"] = segment_result(ticker, bars, lb, use_realistic_costs=False)
            print(f" -> verdict {results[lb][ticker]['zero']['verdict']}", end="")

            print(f" {ticker} lookback {lb}: realistic-cost", end="")
            results[lb][ticker]["real"] = segment_result(ticker, bars, lb, use_realistic_costs=True)
            print(f" -> verdict {results[lb][ticker]['real']['verdict']}")

    # Cost robustness verdict per asset
    per_asset_cost_robustness = {}
    robustness_counts_zero = {"EDGE_SURVIVES": 0, "EDGE_VANISHES": 0, "NO_EDGE": 0}
    robustness_counts_real = {"EDGE_SURVIVES": 0, "EDGE_VANISHES": 0, "NO_EDGE": 0}

    print("\n  Cost robustness verdict counts:")
    for ticker in ticker_order:
        lookback_results_zero = {str(lb): results[lb][ticker]["zero"] for lb in LOOKBACKS}
        lookback_results_real = {str(lb): results[lb][ticker]["real"] for lb in LOOKBACKS}
        verdict, zero_counts, real_counts = cost_robustness_verdict(lookback_results_zero, lookback_results_real)
        robustness_counts_zero[verdict] += 1
        robustness_counts_real[verdict] += 1
        per_asset_cost_robustness[ticker] = {
            "zero_cost_verdict": verdict,
            "zero_counts": zero_counts,
            "real_counts": real_counts,
            "lookback_zero_medians": {str(lb): results[lb][ticker]["zero"]["medians"][0] for lb in LOOKBACKS},
            "lookback_real_medians": {str(lb): results[lb][ticker]["real"]["medians"][0] for lb in LOOKBACKS},
        }
        print(f"    {ticker}: {verdict}")

    print(f"\n  Summary: {robustness_counts_zero['EDGE_SURVIVES']} EDGE_SURVIVES | "
          f"{robustness_counts_zero['EDGE_VANISHES']} EDGE_VANISHES | "
          f"{robustness_counts_zero['NO_EDGE']} NO_EDGE (zero-cost)")
    print(f"           {robustness_counts_real['EDGE_SURVIVES']} EDGE_SURVIVES | "
          f"{robustness_counts_real['EDGE_VANISHES']} EDGE_VANISHES | "
          f"{robustness_counts_real['NO_EDGE']} NO_EDGE (realistic-cost)")

    # Lookback-robustness verdict counts (zero-cost baseline)
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        lookback_results = {str(lb): results[lb][ticker]["zero"] for lb in LOOKBACKS}
        v, _ = robustness_verdict(lookback_results)
        robustness_counts[v] += 1

    print(f"\n  Lookback-robustness verdict counts (zero-cost): "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")

    # Per-asset perturbation profiles
    print("  Per-asset cost robustness profiles:")
    for ticker in ticker_order[:10]:  # Show first 10 for brevity
        p = per_asset_cost_robustness[ticker]
        zero_med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["lookback_zero_medians"][str(lb)],
                                       "EDGE" if p["lookback_zero_medians"][str(lb)] > 0 else "LOSS" if p["lookback_zero_medians"][str(lb)] < 0 else "NO_EDGE")
            for lb in LOOKBACKS
        )
        real_med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["lookback_real_medians"][str(lb)],
                                       "EDGE" if p["lookback_real_medians"][str(lb)] > 0 else "LOSS" if p["lookback_real_medians"][str(lb)] < 0 else "NO_EDGE")
            for lb in LOOKBACKS
        )
        print(f"    {ticker:6s} zero-cost medians: {zero_med:40s} real-cost medians: {real_med:40s} verdict: {p['zero_cost_verdict']}")

    # Determinism: rerun the target subset and assert identical output
    print("\n=== Determinism check (2-asset subset) ===")
    res2 = {lb: {t: segment_result(t, tickers[t], lb, use_realistic_costs=False) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: results[lb][t]["zero"] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    # Also check determinism for realistic costs
    print("Realistic-cost determinism check (2-asset subset) ===")
    res2_real = {lb: {t: segment_result(t, tickers[t], lb, use_realistic_costs=True) for t in target}
                 for lb in LOOKBACKS}
    out1_real = json.dumps({lb: {t: results[lb][t]["real"] for t in target}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_real = json.dumps(res2_real, sort_keys=True)
    print(f"determinism: r1 == r2: {out1_real == out2_real}")
    assert out1_real == out2_real, "non-deterministic output with realistic costs"

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset_cost_robustness,
        universe=dict(
            lookback_5_counts={"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                              "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0},
            robustness_counts=robustness_counts,
            cost_robustness_zero=robustness_counts_zero,
            cost_robustness_real=robustness_counts_real,
            ticker_order=ticker_order,
        ),
        cost_model={
            "commission_per_trade": REALISTIC_COSTS_CFG.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS_CFG.commission_per_share,
            "slippage_cents": REALISTIC_COSTS_CFG.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS_CFG.slippage_proportional,
        },
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-sensitivity check across the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())