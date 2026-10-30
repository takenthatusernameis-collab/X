"""Momentum cost-sensitivity check (lookback 3 / 5 / 10) across the collected universe.

Objective: evaluate whether the momentum edge observed at lookback=5
under zero cost survives the realistic cost model used in the repository's
examples (e.g., ma_crossover.py, volatility_regime_filter.py).

Cost configuration:
- zero cost: BacktestConfig with all cost fields at their defaults (0.0).
- realistic cost: BacktestConfig with realistic commissions and slippage:
  commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005.

Method (fixed a-priori, not tuned to OOS):
- long the previous N-day return; hold 1 day; daily rebalance; lookbacks 3, 5, 10;
- same walk-forward windows, warmup, overlap, regime classification as momentum_lookback_sweep.py.
- For each lookback, compute results with zero cost, then with realistic cost,
  preserving the artifact so an independent verifier can compare the two paths.
- All inputs (dataset, seeds, windows, segments) are identical; only the cost
  configuration differs. This isolates the effect of transaction costs.

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

# Zero cost configuration (default)
CFG_ZERO = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)

# Realistic cost configuration (from examples)
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
        cost_config="zero" if cfg == CFG_ZERO else "realistic",
    )


def lookback_verdict(sr):
    """Robustness verdict for one lookback's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset."""
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
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum cost-sensitivity sweep (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep for both cost configurations.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [cost][lookback][ticker][segment index].
    results_zero = {lb: {t: segment_result(t, all_tickers[t], lb, CFG_ZERO) for t in ticker_order}
                    for lb in LOOKBACKS}
    results_real = {lb: {t: segment_result(t, all_tickers[t], lb, CFG_REAL) for t in ticker_order}
                    for lb in LOOKBACKS}

    # Compute cost impact: difference in medians between realistic and zero cost.
    cost_impact = {}
    for ticker in ticker_order:
        per_lb = {}
        for lb in LOOKBACKS:
            zero_medians = results_zero[lb][ticker]["medians"]
            real_medians = results_real[lb][ticker]["medians"]
            impact = [round(rm - zm, 3) for zm, rm in zip(zero_medians, real_medians)]
            per_lb[str(lb)] = {
                "zero_cost_medians": zero_medians,
                "real_cost_medians": real_medians,
                "cost_impact": impact,
            }
        cost_impact[ticker] = per_lb

    # Summarize robustness counts for each cost configuration.
    robust_counts_zero = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    robust_counts_real = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}

    print("\n  Cost-sensitivity summary:")
    print("  " + "-" * 100)
    print("  asset | lookback | verdict_zero | verdict_real | impact | verdict_change")
    print("  " + "-" * 100)

    for ticker in ticker_order:
        # Build lookback results for verdict determination.
        lookback_results_zero = {str(lb): results_zero[lb][ticker] for lb in LOOKBACKS}
        lookback_results_real = {str(lb): results_real[lb][ticker] for lb in LOOKBACKS}

        # Determine verdicts.
        v_zero, c_zero = robustness_verdict(lookback_results_zero)
        v_real, c_real = robustness_verdict(lookback_results_real)
        robust_counts_zero[v_zero] += 1
        robust_counts_real[v_real] += 1

        # Show impact per lookback.
        for lb in LOOKBACKS:
            verdict_zero = "EDGE" if c_zero["EDGE"] >= 2 else "SENSITIVE" if c_zero["EDGE"] == 1 else "NO_EDGE"
            verdict_real = "EDGE" if c_real["EDGE"] >= 2 else "SENSITIVE" if c_real["EDGE"] == 1 else "NO_EDGE"
            impact = cost_impact[ticker][str(lb)]["cost_impact"][0]
            change = "YES" if verdict_zero != verdict_real else "NO"
            print(f"  {ticker:6s} | {lb:8d} | {verdict_zero:12s} | {verdict_real:12s} | "
                  f"{impact:6.3f} | {change:4s}")

    print("\n  Robustness verdict counts:")
    print(f"  zero cost : ROBUST={robust_counts_zero['ROBUST']} "
          f"SENSITIVE={robust_counts_zero['SENSITIVE']} NO_EDGE={robust_counts_zero['NO_EDGE']}")
    print(f"  realistic cost : ROBUST={robust_counts_real['ROBUST']} "
          f"SENSITIVE={robust_counts_real['SENSITIVE']} NO_EDGE={robust_counts_real['NO_EDGE']}")

    # Determinism: rerun the target subset and assert identical output for each cost configuration.
    print("\n  Determinism checks:")
    for cost_name, results in [("zero", results_zero), ("realistic", results_real)]:
        res2 = {lb: {t: segment_result(t, tickers[t], lb, CFG_ZERO if cost_name == "zero" else CFG_REAL) for t in target}
                for lb in LOOKBACKS}
        out1 = json.dumps({lb: {t: results[lb][t] for t in target} for lb in LOOKBACKS}, sort_keys=True)
        out2 = json.dumps(res2, sort_keys=True)
        match = out1 == out2
        print(f"    {cost_name} cost: {'PASS' if match else 'FAIL'} (r1 == r2)")
        assert match, f"non-deterministic output with {cost_name} cost"

    # Write the artifact with both cost configurations.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost=dict(
            per_asset={t: results_zero[lb][t] for lb in LOOKBACKS for t in ticker_order},
            robustness_counts=robust_counts_zero,
        ),
        realistic_cost=dict(
            per_asset={t: results_real[lb][t] for lb in LOOKBACKS for t in ticker_order},
            robustness_counts=robust_counts_real,
        ),
        cost_impact=cost_impact,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
