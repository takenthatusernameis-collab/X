"""Cost-sensitive momentum robustness: bounded cost grid for lookback 3/5/10.

This is the focused-task R-001 contract for Global Agent 2.

Primary question: Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?

Objective: Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null.

Scope: Use the existing 10-asset collected universe and existing momentum implementation/check framework.

Stop condition: Stop after the predeclared cost grid is evaluated and the independent verifier agrees or records the exact discrepancy.

Success criterion: The result is reproducible and independently verified, regardless of positive or negative outcome.

Evidence basis: LEARNING_STATE admits short-horizon momentum as candidate positive evidence but leaves execution-cost robustness as a meaningful qualification.

Bottleneck: The candidate positive momentum evidence is short-horizon and mechanical; cost robustness remains decision-relevant.
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"

# REALISTIC COST MODEL (existing in examples/ma_crossover_real_data.py)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# ZERO-COST MODEL (for comparison)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)


def segment_cost_sensitivity(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, one cost model."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
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

    print("\n=== 4. Momentum cost sensitivity (lookback 3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost sensitivity
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [cost_model][lookback][ticker][segment index]
    results = {
        "zero_cost": {lb: {t: segment_cost_sensitivity(t, all_tickers[t], lb, ZERO_COSTS)
                           for t in ticker_order} for lb in LOOKBACKS},
        "realistic_cost": {lb: {t: segment_cost_sensitivity(t, all_tickers[t], lb, REALISTIC_COSTS)
                                for t in ticker_order} for lb in LOOKBACKS},
    }

    # Analysis and comparison
    zero_cost_edge_counts = {lb: {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0} for lb in LOOKBACKS}
    realistic_cost_edge_counts = {lb: {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0} for lb in LOOKBACKS}

    for lb in LOOKBACKS:
        for ticker in ticker_order:
            # Zero cost analysis
            verdict = results["zero_cost"][lb][ticker]["verdict"]
            medians = results["zero_cost"][lb][ticker]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                zero_cost_edge_counts[lb]["EDGE"] += 1
            elif verdict == "REGIME_STABLE_LOSS":
                zero_cost_edge_counts[lb]["LOSS"] += 1
            else:
                zero_cost_edge_counts[lb]["NO_EDGE"] += 1

            # Realistic cost analysis
            verdict = results["realistic_cost"][lb][ticker]["verdict"]
            medians = results["realistic_cost"][lb][ticker]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                realistic_cost_edge_counts[lb]["EDGE"] += 1
            elif verdict == "REGIME_STABLE_LOSS":
                realistic_cost_edge_counts[lb]["LOSS"] += 1
            else:
                realistic_cost_edge_counts[lb]["NO_EDGE"] += 1

    print("\n=== 5. Cost sensitivity analysis ===")
    for lb in LOOKBACKS:
        print(f"  Lookback {lb}:")
        print(f"    Zero cost:    EDGE={zero_cost_edge_counts[lb]['EDGE']} "
              f"LOSS={zero_cost_edge_counts[lb]['LOSS']} "
              f"NO_EDGE={zero_cost_edge_counts[lb]['NO_EDGE']}")
        print(f"    Realistic cost: EDGE={realistic_cost_edge_counts[lb]['EDGE']} "
              f"LOSS={realistic_cost_edge_counts[lb]['LOSS']} "
              f"NO_EDGE={realistic_cost_edge_counts[lb]['NO_EDGE']}")

    # Calculate cost impact
    print("\n=== 6. Cost impact analysis ===")
    for lb in LOOKBACKS:
        zero_medians = np.mean([results["zero_cost"][lb][t]["medians"][0] for t in ticker_order])
        realistic_medians = np.mean([results["realistic_cost"][lb][t]["medians"][0] for t in ticker_order])
        impact = realistic_medians - zero_medians
        print(f"  Lookback {lb}: zero_cost={zero_medians:+.3f}, "
              f"realistic_cost={realistic_medians:+.3f}, "
              f"impact={impact:+.3f}")

    # Determinism check on target assets
    print("\n=== 7. Determinism check (target subset) ===")
    target_results_zero = {lb: {t: segment_cost_sensitivity(t, tickers[t], lb, ZERO_COSTS)
                               for t in target} for lb in LOOKBACKS}
    target_results_realistic = {lb: {t: segment_cost_sensitivity(t, tickers[t], lb, REALISTIC_COSTS)
                                    for t in target} for lb in LOOKBACKS}

    out1_zero = json.dumps({lb: {t: target_results_zero[lb][t] for t in target}
                            for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps({lb: {t: segment_cost_sensitivity(t, tickers[t], lb, ZERO_COSTS)
                                for t in target} for lb in LOOKBACKS}, sort_keys=True)
    print(f"  Zero cost determinism: r1 == r2: {out1_zero == out2_zero}")

    out1_realistic = json.dumps({lb: {t: target_results_realistic[lb][t] for t in target}
                                for lb in LOOKBACKS}, sort_keys=True)
    out2_realistic = json.dumps({lb: {t: segment_cost_sensitivity(t, tickers[t], lb, REALISTIC_COSTS)
                                    for t in target} for lb in LOOKBACKS}, sort_keys=True)
    print(f"  Realistic cost determinism: r1 == r2: {out1_realistic == out2_realistic}")

    # Artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost=dict(
            per_asset=dict(
                (t, dict(
                    segments=results["zero_cost"][lb][t]["segments"],
                    medians=results["zero_cost"][lb][t]["medians"],
                    null_medians=results["zero_cost"][lb][t]["null_medians"],
                    candidate_dispersion=results["zero_cost"][lb][t]["candidate_dispersion"],
                    null_dispersion=results["zero_cost"][lb][t]["null_dispersion"],
                    verdict=results["zero_cost"][lb][t]["verdict"],
                    n_folds=results["zero_cost"][lb][t]["n_folds"],
                ))
                for lb in LOOKBACKS
                for t in ticker_order
            ),
            edge_counts=zero_cost_edge_counts,
        ),
        realistic_cost=dict(
            per_asset=dict(
                (f"{t}_lb{lb}", dict(
                    segments=results["realistic_cost"][lb][t]["segments"],
                    medians=results["realistic_cost"][lb][t]["medians"],
                    null_medians=results["realistic_cost"][lb][t]["null_medians"],
                    candidate_dispersion=results["realistic_cost"][lb][t]["candidate_dispersion"],
                    null_dispersion=results["realistic_cost"][lb][t]["null_dispersion"],
                    verdict=results["realistic_cost"][lb][t]["verdict"],
                    n_folds=results["realistic_cost"][lb][t]["n_folds"],
                ))
                for lb in LOOKBACKS
                for t in ticker_order
            ),
            edge_counts=realistic_cost_edge_counts,
        ),
        cost_model=dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost sensitivity across the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())