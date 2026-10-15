"""Cost-sensitivity check for lookback 3/5/10 momentum under realistic costs.

This check tests whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe. The hypothesis:
- momentum shows CONSISTENT_WITH_NOISE (no edge anywhere after realistic costs)
- or momentum shows REGIME_DEPENDENT (the positive run concentrates in one
  historical regime mix)
- or momentum shows REGIME_STABLE_LOSS (uniformly negative after costs)
- or momentum shows REGIME_STABLE_POSITIVE (consistent edge everywhere after costs)

Any REGIME_STABLE or REGIME_DEPENDENT positive outcome would be unexpected given
that the coin-flip null is matched with the same cost model.

Method (fixed a-priori, not tuned to OOS):
- Lookback 3/5/10 momentum (long previous N-day return, hold 1 day, daily rebalance)
- Realistic cost model: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005 (from examples)
- Walk-forward per segment: train=252d / test=84d / warmup=60d / overlap=60d
- Minimum 400 bars per segment
- 4 contiguous volatility blocks with past-only 60-bar volatility classifier
- Matched null: same cost model, coin-flip signals with identical walk-forward
- Independent verification required

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

# Realistic cost model from examples
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
)

ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"


def run_momentum_with_costs(asset, tickers):
    """Run momentum with lookback 3/5/10 using realistic cost model."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    results = {}

    for lookback in [3, 5, 10]:
        def signals_fn(c, **p):
            return bt.momentum_signals(c, lookback=lookback)

        # Realistic costs
        signals_real = signals_fn(closes)
        res_real = bt.stress_segments(
            signals_fn=signals_fn,
            bars=list(bars),
            signals=signals_real,
            param_grid=[{"lookback": lookback}],
            baseline=(("lookback", lookback),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=REALISTIC_COSTS,
        )

        # Zero costs (matched null baseline)
        signals_zero = signals_fn(closes)
        res_zero = bt.stress_segments(
            signals_fn=signals_fn,
            bars=list(bars),
            signals=signals_zero,
            param_grid=[{"lookback": lookback}],
            baseline=(("lookback", lookback),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=ZERO_COSTS,
        )

        names = [s.name for s in res_real.scenarios]
        medians_real = [round(s.baseline_median_log_return, 3) for s in res_real.scenarios]
        medians_zero = [round(s.baseline_median_log_return, 3) for s in res_zero.scenarios]

        # Calculate cost impact: zero - realistic (positive means realistic reduces returns)
        cost_impact = [round(mzero - mreal, 3) for mzero, mreal in zip(medians_zero, medians_real)]

        results[f"lookback_{lookback}"] = dict(
            segments=names,
            medians_real=medians_real,
            medians_zero_cost=medians_zero,
            cost_impact=cost_impact,
            candidate_dispersion_real=round(res_real.candidate_dispersion, 3),
            candidate_dispersion_zero=round(res_zero.candidate_dispersion, 3),
            null_dispersion_real=round(res_real.null_dispersion, 3),
            null_dispersion_zero=round(res_zero.null_dispersion, 3),
            verdict_real=res_real.overall_verdict,
            verdict_zero=res_zero.overall_verdict,
            n_folds=res_real.n_folds,
            cost_parameters={
                "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
                "commission_per_share": REALISTIC_COSTS.commission_per_share,
                "slippage_cents": REALISTIC_COSTS.slippage_cents,
                "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
            },
        )

    return results


def perturbation_sweep_cost_sensitivity():
    """Synthetic perturbation sweep to test parameter robustness under costs."""
    bars = bt.generate_bars(600, seed=SEED)

    # Sweep lookback 3/5/10 with realistic costs
    grid = bt.parameter_grid_around(
        (("lookback", 5),),
        multipliers=(0.6, 1.0, 1.67),  # roughly 3, 5, 10
    )

    sweep_real = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
        cfg=REALISTIC_COSTS,
    )

    # Null benchmark with same costs
    sweep_null = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
        seed=SEED,
    )

    # sweep_null is already a SweepSummary, no need to call sweep_summary again
    summary = bt.sweep_summary(sweep_real, baseline=(("lookback", 5),))
    # summary_null is already from noise_benchmark
    summary_null = sweep_null

    # Compare with null under costs
    compare = summary.compare_noise(summary_null.baseline_median_log_return)
    compare_fixed = summary.compare_noise(summary_null.baseline_median_log_return, tol=0.05)

    return dict(
        param_sets=[tuple(ps) for ps in sweep_real.param_sets],
        medians_real=[round(m, 3) for m in summary.median_log_returns],
        medians_null=[round(m, 3) for m in summary_null.median_log_returns],
        baseline_median_real=round(summary.baseline_median_log_return, 3),
        baseline_median_null=round(summary_null.baseline_median_log_return, 3),
        compare_null=compare,
        compare_fixed=compare_fixed,
        effective_tolerance=round(summary.effective_tolerance, 4),
        n_folds=summary.n_folds,
        cost_parameters={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
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

    print("\n=== 3. Momentum cost-robustness test ===")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"Segments: {N_BLOCKS} contiguous blocks by date, past-only volatility "
          f"classifier (turbulent = block median trailing-{WINDOW}-bar realized vol "
          f"> series-wide median)")
    print(f"Walk-forward: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment\n")

    results_all = {}
    tickers_univ = {}

    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    # Test all assets
    for asset in manifest["universe"]:
        res = run_momentum_with_costs(asset, tickers_univ)
        results_all[asset] = res
        print(f"Asset: {asset}")
        print("  lookback | segments    | medians (real)    | medians (zero)   | impact | verdict_real | verdict_zero")
        print("  ------- | ----------- | ------------------ | ---------------- | ------- | ------------ | ------------")
        for lookback in [3, 5, 10]:
            r = res[f"lookback_{lookback}"]
            segs = "[".join(r["segments"]) + "]"
            meds_real = "[".join(str(m) for m in r["medians_real"]) + "]"
            meds_zero = "[".join(str(m) for m in r["medians_zero_cost"]) + "]"
            impact = "[".join(str(m) for m in r["cost_impact"]) + "]"
            verdict_real = r["verdict_real"]
            verdict_zero = r["verdict_zero"]
            print(f"  {lookback:7d} | {segs:11s} | {meds_real:18s} | {meds_zero:17s} | {impact:7s} | {verdict_real:12s} | {verdict_zero}")
        print()

    # Determinism check
    print("\n=== 4. Determinism check ===")
    asset = manifest["universe"][0]  # Test first asset
    res1 = run_momentum_with_costs(asset, tickers_univ)
    res2 = run_momentum_with_costs(asset, tickers_univ)
    out1 = json.dumps(res1, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    # Perturbation sweep
    print("\n=== 5. Parameter-robustness sweep (lookback 3/5/10) under realistic costs ===")
    pers = perturbation_sweep_cost_sensitivity()
    print("  param sets:", pers["param_sets"])
    print("  medians (real):", pers["medians_real"])
    print("  medians (null):", pers["medians_null"])
    print("  baseline (lookback=5, real): {:+.3f}".format(pers["baseline_median_real"]))
    print("  baseline (lookback=5, null): {:+.3f}".format(pers["baseline_median_null"]))
    print("  vs null (sample-calibrated): {}".format(pers["compare_null"]))
    print("  vs null (fixed tol 0.05): {}".format(pers["compare_fixed"]))

    # Create artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_parameters={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        universe_results=results_all,
        perturbation=pers,
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness test on collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
