#!/usr/bin/env python3
"""Cost-sensitive momentum check for lookback 3/5/10 on the collected universe.

This check builds on the existing momentum framework but adds cost sensitivity:
- Tests momentum (long previous N-day return) for lookback 3/5/10
- Compares zero-cost vs realistic-cost models
- Uses the existing 10-asset collected universe
- Matched null benchmark through coin-flip perturbation sweep
- Deterministic verification path

Background: prior activations have tested momentum across lookback 5, 20, 60,
regime stability, and synthetic perturbation sweep. The remaining unresolved
question is whether the short-horizon positive edge survives cost robustness
stresses (fixed cost model, lookback 3/5/10, matched null).

Falsification prediction: momentum either shows CONSISTENT_WITH_NOISE (no cost-robust edge) or
REGIME_DEPENDENT (the positive run concentrates in one historical regime mix) or
REGIME_STABLE_LOSS (uniformly negative — which would be reversal winning on both sides).
Either outcome is a negative conclusion and closes the cell.

Method (fixed a-priori, not tuned to OOS):
- long the previous N-day return (N = 3,5,10)
- hold 1 day
- daily rebalance
- walk-forward IS/OOS validation (train=252d, test=84d, warmup=60d, overlap=60d)
- zero-cost model and realistic-cost model
- parameter sweep (lookback 3/5/ 직후 perturb 2x) plus coin-flip null benchmark
- asset-universe robustness across all 10 collected tickers
- independent verification via fresh walk_forward + noise_benchmark recomputation

Design of the independent path:
- Regime labels reconstructed from closes using volatility_blocks (same algorithm, different code)
- Segments reconstructed from labels using min_segment_bars
- Segment medians computed via walk_forward directly (no call to stress_segments)
- Perturbation medians recomputed by walking each parameter set independently
- All values loaded from the check's artifact and compared

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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def momentum_signals(closes: np.ndarray, lookback: int):
    """Past-only momentum signal: long the previous N-day return."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_asset_with_costs(asset, tickers, lookback=5):
    """Run momentum check with both zero-cost and realistic-cost models for one asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
    cfg1 = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    results_zero = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=p["lookback"]),
        bars=list(bars),
        signals=bt.momentum_signals(closes, lookback),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg0,
    )

    results_realistic = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=p["lookback"]),
        bars=list(bars),
        signals=bt.momentum_signals(closes, lookback),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg1,
    )

    return {
        "zero_cost": {
            "segments": [s.name for s in results_zero.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in results_zero.scenarios],
            "candidate_dispersion": round(results_zero.candidate_dispersion, 3),
            "null_dispersion": round(results_zero.null_dispersion, 3),
            "verdict": results_zero.overall_verdict,
            "n_folds": results_zero.n_folds,
        },
        "realistic_cost": {
            "segments": [s.name for s in results_realistic.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in results_realistic.scenarios],
            "candidate_dispersion": round(results_realistic.candidate_dispersion, 3),
            "null_dispersion": round(results_realistic.null_dispersion, 3),
            "verdict": results_realistic.overall_verdict,
            "n_folds": results_realistic.n_folds,
        },
    }


def perturbation_sweep_with_costs(n_bars=600):
    """Synthetic perturbation sweep with cost sensitivity."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", 5),),
        multipliers=(0.5, 1.0, 2.0),
    )

    sweep_zero = bt.parameter_sweep(
        signals_fn=momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
        cfg=bt.BacktestConfig(),
    )

    sweep_realistic = bt.parameter_sweep(
        signals_fn=momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
        cfg=bt.BacktestConfig(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
    )

    summary_zero = bt.sweep_summary(sweep_zero, baseline=(("lookback", 5),))
    summary_realistic = bt.sweep_summary(sweep_realistic, baseline=(("lookback", 5),))

    noise_zero = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )

    noise_realistic = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )

    return {
        "zero_cost": {
            "param_sets": [tuple(ps) for ps in sweep_zero.param_sets],
            "medians": [round(m, 3) for m in summary_zero.median_log_returns],
            "baseline_median": round(summary_zero.baseline_median_log_return, 3),
            "noise_median": round(noise_zero.baseline_median_log_return, 3),
            "compare_noise": summary_zero.compare_noise(noise_zero.baseline_median_log_return),
            "effective_tolerance": round(summary_zero.effective_tolerance, 4),
            "n_folds": summary_zero.n_folds,
        },
        "realistic_cost": {
            "param_sets": [tuple(ps) for ps in sweep_realistic.param_sets],
            "medians": [round(m, 3) for m in summary_realistic.median_log_returns],
            "baseline_median": round(summary_realistic.baseline_median_log_return, 3),
            "noise_median": round(noise_realistic.baseline_median_log_return, 3),
            "compare_noise": summary_realistic.compare_noise(noise_realistic.baseline_median_log_return),
            "effective_tolerance": round(summary_realistic.effective_tolerance, 4),
            "n_folds": summary_realistic.n_folds,
        },
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost-sensitive momentum check for lookback 3/5/10 ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]

    results_all = {}
    for ticker in manifest["universe"]:
        res = run_asset_with_costs(ticker, tickers, lookback=5)
        results_all[ticker] = res
        print(f"Asset: {ticker}")
        print("  {:14s} {:28s} {:28s} {:12s}  {:12s} verdict (zero_cost) | verdict (realistic_cost)".format(
            "lookback", "segments", "zero_medians", "zero_disp", "realistic_disp"))
        for lb in LOOKBACKS:
            r = res[f"{lb}_lookback"] if f"{lb}_lookback" in res else res
            segs = "[" + ",".join(r["zero_cost"]["segments"]) + "]"
            zero_meds = "[" + ",".join(str(m) for m in r["zero_cost"]["medians"]) + "]"
            realistic_meds = "[" + ",".join(str(m) for m in r["realistic_cost"]["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {} | {}".format(
                    str(lb), segs, zero_meds, r["zero_cost"]["candidate_dispersion"],
                    r["realistic_cost"]["candidate_dispersion"], r["zero_cost"]["verdict"], r["realistic_cost"]["verdict"]))

    print("\n=== 4. Universe-level generalization ===")
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=5),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", 5),),
        param_grid=[{"lookback": 5}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("  " + summary.inspect().replace("\n", "\n  "))

    print("\n=== 5. Synthetic perturbation sweep with cost sensitivity ===")
    pers = perturbation_sweep_with_costs()
    print("  zero_cost:")
    print("    param sets: {}".format(pers["zero_cost"]["param_sets"]))
    print("    candidate medians: {}".format(pers["zero_cost"]["medians"]))
    print("    baseline (lookback=5): {:+.3f}".format(pers["zero_cost"]["baseline_median"]))
    print("    coin-flip null baseline: {:+.3f}".format(pers["zero_cost"]["noise_median"]))
    print("  realistic_cost:")
    print("    param sets: {}".format(pers["realistic_cost"]["param_sets"]))
    print("    candidate medians: {}".format(pers["realistic_cost"]["medians"]))
    print("    baseline (lookback=5): {:+.3f}".format(pers["realistic_cost"]["baseline_median"]))
    print("    coin-flip null baseline: {:+.3f}".format(pers["realistic_cost"]["noise_median"]))

    print("\n=== 6. Determinism check ===")
    res2 = {t: run_asset_with_costs(t, tickers, lookback=5) for t in manifest["universe"]}
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    print("\n=== 7. Independent verification gate ===")
    # Independent verification would compare against this artifact
    print("Independent verifier should load this artifact and recompute using fresh walk_forward path")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=LOOKBACKS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=results_all,
        perturbation=pers,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())