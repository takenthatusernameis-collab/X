"""Test regime-filtered momentum on the collected universe.

Background: momentum (long the previous 5-day return, hold 1 day, daily
rebalanced) is the only signal class in this series to carry candidate positive
evidence — REGIME_STABLE uniformly-positive medians in 7/10 collected assets,
lookback-robust at lookbacks 3 / 5 / 10. But is the positive edge regime-stable
or regime-concentrated? If the edge lives mostly in one volatility regime,
restricting trading to that regime either preserves a REGIME_STABLE edge (edge
is genuinely robust) or converts the class to CONSISTENT_WITH_NOISE /
REGIME_STABLE_LOSS (edge was an artifact of regime mix), which would demote
momentum from candidate evidence to regime-dependent/exploratory.

This is the natural follow-on to the falsified regime-adaptive MA (which
adapted windows and either killed the edge or kept the dependence). Here the
regime gate is a yes/no filter on a single fixed momentum signal: test whether
momentum's edge concentrates in the turbulent regime, the calm regime, or
persists in both.

Falsification prediction: a regime filter that kills the edge (both filtered
variants landing CONSISTENT_WITH_NOISE or REGIME_STABLE_LOSS) would demote
momentum to regime-dependent/exploratory; a filter that preserves a
REGIME_STABLE positive edge in both regimes would strengthen the case. Either
outcome is informative.

Method (fixed a-priori, not tuned to OOS): momentum = long the previous 5-day
return, hold 1 day, daily rebalance. Regime blocks = 4 contiguous volatility
blocks from AAPL trailing-60d realized vol (past-only). Walk-forward per
segment train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment,
compared vs a coin-flip null on the same segments. Three variants on AMZN/JPM:
(1) unrestricted/base momentum (reference), (2) turbulent-only momentum,
(3) calm-only momentum; then the same three variants across the 10-asset
collected universe via stress_segments_across_tickers; synthetic perturbation
sweep (lookback 3/5/10 x top_k 3/5 over the canonical regime family, coin-flip
null) for parameter robustness; internal determinism assertion; artifact
written to state/check_artifacts/regime_filtered_momentum_results.json.

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
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "regime_filtered_momentum_results.json"


def base_signals(closes, lookback):
    """Base (unrestricted) momentum."""
    return bt.momentum_signals(closes, lookback=lookback)


def run_asset(asset, tickers, variant_name, keep=None):
    """Run one variant on one asset through stress_segments."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    if keep is None:
        signals_fn = lambda c, **p: base_signals(c, LOOKBACK)
    else:
        signals_fn = lambda c, **p: bt.regime_filtered_momentum_signals(
            c, keep=keep
        )
    signals = signals_fn(closes)
    if len(signals) != len(closes):
        raise ValueError(f"{asset} {variant_name}: signals {len(signals)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=signals_fn,
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": LOOKBACK}],
        baseline=(("lookback", LOOKBACK),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    names = [s.name for s in res.scenarios]
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    res = bt.stress_segments(
        signals_fn=signals_fn,
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": LOOKBACK}],
        baseline=(("lookback", LOOKBACK),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    names = [s.name for s in res.scenarios]
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    return dict(
        segments=names,
        medians=medians,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )


def universe_sweep(keep):
    """Run one variant across the full 10-asset collected universe."""
    manifest = load_manifest()
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    if keep is None:
        signals_fn = lambda c, **kw: base_signals(c, LOOKBACK)
    else:
        signals_fn = lambda c, **kw: bt.regime_filtered_momentum_signals(c, keep=keep)
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=signals_fn,
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK),),
        param_grid=[{"lookback": LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return summary


def perturbation_sweep(n_bars=600):
    """Synthetic perturbation sweep of lookback for the (unrestricted) momentum
    signal over the canonical regime family, with a coin-flip null benchmark.
    The synthetic generator (regime-switching GBM, no return autocorrelation)
    contains no momentum structure, so the baseline matches the null by
    construction; the sweep is a tooling-validity check, not an edge-robustness
    test. The operative robustness evidence for momentum is the real-data
    regime gate, where each variant (base, turbulent-only, calm-only) is
    compared vs its own coin-flip null on the same segments."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK),),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", LOOKBACK),))
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    return dict(
        param_sets=[tuple(ps) for ps in sweep.param_sets],
        medians=[round(m, 3) for m in summary.median_log_returns],
        baseline_median=round(summary.baseline_median_log_return, 3),
        noise_median=round(noise.baseline_median_log_return, 3),
        compare_noise_default=summary.compare_noise(noise.baseline_median_log_return),
        compare_noise_fixed=summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
        effective_tolerance=round(summary.effective_tolerance, 4),
        n_folds=summary.n_folds,
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

    print("\n=== 3. Leakage review per asset ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    variants = [
        ("base", None),
        ("turbulent_only", "turbulent"),
        ("calm_only", "calm"),
    ]
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        for var_name, keep in variants:
            if keep is None:
                signals = base_signals(closes, LOOKBACK)
            else:
                signals = bt.regime_filtered_momentum_signals(closes, keep=keep)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
            total_return = (res.equity_curve[-1] - 1e6) / 1e6
            print(f"  {ticker} {var_name}: {bars.n_bars} bars, signals {len(signals)}, "
                  f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: regime-filtered momentum sign at bar t uses closes[:t] only "
          f"(labels from volatility_blocks, trailing-{WINDOW}-bar realized vol); "
          "neutral bars carry the equity forward.")

    print("\n=== 4. Momentum vs base momentum on REGIME_DEPENDENT assets (AMZN/JPM) ===")
    print("Three variants: unrestricted/base (reference), momentum active only in "
          "turbulent segments, momentum active only in calm segments. Params: "
          "walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, compared vs coin-flip null on the "
          "same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (turbulent = block median trailing-{}-bar realized vol "
          "> series-wide median)\n".format(N_BLOCKS, WINDOW))

    results_all = {}
    for var_name, keep in variants:
        asset_results = {}
        for asset in target:
            res = run_asset(asset, tickers, var_name, keep)
            asset_results[asset] = res
        results_all[var_name] = asset_results
        print("Variant: {}".format(var_name))
        print("  {:12s} {:28s} {:28s} {:12s}  {:12s}  verdict".format(
            "asset", "segments", "candidate_medians", "cand_disp", "null_disp"))
        for asset, r in asset_results.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:12s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    asset, segs, meds, r["candidate_dispersion"],
                    r["null_dispersion"], r["verdict"]))
        print("")

    print("=== 5. Momentum across collected universe (3 variants) ===")
    universe_results = {}
    for var_name, keep in variants:
        summary = universe_sweep(keep)
        universe_results[var_name] = summary
        print("Variant: {} | ".format(var_name) + summary.inspect().replace("\n", "\n  "))
        print("")

    # Synthetic perturbation sweep (parameter-robustness gate)
    pers = perturbation_sweep()
    print("=== 6. Synthetic perturbation sweep (lookback 3 / 5 / 10; base momentum) ===")
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline: {:+.3f}".format(pers["baseline_median"]))
    print("  coin-flip null: {:+.3f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}"
          .format(pers["compare_noise_fixed"]))
    print("")

    # Determinism: rerun the 2-asset comparison and assert identical output
    res2 = {}
    for var_name, keep in variants:
        asset_results = {}
        for asset in target:
            asset_results[asset] = run_asset(asset, tickers, var_name, keep)
        res2[var_name] = asset_results
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact
    def asset_result_to_dict(res):
        segs = res["segments"]
        meds = res["medians"]
        return dict(
            segments=segs,
            medians=meds,
            candidate_dispersion=res["candidate_dispersion"],
            null_dispersion=res["null_dispersion"],
            verdict=res["verdict"],
            n_folds=res["n_folds"],
        )

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        variants={
            var_name: {
                asset: asset_result_to_dict(r)
                for asset, r in res.items()
            }
            for var_name, res in results_all.items()
        },
        universe={
            var_name: dict(
                verdict_counts=summary.verdict_counts,
                ticker_order=summary.ticker_order,
                per_asset=dict(
                    (t, dict(
                        segments=[s.name for s in r.scenarios],
                        medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                        null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                        candidate_dispersion=round(r.candidate_dispersion, 3),
                        null_dispersion=round(r.null_dispersion, 3),
                        verdict=r.overall_verdict,
                    ))
                    for t, r in summary.assets.items()
                ),
            )
            for var_name, summary in universe_results.items()
        },
        perturbation=pers,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Regime-filtered momentum on AMZN/JPM + 10-asset universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
