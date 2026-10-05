"""Test the new mean-reversion signal class on the collected universe.

Background: the trend-following MA-crossover class was exhaustively explored
on the collected adjusted-close universe and rejected — no parameter set
peaked at the canonical (20,60) in any asset, the full-sample walk-forward
sat inside the sample-calibrated coin-flip noise band, only 1/10 assets was
marginally significant (t-test) and none survived a Bonferroni family-wise
correction, and the regime-stability verdicts were mixed
(CONSISTENT_WITH_NOISE=4, REGIME_STABLE=4, REGIME_DEPENDENT=2) rather than
REGIME_STABLE everywhere. The MA-crossover frontier cell is therefore closed.

The next frontier cell is a new signal class: mean reversion. Hypothesis:
short-horizon return reversal (bet against the previous 5-day return,
daily-rebalanced) is a REGIME_STABLE edge on large-cap US equities across
2009-2026. Falsification prediction: reversal, like MA crossover, shows
either CONSISTENT_WITH_NOISE (no edge anywhere) or REGIME_DEPENDENT (an
edge concentrated in one historical regime mix), because the 2009-2013
post-crisis reversal environment differs structurally from 2022-2026 and a
1-day-ahead reversal cannot condition on regime quality with past-only data.

This is a principled, fixed a-priori design (short the previous 5-day return;
hold 1 day; daily rebalance), not tuned to OOS results. The variant is run
through the SAME `stress_segments` pipeline and coin-flip null as the base
MA(20/60), so verdicts are directly comparable.

Method: for the two REGIME_DEPENDENT assets from the MA run (AMZN, JPM), run
both base MA(20/60) and reversal(lookback=5) through `stress_segments` over
the 4 contiguous volatility blocks, and sweep reversal across the full
collected universe. Also run a synthetic perturbation sweep with the canonical
regime family plus a coin-flip null to exercise the parameter-robustness gate
on the new signal. Determinism is asserted via an internal independent
recomputation.

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
ARTIFACT_PATH = ARTIFACT_DIR / "mean_reversion_results.json"


def base_signals(closes, fast, slow):
    """Past-only MA crossover (same contract as research/backtest/regime_stability)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fm = float(np.mean(closes[i - fast + 1 : i + 1]))
        sm = float(np.mean(closes[i - slow + 1 : i + 1]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return out


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}

    for name, sig, params in [
        ("base_ma", lambda c, **p: base_signals(c, 20, 60), {"fast": 20, "slow": 60}),
        ("mean_reversion", lambda c, **p: bt.mean_reversion_signals(c, lookback=LOOKBACK), {"lookback": LOOKBACK}),
    ]:
        signals = sig(closes)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} {name}: signals {len(signals)} != bars {len(closes)}")
        res = bt.stress_segments(
            signals_fn=sig,
            bars=list(bars),
            signals=signals,
            param_grid=[params],
            baseline=tuple(params.items()),
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
        results[name] = dict(
            segments=names,
            medians=medians,
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=res.overall_verdict,
            n_folds=res.n_folds,
        )
    return results


def perturbation_sweep(n_bars=600):
    """Synthetic perturbation sweep of reversal lookback over the canonical
    regime family, with a coin-flip null benchmark."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK),),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=bt.mean_reversion_signals,
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
    cfg0 = bt.BacktestConfig(initial_capital=1e6)
    target = ["AMZN", "JPM"]
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.mean_reversion_signals(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg0)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - cfg0.initial_capital) / cfg0.initial_capital
        print(f"  {ticker}: {bars.n_bars} bars, reversal signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    tickers = {t: bt.load_ticker(t)[0] for t in target}

    print("\n=== 4. Mean-reversion vs base MA(20/60) on REGIME_DEPENDENT assets ===")
    print("Reversal: short the previous {}-day return, hold 1 day, daily "
          "rebalanced.  Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, compared vs coin-flip null on the same segments\n".format(
          LOOKBACK, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (turbulent = block median trailing-{}-bar realized vol "
          "> series-wide median)\n".format(N_BLOCKS, WINDOW))

    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:14s} {:28s} {:28s} {:12s}  {:12s}  verdict".format(
            "variant", "segments", "candidate_medians", "cand_disp", "null_disp"))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    name, segs, meds, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
        print("")

    # Full-universe generalization: reversal on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: bt.mean_reversion_signals(c, lookback=LOOKBACK),
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
    print("=== 5. Mean-reversion across collected universe ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    # Synthetic perturbation sweep (parameter-robustness gate)
    pers = perturbation_sweep()
    print("=== 6. Synthetic perturbation sweep (lookback 3 / 5 / 10) ===")
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (lookback=5): {:+.3f}".format(pers["baseline_median"]))
    print("  coin-flip null baseline: {:+.3f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}"
          .format(pers["compare_noise_fixed"]))
    print("")

    # Determinism: rerun the 2-asset comparison and assert identical output
    res2 = {a: run_asset(a, tickers) for a in target}
    out2_lines = []
    for asset in target:
        out2_lines.append("Asset: {}".format(asset))
        for name, r in res2[asset].items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            out2_lines.append(
                "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    name, segs, meds, r["candidate_dispersion"],
                    r["null_dispersion"], r["verdict"]))
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact that the independent verification file recomputes
    # from a separate path and compares against.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=results_all,
        universe=dict(
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
        ),
        perturbation=pers,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Mean-reversion test on AMZN/JPM + universe: exploratory "
          "simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
