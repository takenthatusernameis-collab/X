#!/usr/bin/env python3
"""Test a predeclared volatility-gated momentum signal on the 10-asset universe.

Evidence gate: a momentum signal that only emits positions in 'turbulent' volatility
segments (past-only classification from volatility_blocks). The gate is applied
before any look-ahead bias is introduced: a momentum signal computed at bar i uses
only closes[:i]; the turbulence label comes from the same closes[:i] series; the
final signal is the product of both constraints.

This is the regime-conditioning that the bottleneck asks about: does a single
past-only volatility gate create a predeclared robustness improvement for the
qualified momentum effect?

Method (fixed a-priori, not tuned to OOS):
- lookback=5 momentum signals (long previous 5-day return, hold 1 day, daily rebalance)
- turbulence gate: emit the momentum signal only inside 'turbulent' segments
- regime labels: 4 contiguous blocks by date, 'turbulent' if the block's median
trailing-60-bar realized volatility exceeds the series-wide median (past-only)
- same walk-forward contract as existing momentum tests:
    train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment
- compare against a coin-flip null run on the same segments via the existing
  regime-stability framework.

Predeclared gate definition (do not modify after execution):
    final_signal[t] = momentum[t] if labels[t] == 'turbulent' else 0

Interpretation: a REGIME_DEPENDENT verdict would indicate the edge lives only in
one regime mix — the gate is not separating the edge from the noise. A
REGIME_STABLE verdict would suggest the gate is helping to extract a regime-stable
edge. CONSISTENT_WITH_NOISE indicates no edge regardless of gate;
REGIME_STABLE_LOSS is a negative conclusion (stable losses).

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
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# ============================================================================
# Predeclared volatility-gated momentum signal definition
# Do NOT modify after execution; this is the evidence gate.
# ============================================================================

def volatility_gated_momentum_signals(closes: np.ndarray, labels, **kwargs) -> list[bt.Signal]:
    """Past-only momentum signal gated by turbulence label.

    Implements the predeclared gate: emit momentum positions only in 'turbulent'
    volatility segments. Both momentum calculation and turbulence labeling are
    past-only (no look-ahead). This is the candidate volatility gate being tested.

    Args:
        closes: price series (full length).
        labels: per-bar regime labels from volatility_blocks (past-only).
        **kwargs: ignored, keeps signature compatible with framework.

    Returns:
        Signal series: momentum[t] if labels[t] == 'turbulent' else neutral.
    """
    n = len(closes)
    # 1. momentum component (past-only)
    momentum = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(LOOKBACK, n):
        ret = float(np.mean(np.log(closes[i - LOOKBACK + 1 : i + 1])))
        momentum[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)

    # 2. apply predeclared turbulence gate
    out = []
    for i in range(n):
        if labels[i] == 'turbulent':
            out.append(momentum[i])   # keep momentum signal
        else:
            out.append(bt.Signal(date=i + 1, weight=0.0))  # neutral
    return out


def run_asset(ticker: str, tickers: dict):
    """Run regime-stability test for one asset using the predeclared volatility gate."""
    bars = tickers[ticker]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    results = {}
    for name, sig, params in [
        ("vol_gated_mom", volatility_gated_momentum_signals, {"labels": labels}),
    ]:
        signals = sig(closes, **params)
        if len(signals) != len(closes):
            raise ValueError(f"{ticker} {name}: signals {len(signals)} != bars {len(closes)}")

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


def perturbation_sweep(n_bars: int = 600):
    """Synthetic perturbation sweep of lookback (3/5/10) for the gated momentum.

    This tests parameter robustness of the volatility-gated momentum gate,
    using the same contract as the existing momentum perturbation sweep.
    """
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK),),
        multipliers=(0.5, 1.0, 2.0),
    )

    def gated_mom_signals_for_perturb(closes, lookback=LOOKBACK):
        # Create turbulence labels for synthetic data
        n = len(closes)
        block_size = n // N_BLOCKS
        bar_vols = []
        for i in range(n):
            if i < WINDOW:
                bar_vols.append(0.0)
            else:
                bar_vols.append(
                    float(np.std(np.log(closes[i - WINDOW : i]), ddof=1)) * np.sqrt(252.0)
                )
        active = [v for v in bar_vols if v > 0]
        if not active:
            labels = ['insufficient'] * n
        else:
            series_med = float(np.median(active))
            labels = []
            for i in range(n):
                if i < WINDOW:
                    labels.append('insufficient')
                else:
                    block = i // block_size
                    s, e = block * block_size, (block + 1) * block_size
                    bmed = float(
                        np.median([v for v in bar_vols[s:e] if v > 0])
                        if e - s > WINDOW
                        else 0.0
                    )
                    labels.append('turbulent' if bmed > series_med else 'calm')

        # Compute momentum with given lookback
        sig = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        for i in range(lookback, n):
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            sig[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)

        # Apply same turbulence gate
        out = []
        for i in range(n):
            if labels[i] == 'turbulent':
                out.append(sig[i])
            else:
                out.append(bt.Signal(date=i + 1, weight=0.0))
        return out

    sweep = bt.parameter_sweep(
        signals_fn=gated_mom_signals_for_perturb,
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
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_PATH = ARTIFACT_DIR / "vol_gated_momentum_results.json"

    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Volatility-gated momentum on REGIME_DEPENDENT assets (AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        signals = volatility_gated_momentum_signals(closes, labels)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, leakage [PASS], total_return {total_return:+.3f}")

    print("\n=== 4. Volatility-gated momentum across collected universe ===")
    print(f"lookback={LOOKBACK}, turbulence gate = signals only in 'turbulent' segments")
    print(f"train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, {N_BLOCKS} contiguous blocks by date")

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
            print("  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                name, segs, meds, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
        print("")

    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: volatility_gated_momentum_signals(c, labels=bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW)),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline={"labels": bt.volatility_blocks(bt.load_ticker("AMZN")[0].closes_array(), n_blocks=N_BLOCKS, window=WINDOW)},
        param_grid=[{"labels": bt.volatility_blocks(bt.load_ticker("AMZN")[0].closes_array(), n_blocks=N_BLOCKS, window=WINDOW)}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("=== 5. Universe-level verdict ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    print("=== 6. Synthetic perturbation sweep (lookback 3/5/10) ===")
    pers = perturbation_sweep()
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (lookback=5): {:+.3f}".format(pers["baseline_median"]))
    print("  coin-flip null baseline: {:+.3f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}".format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}".format(pers["compare_noise_fixed"]))
    print("")

    print("=== 7. Determinism (independent re-run) ===")
    res2 = {a: run_asset(a, tickers) for a in target}
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"
    print("")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Volatility-gated momentum test on AMZN/JPM + universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
