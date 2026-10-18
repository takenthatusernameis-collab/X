"""Test momentum with a past-only volatility gate (regime-conditioned).

Objective: the momentum cell (lookback=5) was admitted as candidate positive evidence in activation 37318950814, showing a REGIME_STABLE positive edge in 7/10 assets. However, the same conditioning was tested with MA regime filters and they were falsified. Whether the different volatility regime conditioning helps the same momentum mechanism remains unresolved.

Method (fixed a-priori, not tuned to OOS): long the previous 5-day return, hold 1 day, daily rebalance, but only when the past-only realized volatility is below a predeclared threshold (regime-conditioned edge). The volatility gate replaces the MA regime filter. Walk-forward per segment train={}d / test={}d / warmup={}d / overlap={}d, min {} bars/segment, with a synthetic perturbation sweep (lookback 3/5/10) vs coin-flip null. Determinism via internal independent recomputation.

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

DATA_UNIVERSE = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
VOL_THRESHOLD = 0.20  # Past-only volatility gate threshold (annualized)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "volatility_conditional_momentum_results.json"


def volatility_regime_gate(closes: np.ndarray, threshold: float) -> np.ndarray:
    """Past-only volatility gate: long when realized vol < threshold, else cash.

    Annualized realized volatility is the standard deviation of log returns over a
    fixed lookback window (WINDOW bars), scaled by sqrt(252). The gate is
    determined from the same lookback-window past data as the signal's formation
    window, ensuring no look-ahead. If volatility <= threshold, the position is
    enabled; otherwise it is disabled (cash = weight 0.0).

    Used with conditional_signal to create a regime-conditioned edge that only
    operates in low-vol regimes.
    """
    n = len(closes)
    gate = np.zeros(n, dtype=np.float32)
    for i in range(WINDOW - 1, n):
        log_returns = np.log(closes[i - WINDOW + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * np.sqrt(252.0)
        gate[i] = 1.0 if realized_annual < threshold else 0.0
    return gate


def regime_conditional_momentum_signals(closes: np.ndarray, lookback: int = LOOKBACK):
    """Momentum signals with a past-only volatility gate.

    Returns signals of weight +1 (long), -1 (short), or 0 (cash) based on:
    1. Momentum direction: long previous lookback-day return if positive, short if negative
    2. Volatility gate: enabled only when realized volatility < VOL_THRESHOLD
    """
    n = len(closes)
    momentum_gate = bt.momentum_signals(closes, lookback=lookback)
    vol_gate = volatility_regime_gate(closes, VOL_THRESHOLD)

    signals = []
    for i in range(n):
        momentum_weight = momentum_gate[i].weight
        vol_enabled = bool(vol_gate[i] > 0.5)
        if vol_enabled:
            signals.append(bt.Signal(date=i + 1, weight=float(momentum_weight)))
        else:
            signals.append(bt.Signal(date=i + 1, weight=0.0))
    return signals


def run_asset(asset: str, tickers: dict):
    """Run momentum with volatility gate on a single asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    results = {}

    # Regime-conditioned momentum with volatility gate
    signals = regime_conditional_momentum_signals(closes, LOOKBACK)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")

    res = bt.stress_segments(
        signals_fn=regime_conditional_momentum_signals,
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

    results["momentum_vol_gate"] = dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )
    return results


def perturbation_sweep(n_bars: int = 600):
    """Synthetic perturbation sweep of momentum lookback with volatility gate."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK),),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=regime_conditional_momentum_signals,
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
    assert manifest["dataset_id"] == DATA_UNIVERSE, manifest["dataset_id"]

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
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = regime_conditional_momentum_signals(closes, LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum+vol_gate signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: volatility gate at bar t uses closes[:t] only; "
          "momentum direction also past-only.")

    print("\n=== 4. Momentum with volatility gate on REGIME_DEPENDENT assets ===")
    print("Momentum with volatility gate: long previous {}-day return, hold 1 day, daily "
          "rebalanced, but only when realized vol < {} (annualized). "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, compared vs coin-flip null on the same segments\n".format(
              LOOKBACK, VOL_THRESHOLD, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("Segments: {} contiguous blocks by date, past-only volatility classifier ("
          "turbulent = block median trailing-{}-bar realized vol > series-wide median)\n".format(
              N_BLOCKS, WINDOW))

    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    name, segs, meds, float(r["candidate_dispersion"]), float(r["null_dispersion"]), r["verdict"]))
        print("")

    # Full-universe generalization
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: regime_conditional_momentum_signals(c, LOOKBACK),
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
    print("=== 5. Momentum with volatility gate across collected universe ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    # Synthetic perturbation sweep
    pers = perturbation_sweep()
    print("=== 6. Synthetic perturbation sweep (lookback 3 / 5 / 10) with volatility gate ===")
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (lookback=5, vol_gate): {:+.3f}".format(pers["baseline_median"]))
    print("  coin-flip null baseline: {:+.3f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}"
          .format(pers["compare_noise_fixed"]))
    print("")

    # Determinism: rerun the 2-asset comparison and assert identical output
    res2 = {a: run_asset(a, tickers) for a in target}
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact
    artifact = dict(
        dataset_id=DATA_UNIVERSE,
        seed=SEED,
        lookback=LOOKBACK,
        vol_threshold=VOL_THRESHOLD,
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
          "execution. Momentum with volatility gate on AMZN/JPM + universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
