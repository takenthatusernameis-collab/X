"""Test momentum signal with a past-only volatility gate on the collected universe.

Background: momentum (long previous 5-day return, hold 1 day, daily rebalance) is the direct opponent to reversal.
Reversal was exhaustively explored on the collected 10-asset universe and rejected: every segment of every asset showed uniformly negative walk-forward log returns; the perturbation sweep was flat on the coin-flip null. Momentum is the contraposition to reversal with identical lookback, walk-forward, and coin-flip null; it differs only in signal sign. If reversal loses everywhere, momentum should make the opposite mistake. This test settles whether the 5-day-lookback direction bet is a robust edge in this universe, or whether it is noise with a direction sign attached.

Falsification prediction: momentum either shows CONSISTENT_WITH_NOISE (no edge anywhere) or REGIME_DEPENDENT (the positive run concentrates in one historical regime mix, notably 2009-2013), or REGIME_STABLE_LOSS (uniformly negative — which would be reversal winning on both sides, an inconsistency to explain). Either outcome is a negative conclusion and closes the cell.

Method (fixed a-priori, not tuned to OOS): momentum_signals(closes, lookback=5); volatility_gate(closes) emits signals only when past-only trailing-60-bar realized volatility > VOLATILITY_THRESHOLD. The threshold is predeclared before any execution. Full gate stack: manifest preflight, leakage review, walk-forward regime-stability with matched coin-flip null, synthetic perturbation sweep (lookback 3/5/10), determinism assertion.

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
VOLATILITY_THRESHOLD = 0.30  # 30% annualized realized vol, past-only (used in no-look-ahead volatility_blocks logic)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_with_volatility_gate_results.json"


def base_momentum_signals(closes, lookback=5):
    """Past-only momentum: long previous lookback-day return, hold 1 day, daily rebalance.
    Neutral before the lookback window. This is the original momentum signal class.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def momentum_with_volatility_gate(closes, lookback=5, threshold=VOLATILITY_THRESHOLD):
    """Momentum signals gated by past-only trailing-60-bar realized volatility.
    
    Past-only volatility gate: compute trailing-60-bar realized vol (annualized)
    using only closes[:i]; trade only if vol > threshold. If vol condition fails,
    emit neutral (weight=0) for all bars from the start.
    
    Args:
        closes: array of closing prices.
        lookback: momentum lookback window (bars).
        threshold: annualized realized-vol threshold; trade only if vol > threshold.
    
    Returns:
        List of Signal objects, neutral when vol gate fails.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    # Precompute trailing-60-bar realized vol for each bar (past-only)
    vol_labels = bt.volatility_blocks(closes, n_blocks=1, window=60)
    # Convert to numeric vol values for thresholding
    for i in range(60, n):
        if vol_labels[i] == "insufficient":
            continue
        # vol_labels already classified into 'calm'/'turbulent' based on 60-bar trailing vol
        # Map turbulent -> high vol, calm -> low vol
        if vol_labels[i] == "turbulent":
            # Compute actual realized vol for thresholding (consistent with volatility_blocks)
            window_start = i - 60
            window_rets = np.log(closes[window_start : i + 1]) - np.log(closes[window_start : i])
            rv = float(np.std(window_rets, ddof=1)) * np.sqrt(252.0)
        else:
            # For calm, compute but expect below threshold
            window_start = i - 60
            window_rets = np.log(closes[window_start : i + 1]) - np.log(closes[window_start : i])
            rv = float(np.std(window_rets, ddof=1)) * np.sqrt(252.0)
        
        if rv > threshold:
            # Apply momentum signal only when vol gate passes
            if i >= lookback:
                ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
                out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    
    return out


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}

    for name, sig_fn, params in [
        ("base_momentum", lambda c, **p: base_momentum_signals(c, lookback=LOOKBACK), {"lookback": LOOKBACK}),
        ("momentum_with_gate", lambda c, **p: momentum_with_volatility_gate(c, lookback=LOOKBACK, threshold=VOLATILITY_THRESHOLD), {"lookback": LOOKBACK, "threshold": VOLATILITY_THRESHOLD}),
    ]:
        signals = sig_fn(closes)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} {name}: signals {len(signals)} != bars {len(closes)}")
        res = bt.stress_segments(
            signals_fn=sig_fn,
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
    """Synthetic perturbation sweep of momentum lookback over the canonical
    regime family, with a coin-flip null benchmark.
    """
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

    print("\n=== 3. Leakage review per asset (REGIME_DEPENDENT assets: AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = momentum_with_volatility_gate(closes, lookback=LOOKBACK, threshold=VOLATILITY_THRESHOLD)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum+gate signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward. "
          f"Vol gate: trade only if trailing-60-bar realized vol > {VOLATILITY_THRESHOLD:.2f} (annualized).")

    print("\n=== 4. Momentum with volatility gate vs base momentum on REGIME_DEPENDENT assets ===")
    print("Momentum+vol gate: long previous 5-day return only when trailing-60-bar realized vol > {VOLATILITY_THRESHOLD:.2f}.".format(VOLATILITY_THRESHOLD=VOLATILITY_THRESHOLD))
    print("Params: walk-forward per segment train={}d / test={}d / warmup={}d / overlap={}d, min {} "
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

    # Full-universe generalization: momentum+vol gate on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: momentum_with_volatility_gate(c, lookback=LOOKBACK, threshold=VOLATILITY_THRESHOLD),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK), ("threshold", VOLATILITY_THRESHOLD)),
        param_grid=[{"lookback": LOOKBACK, "threshold": VOLATILITY_THRESHOLD}],
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
        volatility_threshold=VOLATILITY_THRESHOLD,
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
          "execution. Momentum+vol gate test on AMZN/JPM + universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
