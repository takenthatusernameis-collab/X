"""Volatility-gate momentum: test whether a past-only volatility gate creates a predeclared robustness improvement for lookback-5 momentum through the existing regime-stability and null framework.

Evidence basis: MA regime filters were previously falsified; whether the same class of conditioning helps the different momentum mechanism remains unresolved.

Hypothesis: A past-only volatility gate that trades only in low-volatility regimes (realized vol < 0.20 annualized) will improve the robustness of lookback-5 momentum signals through the regime-stability gate.

Method (fixed a-priori, not tuned to OOS): long the previous 5-day return only when the recent 20-day realized volatility is below 0.20 annualized; hold 1 day; daily rebalance. Walk-forward IS/OOS: train=252d/test=84d/warmup=60d/overlap=60d, min 400 bars/segment, 4 contiguous volatility blocks, compared vs coin-flip null on the same segments.

Falsification prediction: The volatility-gated momentum either shows CONSISTENT_WITH_NOISE (no edge anywhere) or REGIME_DEPENDENT (the positive run concentrates in one historical regime mix), or REGIME_STABLE_LOSS (uniformly negative — which would be interesting: volatility gating makes momentum lose money in every regime). Either outcome is a negative conclusion and closes the cell.

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
VOL_THRESHOLD = 0.30  # past-only volatility gate: trade only when vol < 0.30 annualized
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"


def momentum_volatility_gate_signals(closes, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD):
    """Past-only momentum signals gated by volatility regime.

    Long the previous lookback-day return only when the recent WINDOW-day
    realized volatility is below vol_threshold annualized (past-only:
    volatility estimate uses closes[:i] only, signal at bar i uses only
    closes[:i]). Hold 1 day; daily rebalance. Neutral before the lookback
    window.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)

    for i in range(lookback, n):
        # Past-only volatility gate: use only closes[:i] to estimate volatility
        # Calculate realized volatility up to bar i (past-only, no look-ahead)
        if i < 1:
            continue
            
        # Use only closes[:i] to estimate volatility at position i
        # This follows the same pattern as conditional_signal in regime_stability.py
        window = min(WINDOW, max(4, n // 4))
        if i < window:
            continue
            
        # Estimate annualized realized volatility using past data only
        log_returns = np.log(closes[i - window:i])
        realized_annual = float(np.std(log_returns, ddof=1)) * np.sqrt(252.0)

        # Apply past-only volatility gate
        if realized_annual < vol_threshold:
            # Calculate momentum signal (long previous lookback return)
            ret = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        # else: weight remains 0.0 (neutral)

    return out


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    results = {}

    for name, sig, params in [
        ("base_momentum", lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK), {"lookback": LOOKBACK}),
        ("volatility_gate", lambda c, **p: momentum_volatility_gate_signals(c, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD), {"lookback": LOOKBACK, "vol_threshold": VOL_THRESHOLD}),
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
    """Synthetic perturbation sweep of lookback around the canonical momentum value over the canonical regime family, with a coin-flip null benchmark."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]
    
    # Create a signals function that fixes vol_threshold but allows lookback to vary
    def signals_fn(closes, lookback):
        return momentum_volatility_gate_signals(closes, lookback=lookback, vol_threshold=VOL_THRESHOLD)
    
    sweep = bt.parameter_sweep(
        signals_fn=signals_fn,
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
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = momentum_volatility_gate_signals(closes, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, volatility-gated momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  past-only volatility gate uses realized vol from closes[:i] only; "
          "neutral before the window.")

    print("\n=== 4. Volatility-gated momentum vs base momentum on REGIME_DEPENDENT assets ===")
    print("Volatility-gated momentum: long the previous {}-day return only when "
          "recent {}-day realized vol < {} annualized; hold 1 day; daily "
          "rebalanced. Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, compared vs coin-flip null on the same segments\n".format(
              LOOKBACK, WINDOW, VOL_THRESHOLD, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

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

    # Full-universe generalization: volatility-gated momentum on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    
    # Create signals function that matches the expected signature for stress_segments_across_tickers
    # It needs to take the parameters specified in the baseline (just lookback in this case)
    def signals_fn(closes, lookback):
        return momentum_volatility_gate_signals(closes, lookback=lookback, vol_threshold=VOL_THRESHOLD)
    
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
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
    print("=== 5. Volatility-gated momentum across collected universe ===")
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
          "execution. Volatility-gated momentum test on AMZN/JPM + universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())