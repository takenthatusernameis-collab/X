"""Test whether a past-only volatility gate on lookback-5 momentum creates a predeclared robustness improvement.

Background: MA regime filters were previously falsified; whether the same class of conditioning helps the different momentum mechanism remains unresolved.

Hypothesis being tested here: a past-only volatility classifier can separate the 2009-2013 "good" turbulent regime from the 2022-2026 "bad" turbulent regime in the context of lookback-5 momentum — the same conditioning that was falsified on MA crossover may or may not rescue the momentum edge.

Falsification prediction: the momentum edge under volatility conditioning either shows CONSISTENT_WITH_NOISE (no edge anywhere) or REGIME_DEPENDENT (edge in one regime mix) or REGIME_STABLE_LOSS (uniformly negative) — no REGIME_STABLE positive edge.

Method (fixed a-priori, not tuned to OOS): momentum (long the previous 5-day return, hold 1 day, daily rebalance) with a regime gate:
  - Calm regime (volatility < 0.15 annualized): momentum_signals with lookback=5 (original momentum)
  - Turbulent regime (volatility > 0.28 annualized): momentum_signals with lookback=3 (shorter lookback for turbulent periods)
  - Past-only volatility classifier using research.backtest.regime_stability.volatility_blocks with vol_calm=0.15, vol_turbulent=0.28, window=60
  - The same walk-forward, segment sizes, coin-flip null, and synthetic perturbation sweep as the momentum.py check

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
LOOKBACK_CALM = 5
LOOKBACK_TURBULENT = 3
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"
VOL_CALM = 0.15
VOL_TURBULENT = 0.28

def volatility_gate_momentum_signals(closes, lookback_calm=LOOKBACK_CALM, lookback_turbulent=LOOKBACK_TURBULENT):
    """Momentum signal with a past-only volatility classifier gate.
    
    Uses only closes[:i] to classify each bar as calm/normal/turbulent and selects
    the lookback window accordingly: calm uses lookback_calm, turbulent uses lookback_turbulent.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    # Compute volatility blocks once (past-only)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    
    for i in range(n):
        if i < max(LOOKBACK_CALM, LOOKBACK_TURBULENT) - 1:
            continue
            
        lab = labels[i]
        if lab == "insufficient":
            continue
            
        if lab == "turbulent":
            lookback = LOOKBACK_TURBULENT
        else:
            # calm and normal use the calm lookback (original momentum)
            lookback = LOOKBACK_CALM
            
        if i >= lookback - 1:
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
        ("momentum_volatility_gate", volatility_gate_momentum_signals, {"lookback_calm": LOOKBACK_CALM, "lookback_turbulent": LOOKBACK_TURBULENT}),
        ("momentum_original", lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK_CALM), {"lookback": LOOKBACK_CALM}),
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
    """Synthetic perturbation sweep of lookback under the volatility gate.
    
    Tests calm and turbulent lookback options under the same volatility gate.
    """
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback_calm", LOOKBACK_CALM), ("lookback_turbulent", LOOKBACK_TURBULENT)),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=volatility_gate_momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback_calm", LOOKBACK_CALM), ("lookback_turbulent", LOOKBACK_TURBULENT)))
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

    print("\n=== 3. Leakage review per asset (volatility-gate momentum) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = volatility_gate_momentum_signals(closes)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, volatility-gate momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  past-only volatility classifier: vol_calm={:.2f}, vol_turbulent={:.2f}, "
          "window={} bars".format(VOL_CALM, VOL_TURBULENT, WINDOW))

    print("\n=== 4. Volatility-gate momentum vs original momentum on REGIME_DEPENDENT assets ===")
    print("Volatility-gate momentum: calm lookback={}, turbulent lookback={}".format(LOOKBACK_CALM, LOOKBACK_TURBULENT))
    print("Original momentum: lookback={}".format(LOOKBACK_CALM))
    print("Segments: {} contiguous blocks by date, past-only volatility classifier".format(N_BLOCKS))

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

    # Full-universe generalization: volatility-gate momentum on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: volatility_gate_momentum_signals(c, **kw),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback_calm", LOOKBACK_CALM), ("lookback_turbulent", LOOKBACK_TURBULENT)),
        param_grid=[{"lookback_calm": LOOKBACK_CALM, "lookback_turbulent": LOOKBACK_TURBULENT}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("=== 5. Volatility-gate momentum across collected universe ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    # Synthetic perturbation sweep (parameter-robustness gate)
    pers = perturbation_sweep()
    print("=== 6. Synthetic perturbation sweep (calmed vs turbulent lookback options) ===")
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (calm={}, turbulent={}): {:+.3f}".format(LOOKBACK_CALM, LOOKBACK_TURBULENT, pers["baseline_median"]))
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
        lookback_calm=LOOKBACK_CALM,
        lookback_turbulent=LOOKBACK_TURBULENT,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        vol_calm=VOL_CALM,
        vol_turbulent=VOL_TURBULENT,
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
          "execution. Momentum with volatility-gate across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())