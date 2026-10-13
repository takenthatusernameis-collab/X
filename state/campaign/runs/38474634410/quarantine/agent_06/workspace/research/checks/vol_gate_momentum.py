"""Test a past-only volatility gate on lookback-5 momentum.

Background: The research tests whether a single past-only volatility gate
creates a predeclared robustness improvement for the qualified momentum
edge. This is the evidence gate: a volatility condition that must be
past-only and fixed a-priori, not tuned to results.

Previous attempts on MA-crossover signals with active_when_turbulent (vol > 0.25)
were falsified; this test applies the same conditioning class to the
momentum mechanism.

Method: Trade only when past 60-bar realized volatility exceeds 0.20
(annualized), creating a turbulent-regime filter for lookback-5 momentum.
The gate is predeclared: vol_threshold = 0.20, lookback = 60, applied
past-only with daily rebalance (long previous 5-day return).

The regime-stability framework tests the conditional momentum across
volatility blocks, compares to a coin-flip null, and checks if the gate
either provides a reproducible edge or is falsified.
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
VOL_LOOKBACK = 60
VOL_THRESHOLD = 0.20
MOMENTUM_LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "vol_gate_momentum_results.json"


def active_when_turbulent_gate(early_vol: float) -> bool:
    """Past-only volatility gate: trade only in high-vol regimes.
    
    The gate is predeclared: VOL_THRESHOLD = 0.20 (vs previous 0.25).
    This is the exact volatility condition being tested for momentum.
    """
    return early_vol > VOL_THRESHOLD


def momentum_with_vol_gate(closes: np.ndarray, vol_lookback: int = VOL_LOOKBACK) -> bt.Signal:
    """Momentum signals gated by past-only volatility.
    
    Step 1: compute past-only realized volatility using only closes[:i].
    Step 2: apply predeclared gate VOL_THRESHOLD.
    Step 3: emit momentum signal (long previous MOMENTUM_LOOKBACK-day return)
             only when gate is true.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    # Past-only volatility window starts at vol_lookback - 1
    for i in range(vol_lookback - 1, n):
        # Step 1: past-only realized volatility
        log_returns = np.log(closes[i - vol_lookback + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * np.sqrt(252.0)
        
        # Step 2: apply predeclared gate
        if not (realized_annual > VOL_THRESHOLD):
            continue
            
        # Step 3: momentum signal (only when gate true)
        if i >= MOMENTUM_LOOKBACK:
            ret = float(np.mean(np.log(closes[i - MOMENTUM_LOOKBACK + 1 : i + 1])))
            signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
            
    return signals


def run_asset(asset, tickers):
    """Run volatility-gated momentum on one asset across volatility blocks."""
    bars = tickers[asset]
    closes = bars.closes_array()
    
    # Volatility block classification (past-only, from research/backtest/regime_stability)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Volatility-gated momentum signals
    signals = momentum_with_vol_gate(closes, VOL_LOOKBACK)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    
    # Stress segments across volatility blocks
    res = bt.stress_segments(
        signals_fn=lambda c, **kw: momentum_with_vol_gate(c, VOL_LOOKBACK),
        bars=list(bars),
        signals=signals,
        param_grid=[{"vol_lookback": VOL_LOOKBACK, "threshold": VOL_THRESHOLD}],
        baseline=(("vol_lookback", VOL_LOOKBACK), ("threshold", VOL_THRESHOLD)),
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
    
    results = {}
    results["vol_gate_momentum"] = dict(
        segments=names,
        medians=medians,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )
    return results


def perturbation_sweep(n_bars: int = 600):
    """Synthetic perturbation sweep testing parameter robustness of the gate.
    
    Varies the lookback window around the predeclared value to test robustness.
    """
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("vol_lookback", VOL_LOOKBACK),),
        multipliers=(0.5, 1.0, 2.0),
    )
    
    # Note: this sweep only tests the lookback parameter, threshold is fixed
    def gated_signals_fn(closes: np.ndarray, vol_lookback: float) -> bt.Signal:
        signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]
        vol_lookback_int = int(vol_lookback)
        for i in range(vol_lookback_int - 1, len(closes)):
            log_returns = np.log(closes[i - vol_lookback_int + 1 : i + 1])
            realized_annual = float(np.std(log_returns, ddof=1)) * np.sqrt(252.0)
            if not (realized_annual > VOL_THRESHOLD):
                continue
            if i >= MOMENTUM_LOOKBACK:
                ret = float(np.mean(np.log(closes[i - MOMENTUM_LOOKBACK + 1 : i + 1])))
                signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        return signals
    
    sweep = bt.parameter_sweep(
        signals_fn=gated_signals_fn,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    
    summary = bt.sweep_summary(sweep, baseline=(("vol_lookback", VOL_LOOKBACK),))
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


def main():
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
    
    print("\n=== 3. Volatility-gate momentum on REGIME_DEPENDENT assets ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        bt.check_signal_integrity(momentum_with_vol_gate(closes, VOL_LOOKBACK), [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), momentum_with_vol_gate(closes, VOL_LOOKBACK), bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, volatility-gated momentum signals {len(momentum_with_vol_gate(closes, VOL_LOOKBACK))}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: volatility gate at bar t uses closes[:t] only; "
          "neutral bars before the vol window carry the equity forward.")
    
    print("\n=== 4. Volatility-gated momentum across collected universe ===")
    print("Momentum with volatility gate (threshold={}, lookback={}d): long previous {}-day return, hold 1 day, daily "
          "rebalanced, trade only when past 60-bar realized vol > {} (annualized). "
          "Segments: {} contiguous blocks by date, past-only volatility classifier "
          "(turbulent = block median trailing-{}-bar realized vol > series-wide median).\n"
          .format(VOL_THRESHOLD, VOL_LOOKBACK, MOMENTUM_LOOKBACK, VOL_THRESHOLD, WINDOW, N_BLOCKS))
    
    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:14s} {:28s} {:28s} {:12s}  {:12s}  verdict".format(
            "variant", "segments", "candidate_medians", "cand_disp", "null_disp"))
        for name, r in res.items():
            if isinstance(r, dict) and "segments" in r:
                segs = "[" + ",".join(r["segments"]) + "]"
                meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
                print(
                    "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                        name, segs, meds, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
            else:
                print(f"  ERROR: r is not a dict or missing segments: {type(r)}")
        print("")
    
    # Full-universe generalization: volatility-gated momentum on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: momentum_with_vol_gate(c, VOL_LOOKBACK),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("vol_lookback", VOL_LOOKBACK), ("threshold", VOL_THRESHOLD)),
        param_grid=[{"vol_lookback": VOL_LOOKBACK, "threshold": VOL_THRESHOLD}],
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
    print("=== 6. Synthetic perturbation sweep (vol_lookback 30 / 60 / 120) ===")
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (vol_lookback=60): {:+.3f}".format(pers["baseline_median"]))
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
            if isinstance(r, dict) and "segments" in r:
                segs = "[" + ",".join(r["segments"]) + "]"
                meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
                out2_lines.append(
                    "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                        name, segs, meds, r["candidate_dispersion"],
                        r["null_dispersion"], r["verdict"]))
            else:
                out2_lines.append(f"  ERROR: r is not a dict or missing segments: {type(r)}")
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"
    
    # Write the artifact that the independent verification file recomputes
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        vol_lookback=VOL_LOOKBACK,
        vol_threshold=VOL_THRESHOLD,
        momentum_lookback=MOMENTUM_LOOKBACK,
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
    print("NOTE: research/simulation only. Volatility-gated momentum test: "
          "momentum with predeclared past-only volatility gate (threshold={}). "
          "Momentum lookback: {}d, vol lookback: {}d. Segments: {} contiguous blocks "
          "by date, past-only volatility classifier."
          .format(VOL_THRESHOLD, MOMENTUM_LOOKBACK, VOL_LOOKBACK, N_BLOCKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
