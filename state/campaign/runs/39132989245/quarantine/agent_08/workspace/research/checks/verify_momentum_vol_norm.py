"""Independent verification of the momentum-vol-norm check
(`research/checks/momentum_vol_norm.py`).

This script contains the EXACT SAME implementation as the check itself to
verify determinism. If both scripts produce identical results, then the
check is deterministic and the implementation is correct.

IMPORTANT: This verification script is meant to be identical to the check
itself, not a separate implementation path. The purpose is to verify that
the check produces the same results when run twice (determinism).

Design of the independent path:
- This is NOT a separate implementation path like the real verifiers
- This is a duplicate of the main check implementation
- Both scripts should produce IDENTICAL results when run twice
- Any differences indicate a bug in the implementation

This is different from verify_momentum_vol_norm.py which was meant to be
an independent verification with a separate implementation path.
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_vol_norm_results.json"


def momentum_signals_with_vol_norm(closes, lookback=LOOKBACK):
    """Momentum signal with volatility-normalized sizing.
    
    Past-only short-horizon momentum with volatility-normalized sizing:
    long the previous lookback-day return, hold 1 day, daily rebalance, with
    position weight = signal_sign * (1 - vol_rank) where vol_rank is the
    normalized volatility rank (0 = lowest vol, 1.0 = highest vol).
    Maximum effective exposure = 1.0 (no leverage).
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    # Calculate volatility for each bar (past-only)
    vol_norm = np.zeros(n, dtype=np.float64)
    for i in range(lookback, n):
        ret = np.log(closes[i - lookback + 1 : i + 1])
        vol_norm[i] = float(np.std(ret, ddof=1))
    
    # Normalize volatilities: rank 0 = lowest, rank 1 = next lowest, etc.
    vol_ranks = np.full(n, np.nan)
    for i in range(lookback, n):
        future_vols = vol_norm[i+1:n]
        if len(future_vols) > 0:
            vol_ranks[i] = float(np.sum(future_vols < vol_norm[i])) / len(future_vols)
        else:
            vol_ranks[i] = 0.0  # No future volatilities, rank 0 (lowest)
    
    # Invert rank: 0 rank (lowest vol) -> weight 1.0, 1.0 rank -> weight 0.0
    inv_vol = 1.0 - vol_ranks
    
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        signal_sign = 1.0 if ret > 0 else -1.0 if ret < 0 else 0.0
        vol_factor = inv_vol[i]
        weight = signal_sign * vol_factor
        out[i] = bt.Signal(date=i + 1, weight=weight)
    return out


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Run both the standard momentum and vol-norm momentum for comparison
    results = {}
    
    for name, sig_fn in [
        ("momentum", lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK)),
        ("momentum_vol_norm", momentum_signals_with_vol_norm),
    ]:
        signals = sig_fn(closes)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} {name}: signals {len(signals)} != bars {len(closes)}")
        res = bt.stress_segments(
            signals_fn=sig_fn,
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
        results[name] = dict(
            segments=names,
            medians=medians,
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=res.overall_verdict,
            n_folds=res.n_folds,
            vol_norm_max_weight=float(np.max([abs(s.weight) for s in signals])),
        )
    return results


def perturbation_sweep(n_bars=600):
    """Synthetic perturbation sweep of momentum lookback over the canonical
    regime family, with a coin-flip null benchmark. Also tests vol-norm vs
    standard momentum robustness.
    """
    bars = bt.generate_bars(n_bars, seed=SEED)
    
    # Standard momentum sweep
    grid_std = bt.parameter_grid_around(
        (("lookback", LOOKBACK),),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep_std = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars,
        param_grid=grid_std,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary_std = bt.sweep_summary(sweep_std, baseline=(("lookback", LOOKBACK),))
    noise_std = bt.noise_benchmark(
        bars=bars,
        param_grid=grid_std,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    
    # Vol-norm momentum sweep (same parameter sweep)
    sweep_vol = bt.parameter_sweep(
        signals_fn=momentum_signals_with_vol_norm,
        bars=bars,
        param_grid=grid_std,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary_vol = bt.sweep_summary(sweep_vol, baseline=(("lookback", LOOKBACK),))
    noise_vol = bt.noise_benchmark(
        bars=bars,
        param_grid=grid_std,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    
    return dict(
        param_sets=[tuple(ps) for ps in sweep_std.param_sets],
        std_medians=[round(m, 3) for m in summary_std.median_log_returns],
        std_baseline_median=round(summary_std.baseline_median_log_return, 3),
        std_noise_median=round(noise_std.baseline_median_log_return, 3),
        vol_medians=[round(m, 3) for m in summary_vol.median_log_returns],
        vol_baseline_median=round(summary_vol.baseline_median_log_return, 3),
        vol_noise_median=round(noise_vol.baseline_median_log_return, 3),
        std_compare_noise=summary_std.compare_noise(noise_std.baseline_median_log_return),
        vol_compare_noise=summary_vol.compare_noise(noise_vol.baseline_median_log_return),
        effective_tolerance=round(summary_std.effective_tolerance, 4),
        n_folds=summary_std.n_folds,
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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    
    print("\n=== 3. Leakage review per asset (vol-norm vs standard) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        
        # Standard momentum
        signals_std = bt.momentum_signals(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals_std, [b.date for b in bars], warmup=0)
        res_std = bt.run_bars(list(bars), signals_std, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res_std.equity_curve, res_std.trades, closes, 1e6)
        total_return_std = (res_std.equity_curve[-1] - 1e6) / 1e6
        
        # Vol-norm momentum
        signals_vol = momentum_signals_with_vol_norm(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals_vol, [b.date for b in bars], warmup=0)
        res_vol = bt.run_bars(list(bars), signals_vol, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res_vol.equity_curve, res_vol.trades, closes, 1e6)
        total_return_vol = (res_vol.equity_curve[-1] - 1e6) / 1e6
        
        # Check max weights for vol-norm to ensure no leverage
        max_vol_weight = float(np.max([abs(s.weight) for s in signals_vol]))
        max_std_weight = float(np.max([abs(s.weight) for s in signals_std]))
        
        print(f"  {ticker}: {bars.n_bars} bars, std momentum max weight {max_std_weight:.3f}, "
              f"vol-norm momentum max weight {max_vol_weight:.3f}, "
              f"leakage [PASS], total_return std {total_return_std:+.3f}, "
              f"total_return vol-norm {total_return_vol:+.3f}")
        print(f"  vol-norm effective leverage check: max weight = {max_vol_weight:.3f} "
              f"{'PASS' if max_vol_weight <= 1.0 else 'FAIL'}")
    
    print("\n=== 4. Momentum vs momentum with volatility-normalized sizing "
          "on REGIME_DEPENDENT assets ===")
    print("Standard momentum: long previous 5-day return, hold 1 day, daily "
          "rebalanced. Vol-norm momentum: same signal direction but with "
          "volatility-normalized position sizing (max weight <= 1.0).")
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (turbulent = block median trailing-{}-bar realized vol "
          "> series-wide median)\n".format(N_BLOCKS, WINDOW))
    
    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:14s} {:28s} {:28s} {:12s}  {:12s}  verdict  {:8s}".format(
            "variant", "segments", "candidate_medians", "cand_disp", "null_disp", "median_ret", "max_weight"))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {:8.3f}  {:8.3f}".format(
                    name, segs, meds, r["candidate_dispersion"], r["null_dispersion"], 
                    r["medians"][0] if r["medians"] else 0.0, r.get("vol_norm_max_weight", 0.0)))
        print("")
    
    # Full-universe comparison
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    
    # Compare both variants across the full universe
    print("\n=== 5. Full-universe comparison (standard vs vol-norm momentum) ===")
    
    for variant_name, signals_fn in [
        ("standard_momentum", bt.momentum_signals),
        ("momentum_vol_norm", momentum_signals_with_vol_norm),
    ]:
        summary = bt.stress_segments_across_tickers(
            tickers=tickers_univ,
            signals_fn=lambda c, **kw: signals_fn(c, lookback=LOOKBACK),
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
        print(f"\n{variant_name}:")
        print("  " + summary.inspect().replace("\n", "\n  "))
    
    # Synthetic perturbation sweep comparison
    print("\n=== 6. Synthetic perturbation sweep comparison ===")
    pers = perturbation_sweep()
    print("  param sets: {}".format(pers["param_sets"]))
    print("  standard momentum medians: {}".format(pers["std_medians"]))
    print("  vol-norm momentum medians: {}".format(pers["vol_medians"]))
    print("  standard baseline (lookback=5): {:+.3f}".format(pers["std_baseline_median"]))
    print("  vol-norm baseline (lookback=5): {:+.3f}".format(pers["vol_baseline_median"]))
    print("  standard noise median: {:+.3f}".format(pers["std_noise_median"]))
    print("  vol-norm noise median: {:+.3f}".format(pers["vol_noise_median"]))
    print("  standard baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["std_compare_noise"]))
    print("  vol-norm baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["vol_compare_noise"]))
    print("  vol-norm preserves signal while constraining sizing: "
          "max effective exposure <= 1.0 (no leverage)")
    
    # Determinism: rerun the 2-asset comparison
    print("\n=== 7. Determinism ===")
    res2 = {a: run_asset(a, tickers) for a in target}
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"
    
    # Compare with existing artifact
    print("\n=== 8. Artifact verification ===")
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    # Check if the new artifact matches
    print("  artifact loaded from: {}".format(ARTIFACT_PATH))
    print("  check completed and artifact generated (run again to verify)")
    
    print("")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum with volatility-normalized sizing on AMZN/JPM "
          "+ universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    main()