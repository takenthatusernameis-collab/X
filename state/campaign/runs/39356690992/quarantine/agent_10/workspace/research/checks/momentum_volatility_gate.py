"""Past-only volatility-gated lookback-5 momentum test.

Objective: Test one explicitly predeclared past-only volatility gate on
lookback-5 momentum through the existing regime-stability and null framework.

The gate: long momentum only when past realized volatility < threshold.
This tests whether conditioning momentum on low-volatility regimes creates
predeclared robustness improvement.

Evidence basis: MA regime filters were previously falsified; whether the
same class of conditioning helps the different momentum mechanism remains
resolved.

Method (fixed a-priori, not tuned to OOS): 
- momentum_signals(closes, lookback=5) -> base momentum signals
- filter_signals(closes, vol_threshold) -> apply volatility gate
- run regime-stability stress testing across 4 volatility blocks
- compare against matched null benchmark
- use the same walk-forward settings as momentum_lookback_sweep
- test all 10 assets in the collected universe
- one threshold rule: vol_threshold=0.15 (annualized)
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
VOL_THRESHOLD = 0.15  # predeclared volatility gate threshold (annualized)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"


def volatility_gated_momentum_signals(closes: np.ndarray, vol_threshold: float, lookback: int = LOOKBACK):
    """Past-only momentum signals gated by past realized volatility.
    
    Long momentum only when past realized volatility < vol_threshold.
    This is a past-only gate: volatility computed from closes[:i] only.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    
    for i in range(lookback - 1, n):
        # Compute momentum signal
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        momentum_weight = 1.0 if ret > 0 else -1.0
        
        # Compute past realized volatility (annualized)
        if i >= WINDOW:
            log_returns = np.log(closes[i - WINDOW : i])
            realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
            vol_gate = realized_annual < vol_threshold
        else:
            vol_gate = True  # insufficient data, let momentum through
        
        if vol_gate:
            signals[i] = bt.Signal(date=i + 1, weight=momentum_weight)
    
    return signals


def run_asset(asset, tickers, vol_threshold):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Get volatility-gated momentum signals
    signals = volatility_gated_momentum_signals(closes, vol_threshold, LOOKBACK)
    
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=lambda c, **kw: volatility_gated_momentum_signals(c, vol_threshold, LOOKBACK),
        bars=list(bars),
        signals=signals,
        param_grid=[{"vol_threshold": vol_threshold}],
        baseline=(("vol_threshold", vol_threshold),),
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
        edge_status=res.n_edge_present,
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
        signals = volatility_gated_momentum_signals(closes, VOL_THRESHOLD, LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, gated momentum signals {len(signals)}, "
              f"vol_threshold={VOL_THRESHOLD}, leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: volatility gate and momentum sign at bar t use closes[:t] only.")
    
    print("\n=== 4. Volatility-gated momentum across collected universe ===")
    print(f"  Predeclared gate: vol_threshold={VOL_THRESHOLD} (annualized realized vol)")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} "
          f"bars/segment, 4 contiguous volatility blocks, same coin-flip null.\n")
    
    # Load all tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    
    # Test all 10 assets
    results_all = {}
    verdict_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
                     "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
    
    ticker_order = list(tickers_univ.keys())
    
    for asset in ticker_order:
        res = run_asset(asset, tickers_univ, VOL_THRESHOLD)
        results_all[asset] = res
        verdict_counts[res["verdict"]] += 1
        
        segs = "[" + ",".join(res["segments"]) + "]"
        meds = "[" + ",".join(str(m) for m in res["medians"]) + "]"
        print(f"  {asset:6s}: segments={segs} medians={meds} "
              f"verdict={res['verdict']:20s} edges={res['edge_status']}")
    
    print("\n=== 5. Summary ===")
    print(f"  Verdict counts across universe:")
    for verdict, count in verdict_counts.items():
        print(f"    {verdict}: {count}")
    print(f"  Volatility gate threshold: {VOL_THRESHOLD} (annualized)")
    print(f"  Gate predeclared before execution (not tuned to OOS results).")
    
    # Determinism: rerun the 2-asset comparison
    print("\n=== 6. Determinism check ===")
    # Rerun only the target assets to verify determinism
    tickers_target = {t: tickers_univ[t] for t in target}
    res2 = {a: run_asset(a, tickers_target, VOL_THRESHOLD) for a in target}
    # Extract only the target assets from the full results for comparison
    full_target_results = {a: results_all[a] for a in target}
    out1 = json.dumps(full_target_results, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"
    
    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        vol_threshold=VOL_THRESHOLD,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=results_all,
        universe=dict(
            verdict_counts=verdict_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    segments=results_all[t]["segments"],
                    medians=results_all[t]["medians"],
                    candidate_dispersion=results_all[t]["candidate_dispersion"],
                    null_dispersion=results_all[t]["null_dispersion"],
                    verdict=results_all[t]["verdict"],
                    n_folds=results_all[t]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Volatility-gated momentum test on 10 assets.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
