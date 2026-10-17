"""Test a past-only volatility gate on lookback-5 momentum.

Background: MA crossover momentum conditioning was previously falsified for
trend-following signals. Whether the same class of conditioning (past-only
volatility gate) helps the different momentum mechanism (long the previous 5-day
return) remains unresolved.

Hypothesis: Momentum may be REGIME_DEPENDENT (edge concentrated in one
regime mix), CONSISTENT_WITH_NOISE (no edge anywhere), or REGIME_STABLE_LOSS
(uniformly negative). A past-only volatility gate could either demonstrate a
predeclared robustness improvement (momentum becomes REGIME_STABLE when gated)
or be falsified/unsupported.

Method (fixed a-priori, not tuned to OOS):
- Past-only volatility gate: only emit momentum signals when realized vol < threshold
- One threshold rule declared before execution (threshold = 0.20 annualized)
- Existing 10-asset universe (collected from Yahoo Finance)
- Same walk-forward settings as base MA crossover (train=252d/test=84d/warmup=60d/overlap=60d)
- Compare gated vs ungated momentum through the same regime-stability pipeline
- Determinism asserted via internal independent recomputation

Research / simulation only. No live trading.
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
VOL_THRESHOLD = 0.20  # Predeclared threshold: low vol only (<20% annualized)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "volatility_gated_momentum_results.json"


def volatility_gate_signals(closes, lookback, vol_threshold):
    """Past-only momentum signals gated by past-only realized volatility.
    
    Long the previous lookback-day return only when recent annualized realized
    volatility < vol_threshold. This is the predeclared past-only volatility gate.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    # Past-only volatility calculation: use closes[:i] only (no lookahead)
    for i in range(max(lookback, WINDOW) - 1, n):
        # Calculate momentum signal
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        momentum_weight = 1.0 if ret > 0 else -1.0
        
        # Calculate past-only realized volatility
        if i >= WINDOW:
            log_returns = np.log(closes[i - WINDOW : i])
            realized_annual = float(np.std(log_returns, ddof=1)) * np.sqrt(252.0)
            # Apply gate: only trade in low volatility regimes
            if realized_annual < vol_threshold:
                signals[i] = bt.Signal(date=i + 1, weight=momentum_weight)
    
    return signals


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}
    
    for name, sig, params in [
        ("ungated_momentum", lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK), {"lookback": LOOKBACK}),
        ("volatility_gated_momentum", lambda c, **p: volatility_gate_signals(c, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD), {"lookback": LOOKBACK, "vol_threshold": VOL_THRESHOLD}),
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
        signals = bt.momentum_signals(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS]")
    
    print("\n=== 4. Volatility-gated momentum vs ungated momentum (REGIME_DEPENDENT assets) ===")
    print(f"Volatility gate: past-only annualized realized vol < {VOL_THRESHOLD} (low volatility regimes only)")
    print(f"Momentum: long the previous {LOOKBACK}-day return, hold 1 day, daily rebalanced")
    print(f"Walk-forward settings: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment")
    print(f"Segments: {N_BLOCKS} contiguous blocks by date, past-only volatility classifier")
    
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
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: volatility_gate_signals(c, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK), ("vol_threshold", VOL_THRESHOLD)),
        param_grid=[{"lookback": LOOKBACK, "vol_threshold": VOL_THRESHOLD}],
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
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Volatility-gated momentum test on AMZN/JPM + universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())