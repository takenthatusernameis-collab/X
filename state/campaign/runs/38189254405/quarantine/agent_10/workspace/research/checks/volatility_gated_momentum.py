"""Volatility-gated momentum test for Agent 10's R-003 task.

Background: the qualified momentum effect (lookback=5, hold=1) shows mixed
REGIME_STABLE/REGIME_DEPENDENT evidence across the collected 10-asset
universe. MA-based volatility regime filters were previously falsified on the
MA crossover class; whether a different type of volatility conditioning helps
the momentum mechanism remains unresolved.

Falsification prediction: a single past-only volatility gate will either
demonstrate a reproducible improvement in robustness (REGIME_STABLE positive
edges where the ungated momentum was REGIME_DEPENDENT or CONSISTENT_WITH_NOISE)
or be falsified/unsupported (no meaningful improvement).

Method (fixed a-priori, not tuned to OOS): long the previous 5-day return,
hold 1 day, daily rebalance, but only when past realized volatility is
below a predeclared threshold (v_threshold). The gate is applied to the
momentum signal: if volatility at bar i (using trailing vol_window bars) >=
v_threshold, signal weight = 0; otherwise, weight = momentum signal.

The same walk-forward windows, 4 contiguous volatility blocks, min_segment_bars
threshold, and coin-flip null as the existing momentum check
(`research/checks/momentum.py`) are used; the only change is the volatility
gate wrapper on the momentum signal.

A-priori falsification prediction: the gated variant either shows
REGIME_DEPENDENT or CONSISTENT_WITH_NOISE verdicts (no improvement), or
REGIME_STABLE_LOSS (negative stability), which closes the cell.

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
V_THRESHOLD = 0.20  # predeclared volatility threshold (annualized realized vol)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "volatility_gated_momentum_results.json"


def momentum_signals(closes, lookback):
    """Pure momentum signal - long if previous lookback days were up."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def volatility_gate_momentum_signals(closes, lookback, v_threshold, vol_window):
    """Momentum signals gated by past-only volatility.
    
    Only emit momentum signals when recent realized volatility is below v_threshold.
    Past-only: volatility at bar i uses closes[:i] only.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    
    # First compute base momentum signals
    momentum_sig = momentum_signals(closes, lookback)
    
    # Apply volatility gate
    for i in range(vol_window - 1, n):
        # Calculate realized volatility over trailing vol_window bars
        log_returns = np.log(closes[i - vol_window + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        
        if realized_annual < v_threshold:
            # Volatility below threshold: use momentum signal
            out[i] = momentum_sig[i]
        # else: weight remains 0 (neutral)
    
    # Handle bars before vol_window (insufficient data)
    for i in range(vol_window - 1):
        out[i] = momentum_sig[i]
    
    return out


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}

    for name, sig_fn, params in [
        ("momentum", lambda c, **p: momentum_signals(c, LOOKBACK), {"lookback": LOOKBACK}),
        ("vol_gated", lambda c, **p: volatility_gate_momentum_signals(c, LOOKBACK, V_THRESHOLD, WINDOW), {"v_threshold": V_THRESHOLD, "vol_window": WINDOW}),
    ]:
        signals = sig_fn(closes)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} {name}: signals {len(signals)} != bars {len(closes)}")
        
        # For volatility gated, we need to ensure signals match asset data
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
        signals = volatility_gate_momentum_signals(closes, LOOKBACK, V_THRESHOLD, WINDOW)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, vol-gated momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: volatility gate at bar t uses closes[:t] only; "
          "neutral bars before the vol_window carry the equity forward.")

    print("\n=== 4. Volatility-gated momentum vs base momentum on REGIME_DEPENDENT assets ===")
    print("Vol-gated momentum: long the previous {}-day return, hold 1 day, daily "
          "rebalanced, but only when trailing realized volatility < {} (annualized). Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, compared vs coin-flip null on the same segments\n".format(
              LOOKBACK, V_THRESHOLD, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
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

    # Full-universe generalization
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: volatility_gate_momentum_signals(c, LOOKBACK, V_THRESHOLD, WINDOW),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("v_threshold", V_THRESHOLD), ("vol_window", WINDOW)),
        param_grid=[{"v_threshold": V_THRESHOLD, "vol_window": WINDOW}],
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
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        v_threshold=V_THRESHOLD,
        vol_window=WINDOW,
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
          "execution. Volatility-gated momentum test on AMZN/JPM + universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
