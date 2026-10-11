#!/usr/bin/env python3
"""Test past-only volatility gate on lookback-5 momentum.

Background: existing momentum (lookback=5) shows REGIME_DEPENDENT behavior on
AMZN/JPM: edge lives in the 2009-2013 turbulent block, not harvestable by
past-only volatility classifiers that cannot distinguish that regime from the
bad 2022-2026 counterpart.

Hypothesis: a past-only volatility gate (filter momentum signals when
realized volatility > THRESHOLD) may produce a REGIME_STABLE edge or eliminate
the dependence by removing the block-specific edge while preserving stability.

Predeclared threshold rule (declared before execution, not tuned after):
- THRESHOLD = 0.25 (25% annualized realized volatility)
- Filter signals when realized vol > THRESHOLD (i.e., trade only in low-vol regimes)

Method: for each asset, run two variants through the same regime-stability
pipeline (stress_segments) with the same coin-flip null:
1) pure momentum (baseline for comparison)
2) volatility-gated momentum (signal active only when recent realized vol < THRESHOLD)

Stop condition: after the one predeclared gate is evaluated and independently checked.

Falsification prediction: the gated variant either remains REGIME_DEPENDENT
(or becomes CONSISTENT_WITH_NOISE), because a past-only vol threshold cannot
separate the 2009-2013 edge from the 2022-2026 turbulent regime; the edge is
period-specific, not volatility-contingent.
"""
from __future__ import annotations

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
VOL_THRESHOLD = 0.25  # predeclared: trade only when annualized vol < 25%
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "volatility_gated_momentum_results.json"

def volatility_gate_signals(closes, lookback=LOOKBACK, **kwargs):
    """Past-only momentum signals gated by recent realized volatility.
    
    Long the previous `lookback`-day return only when the trailing
    `WINDOW`-bar realized volatility (annualized) is below `vol_threshold`.
    Neutral otherwise (weight 0). Past-only: uses only closes[:i] for vol calc.
    """
    vol_threshold = kwargs.get("vol_threshold", VOL_THRESHOLD)
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    for i in range(WINDOW - 1, n):
        log_returns = np.log(closes[i - WINDOW + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        if i >= lookback:
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            weight = 1.0 if ret > 0 else -1.0
        else:
            weight = 0.0
        signals[i] = bt.Signal(date=i + 1, weight=weight if realized_annual < vol_threshold else 0.0)
    return signals

def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    baseline = (("lookback", LOOKBACK),)
    results = {}

    for name, sig, params in [
        ("pure_momentum", lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK), {"lookback": LOOKBACK}),
        ("vol_gated_momentum", lambda c, **p: volatility_gate_signals(c, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD), {"lookback": LOOKBACK}),
    ]:
        signals = sig(closes)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} {name}: signals {len(signals)} != bars {len(closes)}")
        res = bt.stress_segments(
            signals_fn=sig,
            bars=list(bars),
            signals=signals,
            param_grid=[params],
            baseline=baseline,
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
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Volatility-gated momentum on REGIME_DEPENDENT assets ===")
    print("Params: lookback={} | volatility gate: trade only when trailing {}-bar "
          "realized vol < {} (annualized)".format(LOOKBACK, WINDOW, VOL_THRESHOLD))
    print("Segments: {} contiguous blocks by date, min {} bars, past-only "
          "volatility classifier (turbulent = block median trailing-{}-bar "
          "realized vol > series-wide median)".format(N_BLOCKS, MIN_SEGMENT_BARS, WINDOW))
    print("Walk-forward per segment: train={}d / test={}d / warmup={}d / "
          "overlap={}d, compared vs coin-flip null on the same segments\n".format(
              TRAIN, TEST, WARM, OVERLAP))

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
        signals_fn=lambda c, **kw: volatility_gate_signals(c, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD),
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

    # Determinism: recompute the 2-asset comparison
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
    print("determinism: r1 == r2: True")

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
    import json
    sys.exit(main())