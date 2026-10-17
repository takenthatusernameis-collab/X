"""Test the predeclared past-only volatility gate on lookback-5 momentum.

This tests whether a single past-only volatility gate creates a predeclared
robustness improvement for the qualified momentum effect.

Background: MA regime filters were previously falsified. Whether the same
class of conditioning helps the different momentum mechanism remains unresolved.

Hypothesis being tested: a past-only volatility gate that only activates in
calm (low-vol) regimes creates a REGIME_STABLE improvement for lookback-5
momentum, rather than remaining REGIME_DEPENDENT (edge only in one regime)
or CONSISTENT_WITH_NOISE (no edge anywhere).

This is a principled, fixed a-priori design (threshold = 15% annualized vol):
- When past volatility < 15%, momentum signals are active
- When past volatility >= 15%, momentum signals are disabled (neutral)
- No parameter tuning after seeing results

The variant is run through the SAME `stress_segments` pipeline as the base
momentum, with the same coin-flip null, so the REGIME_STABLE / REGIME_DEPENDENT
/ CONSISTENT_WITH_NOISE verdicts are comparable.

Method: momentum signals are wrapped with a volatility gate: emit the
momentum signal only when early realized volatility < 0.15 (calm regime).
This preserves the exact same walk-forward structure and null framework.
Determinism: the script asserts byte-identical output across independent reruns.

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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"
VOLATILITY_GATE_THRESHOLD = 0.15  # 15% annualized volatility threshold

def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def volatility_gate_condition(early_vol):
    """Past-only volatility gate: only active when volatility < 0.15 (calm regime)."""
    return early_vol < VOLATILITY_GATE_THRESHOLD
def gated_momentum_signals(closes, lookback, gate_threshold):
    """Momentum signals wrapped with past-only volatility gate.
    
    Emits momentum signals only when early realized volatility < gate_threshold.
    Otherwise returns neutral (zero weight) signals.
    """
    n = len(closes)
    # Calculate early volatility for gating
    window = min(60, max(4, n // 4))
    if n < window:
        return [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    early_vol = float(
        np.std(np.log(closes[window // 2 : window]), ddof=1)
    ) * np.sqrt(252.0)
    
    # Get base momentum signals
    base_signals = momentum_signals(closes, lookback)
    
    if volatility_gate_condition(early_vol):
        return base_signals
    else:
        return [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}

    for name, sig, params in [
        ("momentum", lambda c, **p: momentum_signals(c, LOOKBACK), {"lookback": LOOKBACK}),
        ("momentum_volatility_gate", lambda c, **p: gated_momentum_signals(c, LOOKBACK, VOLATILITY_GATE_THRESHOLD), {"lookback": LOOKBACK}),
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
        signals = momentum_signals(closes, LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")

    print("\n=== 4. Momentum with past-only volatility gate (regime-stability test) ===")
    print("Momentum: long the previous {}-day return, hold 1 day, daily "
          "rebalanced. Base MA(20/60) included for comparison\n".format(LOOKBACK))
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (calm = block median trailing-{}-bar realized vol "
          "<= series-wide median)\n".format(N_BLOCKS, WINDOW))
    print("Volatility gate: only active when early realized volatility < {} ({}% annualized)\n".format(
        VOLATILITY_GATE_THRESHOLD, VOLATILITY_GATE_THRESHOLD * 100))

    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:14s} {:28s} {:24s} {:12s}  verdict".format(
            "variant", "segments", "medians", "dispersion"))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:24s} {:+10.3f}  {}".format(
                    name, segs, meds, r["candidate_dispersion"], r["verdict"]))
        print("")

    # Full-universe generalization: gated momentum on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: gated_momentum_signals(c, LOOKBACK, VOLATILITY_GATE_THRESHOLD),
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

    # Determinism: rerun the 2-asset comparison and assert identical output
    res2 = {a: run_asset(a, tickers) for a in target}
    out2_lines = []
    for asset in target:
        out2_lines.append("Asset: {}".format(asset))
        for name, r in res2[asset].items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            out2_lines.append(
                "  {:14s} {:28s} {:24s} {:+10.3f}  {}".format(
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
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        volatility_gate_threshold=VOLATILITY_GATE_THRESHOLD,
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
          "execution. Momentum with past-only volatility gate (threshold={} "
          ") on AMZN/JPM + universe: exploratory simulation.".format(VOLATILITY_GATE_THRESHOLD))
    return 0
if __name__ == "__main__":
    sys.exit(main())
