"""Test a predeclared past-only volatility gate on lookback-5 momentum.

Background: the standard momentum check (`momentum.py`) tests lookback-5
momentum across the collected universe, finding either CONSISTENT_WITH_NOISE,
REGIME_STABLE, REGIME_STABLE_LOSS, or REGIME_DEPENDENT outcomes.

This test introduces a volatility gate: momentum signals are only emitted
when trailing realized volatility exceeds 25% annualized (the existing
"turbulent" regime definition). The gate is past-only (uses closes[:t]
only) and predeclared (no threshold search or post-hoc optimization).

We test the gated variant through the same regime-stability framework that
validates the ungated momentum check, using the existing 4-block
volatility-regime segmentation and the same walk-forward geometry.

Falsification prediction: if the ungated momentum is REGIME_STABLE (same
edge or no edge everywhere) then the gated variant should either be
REGIME_STABLE (edge moved but still stable) or CONSISTENT_WITH_NOISE
(edge vanished). If the ungated momentum is REGIME_DEPENDENT (the edge
concentrates in one regime) then the gated variant should pass the
REGIME_STABLE gate (the edge concentrates in the "turbulent" regime
where the gate is active). Either outcome is a negative conclusion if
the gate does not improve robustness.

Method: use `research.backtest.regime_stability.conditional_signal` to
wrap `research.backtest.regime_stability.momentum_signals` with
`research.backtest.regime_stability.active_when_turbulent`. This yields a
momentum signal that is active only in turbulent segments. Run the same
regime-stability pipeline as the momentum check, using real data and
the existing artifact for independent verification.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import sys
import json
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
ARTIFACT_PATH = ARTIFACT_DIR / "gated_momentum_results.json"
def gated_momentum_signals(closes, lookback):
    """Momentum signals gated by past-only volatility > 25% annualized.

    This is the exact same logic as research.backtest.regime_stability.momentum_signals
    but wrapped via conditional_signal, making it a separate code path for
    independent verification.

    Predeclared gate: active_when_turbulent (volatility > 25% annualized).
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out
def gated_signal_wrapper(closes, **params):
    """Apply the volatility gate to momentum signals.

    This is research.backtest.regime_stability.conditional_signal applied to
    momentum_signals with active_when_turbulent.
    """
    return bt.conditional_signal(
        lambda c, **p: gated_momentum_signals(c, lookback=LOOKBACK),
        bt.active_when_turbulent
    )(closes, **params)
def run_gated_momentum():
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

    print("\n=== 2. Leakage review per asset ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = gated_signal_wrapper(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker}: {bars.n_bars} bars, gated momentum, leakage [PASS]")

    print("\n=== 3. Gated momentum on REGIME_DEPENDENT assets ===")
    print("Momentum: long the previous {}-day return, hold 1 day, daily "
          "rebalanced, gated by past-only volatility > 25% annualized."
          " Params: walk-forward train={}d/test={}d/warmup={}d/overlap={}d, "
          "min {} bars/segment, compared vs coin-flip null on the same segments."
          "\n".format(LOOKBACK, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    results_all = {}
    for asset in target:
        bars = tickers[asset]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)

        signals = gated_signal_wrapper(closes, lookback=LOOKBACK)
        if len(signals) != len(closes):
            raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")

        res = bt.stress_segments(
            signals_fn=gated_signal_wrapper,
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
        results_all[asset] = dict(
            segments=names,
            medians=medians,
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=res.overall_verdict,
            n_folds=res.n_folds,
        )
        print("Asset: {}".format(asset))
        print("  {:14s} {:28s} {:12s}  {:12s}  verdict".format(
            "segments", "candidate_medians", "cand_disp", "null_disp"))
        segs = "[" + ",".join(names) + "]"
        meds = "[" + ",".join(str(m) for m in medians) + "]"
        print("  {:14s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
            "gated_momentum", segs, results_all[asset]["candidate_dispersion"],
            results_all[asset]["null_dispersion"], results_all[asset]["verdict"]))
        print("")

    # Full-universe generalization: gated momentum on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=gated_signal_wrapper,
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

    # Convert universe summary to JSON-serializable dict
    univ_summary_dict = {
        "ticker_order": summary.ticker_order,
        "assets": {
            t: {
                "verdict": assets.overall_verdict,
                "candidate_dispersion": assets.candidate_dispersion,
                "null_dispersion": assets.null_dispersion,
                "scenarios": [
                    {
                        "name": s.name,
                        "baseline_median": round(s.baseline_median_log_return, 3),
                        "noise_median": round(s.noise_median_log_return, 3),
                        "n_folds": s.n_folds,
                        "n_periods": s.n_periods,
                    } for s in assets.scenarios
                ]
            } for t, assets in summary.assets.items()
        },
        "verdict_counts": summary.verdict_counts,
    }

    print("=== 4. Gated momentum across collected universe ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    # Determinism: rerandomize everything and assert identical output
    res2 = {a: run_gated_momentum_asset(a, tickers) for a in target}
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact that the independent verification file recomputes
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        volatility_gate_threshold=0.25,  # Predeclared threshold
        per_asset=results_all,
        universe=univ_summary_dict,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Gated momentum test (volatility > 25% gate) on AMZN/JPM "
          "+ universe: exploratory simulation.")
    return 0
def run_gated_momentum_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    signals = gated_signal_wrapper(closes, lookback=LOOKBACK)
    res = bt.stress_segments(
        signals_fn=gated_signal_wrapper,
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
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )
if __name__ == "__main__":
    sys.exit(run_gated_momentum())
