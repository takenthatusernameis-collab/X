"""Test predeclared volatility gate on lookback-5 momentum signals through regime-stability.

This script tests the explicitly predeclared past-only volatility gate on momentum
lookback-5 signals through the existing regime-stability and null framework.

Evidence basis: MA regime filters were previously falsified; this tests whether
the same class of conditioning helps the different momentum mechanism.

Predeclared gate: momentum signals only emitted when past-only realized volatility
is below 20% annualized (vol_window=20 bars). This is declared before execution
and not tuned after seeing results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.checks.volatility_gate_momentum import (
    volatility_gate_momentum_signals,
    predeclare_volatility_gate,
    momentum_signals,
)
from tests.test_loader import load_manifest
from research.data.preflight import sha256_file

# Predeclared gate parameters (fixed a-priori)
GATE_PARAMS = predeclare_volatility_gate()

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TARGET = ["AMZN", "JPM"]
FAST = 20
SLOW = 60
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
OUTLIER_TOL = 0.05

# Gate-specific parameters for the test
VOL_THRESHOLD = GATE_PARAMS["vol_threshold"]  # 0.20
VOL_WINDOW = GATE_PARAMS["vol_window"]       # 20


def run_regime_stability_test(ticker, bars, signals, gate_name):
    """Run regime-stability stress test on a ticker.

    Args:
        ticker: ticker symbol
        bars: BarSequence for the ticker
        signals: signal series (already filtered by gate if applicable)
        gate_name: descriptive name of the gate used

    Returns:
        dict with test results
    """
    print(f"--- {ticker} ({gate_name}) ---")

    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    baseline = (("fast", FAST), ("slow", SLOW))
    grid = [{"fast": FAST, "slow": SLOW}]

    # Create wrapper functions that match the signals_fn signature
    # The signals_fn should take closes and **params, but we're using fixed signals
    def signals_fn_wrapper(closes, **params):
        # For fixed signal series tests, just return the precomputed signals
        return signals

    result = bt.stress_segments(
        signals_fn=signals_fn_wrapper,
        bars=list(bars),
        signals=signals,
        param_grid=grid,
        baseline=baseline,
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )

    print(f"  overall verdict: {result.overall_verdict}")
    print(f"  segments: [{', '.join(s.name for s in result.scenarios)}]")
    print(f"  candidate dispersion: {result.candidate_dispersion:+.4f}")
    print(f"  null dispersion: {result.null_dispersion:+.4f}")

    for sres in result.scenarios:
        med = sres.baseline_median_log_return
        null = sres.noise_median_log_return
        diff = med - null
        edge = "pos edge" if diff > OUTLIER_TOL else ("neg edge" if diff < -OUTLIER_TOL else "no edge")
        print(f"    {sres.name:12s}: {med:+.3f} vs {null:+.3f} [{edge}]")

    return {
        "ticker": ticker,
        "gate_name": gate_name,
        "overall_verdict": result.overall_verdict,
        "segments": [s.name for s in result.scenarios],
        "candidate_dispersion": result.candidate_dispersion,
        "null_dispersion": result.null_dispersion,
        "regime_dependent": result.regime_dependent,
        "n_edge_present": result.n_edge_present,
        "n_edge_free": result.n_edge_free,
    }


def independent_verification(ticker, base_signals, gated_signals):
    """Independent verification using fresh walk_forward path.

    This recomputes segment medians using a separate implementation path
    from stress_segments to provide independent verification.
    """
    np.random.seed(42)
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)

    runs = []
    for sig_name, sig in [("base", base_signals), ("gated", gated_signals)]:
        # Use fresh walk_forward instead of stress_segments for verification
        seg_fn = bt.segment_fn_from_labels(labels)

        # For verification, we need to compute signals per segment
        # Extract the segment signals
        seg_indices = []
        for i, label in enumerate(labels):
            if label != "insufficient" and i >= WINDOW:
                seg_indices.append(i)

        # Run walk-forward on each qualifying segment
        segment_results = []
        for start_idx in seg_indices:
            # Simple segment: from start_idx to start_idx + 400 (MIN_SEGMENT_BARS)
            end_idx = min(start_idx + MIN_SEGMENT_BARS, len(closes))
            if end_idx - start_idx < MIN_SEGMENT_BARS:
                continue

            seg_closes = closes[start_idx:end_idx]
            seg_dates = dates[start_idx:end_idx]
            seg_bars = list(bt.BarSequence(
                dates=np.arange(start_idx + 1, end_idx + 1, dtype=np.int64),
                opens=np.zeros_like(seg_closes), highs=np.zeros_like(seg_closes),
                lows=np.zeros_like(seg_closes), closes=seg_closes,
                volumes=np.zeros_like(seg_closes),
            ))

            # Use the appropriate signal series for this segment
            seg_signals = sig[start_idx:end_idx]

            cfg = bt.BacktestConfig(warmup_periods=WARM)
            fold_result = bt.walk_forward(
                seg_bars, seg_signals,
                train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg,
            )

            if fold_result.folds:
                log_rets = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                           for f in fold_result.folds]
                segment_results.append({
                    "mean": float(np.mean(log_rets)),
                    "median": float(np.median(log_rets)),
                    "std": float(np.std(log_rets, ddof=1)),
                    "n_folds": len(log_rets),
                    "positive_folds": int((np.array(log_rets) > 0).sum()),
                })

        runs.append({
            "segment_mean": segment_results,
            "sig_name": sig_name,
        })

    return runs


def main():
    np.random.seed(42)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Predeclared Volatility Gate on Lookback-5 Momentum ===")
    print(f"Gate parameters: {GATE_PARAMS}")
    print(f"Target assets: {', '.join(TARGET)}")
    print()

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    print("=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    print("  [PASS] preflight completed")
    print()

    print("=== 3. Leakage review per asset ===")
    cfg0 = bt.BacktestConfig(initial_capital=1e6)
    tickers_data = {}

    for ticker in TARGET:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()

        # Test base momentum signals
        base_signals = momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(base_signals, [b.date for b in bars], warmup=0)

        # Test gated momentum signals
        gated_signals = volatility_gate_momentum_signals(
            closes, lookback=5, vol_window=VOL_WINDOW, vol_threshold=VOL_THRESHOLD
        )
        bt.check_signal_integrity(gated_signals, [b.date for b in bars], warmup=0)

        # Run full-sample backtests for integrity checks
        for sig_name, signals in [("base", base_signals), ("gated", gated_signals)]:
            res = bt.run_bars(list(bars), signals, cfg0)
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
            trades = len([f for f in res.trades if abs(f.shares) > 0])
            print(f"  {ticker} {sig_name:6s}: {bars.n_bars} bars, {trades} trades, leakage [PASS]")

        tickers_data[ticker] = {"bars": bars, "dates": dates, "closes": closes}

    print()

    print("=== 4. Regime-stability stress test (predeclared gate) ===")
    all_results = []

    for ticker in TARGET:
        data = tickers_data[ticker]
        closes = data["closes"]

        # Run both base and gated momentum tests
        base_signals = momentum_signals(closes, lookback=5)
        gated_signals = volatility_gate_momentum_signals(
            closes, lookback=5, vol_window=VOL_WINDOW, vol_threshold=VOL_THRESHOLD
        )

        # Test base momentum
        base_result = run_regime_stability_test(ticker, data["bars"], base_signals, "base_momentum")
        all_results.append(base_result)

        # Test gated momentum
        gated_result = run_regime_stability_test(ticker, data["bars"], gated_signals, "volatility_gate")
        all_results.append(gated_result)

    print()
    print("=== 5. Independent verification ===")

    for ticker in TARGET:
        print(f"\n{ticker} independent verification:")
        data = tickers_data[ticker]
        closes = data["closes"]

        base_signals = momentum_signals(closes, lookback=5)
        gated_signals = volatility_gate_momentum_signals(
            closes, lookback=5, vol_window=VOL_WINDOW, vol_threshold=VOL_THRESHOLD
        )

        verification = independent_verification(ticker, base_signals, gated_signals)
        for run in verification:
            sig_name = run["sig_name"]
            segments = run["segment_mean"]
            if segments:
                mean_medians = [s["median"] for s in segments]
                print(f"  {sig_name}: {len(segments)} segments, "
                      f"mean median log return {np.mean(mean_medians):+.4f}")
            else:
                print(f"  {sig_name}: no qualifying segments")

    print()
    print("=== 6. Null (coin-flip) benchmark ===")

    for ticker in TARGET:
        print(f"\n{ticker}:")
        data = tickers_data[ticker]
        closes = data["closes"]

        # Create null benchmark on the gated signals
        n = len(closes)
        base_signals = momentum_signals(closes, lookback=5)
        gated_signals = volatility_gate_momentum_signals(
            closes, lookback=5, vol_window=VOL_WINDOW, vol_threshold=VOL_THRESHOLD
        )

        # For null benchmark, we need to run the same stress_segments but with
        # a coin-flip signal. Since stress_segments uses the signals_fn directly,
        # we'll use the same approach but with a random signal.
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)

        def random_signals_fn(closes, **params):
            return bt.random_signals(len(closes), seed=42)

        baseline = (("fast", FAST), ("slow", SLOW))
        grid = [{"fast": FAST, "slow": SLOW}]

        null_result = bt.stress_segments(
            signals_fn=random_signals_fn,
            bars=list(data["bars"]),
            signals=[],  # not used for random signals
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )

        print(f"  null benchmark verdict: {null_result.overall_verdict}")
        print(f"  null candidate dispersion: {null_result.candidate_dispersion:+.4f}")

    print()
    print("=== 7. Determinism (independent re-run) ===")

    deterministic_passed = True
    for ticker in TARGET:
        print(f"\n{ticker} determinism check:")
        data = tickers_data[ticker]
        closes = data["closes"]

        # Run gated test twice with different seeds
        np.random.seed(42)
        gated_signals1 = volatility_gate_momentum_signals(
            closes, lookback=5, vol_window=VOL_WINDOW, vol_threshold=VOL_THRESHOLD
        )

        np.random.seed(123)  # Different seed
        gated_signals2 = volatility_gate_momentum_signals(
            closes, lookback=5, vol_window=VOL_WINDOW, vol_threshold=VOL_THRESHOLD
        )

        # For determinism, we need to check that the gate behavior is the same
        # regardless of the RNG seed (since it should be deterministic)
        gate_deterministic = True
        for i in range(len(closes)):
            if (gated_signals1[i].weight > 1e-12) != (gated_signals2[i].weight > 1e-12):
                gate_deterministic = False
                break

        if gate_deterministic:
            print(f"  volatility gate: deterministic across seeds ✓")
        else:
            print(f"  volatility gate: NON-DETERMINISTIC across seeds ✗")
            deterministic_passed = False

    if deterministic_passed:
        print("\n  OVERALL DETERMINISM: PASSED")
    else:
        print("\n  OVERALL DETERMINISM: FAILED")

    print()
    print("=== 8. Summary of findings ===")

    print(f"Target assets: {', '.join(TARGET)}")
    print(f"Gate: volatility threshold {VOL_THRESHOLD:.2%} (window={VOL_WINDOW} bars)")
    print()

    for ticker in TARGET:
        base_verdict = None
        gated_verdict = None

        for result in all_results:
            if result["ticker"] == ticker:
                if result["gate_name"] == "base_momentum":
                    base_verdict = result["overall_verdict"]
                elif result["gate_name"] == "volatility_gate":
                    gated_verdict = result["overall_verdict"]

        print(f"{ticker}:")
        print(f"  base momentum: {base_verdict}")
        print(f"  with volatility gate: {gated_verdict}")

        # Assess whether gate created robustness improvement
        if base_verdict == "REGIME_DEPENDENT" and gated_verdict in ["REGIME_STABLE", "CONSISTENT_WITH_NOISE"]:
            print(f"  → Gate created robustness improvement!")
        elif base_verdict == gated_verdict:
            print(f"  → Gate did not change regime-stability verdict")
        else:
            print(f"  → Gate changed verdict (evaluate impact)")

    print()
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("This tests whether a predeclared past-only volatility gate creates")
    print("robustness improvement for lookback-5 momentum differently from")
    print("previously falsified MA regime filters.")


if __name__ == "__main__":
    main()
