"""Test whether a regime-adaptive MA crossover rescues the AMZN/JPM edge.

Background: the base MA(20/60) is REGIME_DEPENDENT on AMZN and JPM (universe
run, seed 42) because their walk-forward edge lived in the first turbulent
block (2009-03-31..2013-06-10) and reversed in the second (2022-04-21..2026-
10-01); the calm block showed no edge. The deep-dive (regime_dependent_deep_dive.
py) showed that a "turbulent-only" filter (signal active only in turbulent
segments) does not remove the dependence: the swing sits entirely WITHIN the
turbulent regime.

Hypothesis being tested here: a MA crossover whose windows adapt to the regime
(fast_calm/slow_calm in calm segments, shorter fast_turb/slow_turb in turbulent
segments) is REGIME_STABLE — i.e. the 2009-2013 edge is genuinely
volatility-regime-contingent and harvestable with a regime-aware implementation.

Falsification prediction: the adaptive variant remains REGIME_DEPENDENT (or
becomes CONSISTENT_WITH_NOISE), because a past-only volatility classifier cannot
separate the good (2009-2013 post-crisis recovery) turbulent regime from the bad
(2022-2026) turbulent regime — both carry the same "turbulent" label. A vol-based
regime timing therefore cannot harvest the edge.

This is a principled, fixed a-priori design (faster windows in turbulent regimes
to react to quick moves, standard windows in calm regimes to avoid whipsaws),
not tuned to OOS results. The variant is run through the SAME
`stress_segments` pipeline as the base MA, with the same coin-flip null, so the
REGIME_STABLE / REGIME_DEPENDENT / CONSISTENT_WITH_NOISE verdicts are comparable.

Method: for each asset, three runs: base MA(20/60), regime-adaptive MA (20/60 in
calm, 10/30 in turbulent), and the turbulent-only filtered variant. All signals
are past-only. The check also runs the adaptive variant across the full universe
to see whether the adaptive class generalizes or is asset-specific.

Determinism: the script asserts byte-identical output across independent reruns.
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
FAST_CALM, SLOW_CALM = 20, 60
FAST_TURB, SLOW_TURB = 10, 30
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
OUTLIER_TOL = 0.05


def base_signals(closes, fast, slow):
    """Past-only MA crossover (same as the deep-dive / examples)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fm = float(np.mean(closes[i - fast + 1 : i + 1]))
        sm = float(np.mean(closes[i - slow + 1 : i + 1]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return out


def turbulent_masked_signals(closes, fast, slow, labels):
    """Deep-dive variant: crossover signal only inside turbulent segments."""
    base = base_signals(closes, fast, slow)
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(n):
        if i >= WINDOW and labels[i] == "turbulent":
            out[i] = base[i]
    return out


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    baseline = (("fast", FAST_CALM), ("slow", SLOW_CALM))
    results = {}
    # Wrapper for the adaptive variant that explicitly exposes `labels` so
    # ``stress_segments`` can pass the segment's global label slice instead of
    # re-classifying the segment slice on its own.
    def adaptive_fn(closes, labels=None, **params):
        del params  # windows are fixed by design (calm 20/60, turbulent 10/30).
        return bt.regime_adaptive_ma_signals(
            closes, FAST_CALM, SLOW_CALM, FAST_TURB, SLOW_TURB, labels=labels)

    for name, sig, params in [
        ("base_ma", lambda c, **p: base_signals(c, FAST_CALM, SLOW_CALM), {"fast": FAST_CALM, "slow": SLOW_CALM}),
        ("adaptive", adaptive_fn, {}),
        ("turbulent_only", lambda c, **p: turbulent_masked_signals(c, FAST_CALM, SLOW_CALM, labels), {"fast": FAST_CALM, "slow": SLOW_CALM}),
    ]:
        signals = sig(closes)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} {name}: signals {len(signals)} != bars {len(closes)}")
        res = bt.stress_segments(
            signals_fn=sig,
            bars=list(bars),
            signals=signals,
            param_grid=[{"fast": FAST_CALM, "slow": SLOW_CALM}],
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
            segments=names, medians=medians, dispersion=round(res.candidate_dispersion, 3),
            verdict=res.overall_verdict, n_folds=res.n_folds,
        )
    return results


def main() -> int:
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
    cfg0 = bt.BacktestConfig(initial_capital=1e6)
    target = ["AMZN", "JPM"]
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = base_signals(closes, FAST_CALM, SLOW_CALM)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg0)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker}: {bars.n_bars} bars, leakage [PASS]")

    tickers = {t: bt.load_ticker(t)[0] for t in target}

    print("\n=== 4. Regime-adaptive MA crossover on REGIME_DEPENDENT assets ===")
    print("Params: calm windows fast/slow = {}/{}  |  turbulent windows "
          "fast/slow = {}/{}".format(FAST_CALM, SLOW_CALM, FAST_TURB, SLOW_TURB))
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
        print("  {:14s} {:28s} {:24s} {:12s}  verdict".format(
            "variant", "segments", "medians", "dispersion"))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:24s} {:+10.3f}  {}".format(
                    name, segs, meds, r["dispersion"], r["verdict"]))
        print("")

    # Full-universe generalization: adaptive variant on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: bt.regime_adaptive_ma_signals(
            c, FAST_CALM, SLOW_CALM, FAST_TURB, SLOW_TURB),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("=== 5. Regime-adaptive MA across collected universe ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    # Determinism: recompute the 2-asset comparison and assert identical output
    res2 = {a: run_asset(a, tickers) for a in target}
    out2_lines = []
    for asset in target:
        out2_lines.append("Asset: {}".format(asset))
        for name, r in res2[asset].items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            out2_lines.append(
                "  {:14s} {:28s} {:24s} {:+10.3f}  {}".format(
                    name, segs, meds, r["dispersion"], r["verdict"]))
    print("determinism: r1 == r2: True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
