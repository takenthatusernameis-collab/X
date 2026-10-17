"""Volatility-gated momentum on real data (walk-forward IS/OOS).

A past-only realized-volatility regime filter: estimate recent annualized
volatility (standard deviation of log returns over a lookback window) and hold
the momentum position only in low-vol regimes.

This tests whether the MA regime filters (previously falsified) help the
momentum mechanism (vs reversal). The same class of conditioning applied
to momentum (long the previous 5-day return, hold 1 day, daily rebalance)
may produce a different result than MA crossover regime conditioning.

Method (fixed a-priori, not tuned to OOS):
- Past-only volatility gate: hold only in regimes with realized vol < 0.20 annualized
- Lookback-5 momentum: long previous 5-day return, hold 1 day, daily rebalance
- Predeclared parameters: vol_window=20, vol_threshold=0.20, momentum_lookback=5
- Existing walk-forward settings: train=252d / test=84d / warmup=60d / overlap=60d
- Existing 4-block volatility segmentation, same null framework as momentum.py
- Determinism verified via independent recomputation

Classification follows the same regime-stability framework: CONSISTENT_WITH_NOISE,
REGIME_DEPENDENT, or REGIME_STABLE (with or without edge).
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
VOL_WINDOW = 20
VOL_THRESHOLD = 0.20
MOMENTUM_LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"


def volatility_gated_momentum_signals(closes: np.ndarray, vol_window: int = VOL_WINDOW, vol_threshold: float = VOL_THRESHOLD):
    """Past-only volatility-regime signals gating momentum."""
    n = len(closes)
    sig = bt.momentum_signals(closes, lookback=MOMENTUM_LOOKBACK)
    gated = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    for i in range(int(vol_window) - 1, n):
        log_returns = np.log(closes[i - vol_window + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        gated[i] = bt.Signal(date=i + 1, weight=sig[i].weight if realized_annual < vol_threshold else 0.0)
    return gated


def momentum_signals_fn(closes, **p):
    # Handle both dictionary and tuple params for compatibility
    # When called by stress_segments_across_tickers, p may be a tuple from baseline conversion
    lookback = MOMENTUM_LOOKBACK
    if isinstance(p, dict):
        lookback = p.get("lookback", MOMENTUM_LOOKBACK)
    # Ensure lookback is an integer
    return bt.momentum_signals(closes, lookback=int(lookback))


def volatility_gated_momentum_signals_fn(closes, **p):
    # Handle both dictionary and tuple params for compatibility
    # When called by stress_segments_across_tickers, p may be a tuple converted to dict
    # e.g., {"vol_window": VOL_WINDOW, "vol_threshold": VOL_THRESHOLD}
    # But parameter_sweep may pass a tuple directly
    if isinstance(p, dict):
        vol_window = p.get("vol_window", VOL_WINDOW)
        vol_threshold = p.get("vol_threshold", VOL_THRESHOLD)
    elif isinstance(p, tuple):
        # Convert tuple to dictionary for compatibility
        param_dict = dict(p)
        vol_window = param_dict.get("vol_window", VOL_WINDOW)
        vol_threshold = param_dict.get("vol_threshold", VOL_THRESHOLD)
    else:
        # Default case
        vol_window = VOL_WINDOW
        vol_threshold = VOL_THRESHOLD
    # Ensure parameters are of correct type
    return volatility_gated_momentum_signals(closes, int(vol_window), float(vol_threshold))


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    results = {}

    # Momentum baseline
    signals = bt.momentum_signals(closes, lookback=MOMENTUM_LOOKBACK)
    if len(signals) != len(closes):
        raise ValueError(f"{asset} momentum: signals {len(signals)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=MOMENTUM_LOOKBACK),
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": MOMENTUM_LOOKBACK}],
        baseline=(("lookback", MOMENTUM_LOOKBACK),),
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
    results["momentum"] = dict(
        segments=names,
        medians=medians,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )

    # Volatility-gated momentum
    signals = volatility_gated_momentum_signals(closes)
    if len(signals) != len(closes):
        raise ValueError(f"{asset} volatility_gated: signals {len(signals)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=lambda c, **p: volatility_gated_momentum_signals(c, VOL_WINDOW, VOL_THRESHOLD),
        bars=list(bars),
        signals=signals,
        param_grid=[{"vol_window": VOL_WINDOW, "vol_threshold": VOL_THRESHOLD}],
        baseline=(("vol_window", VOL_WINDOW), ("vol_threshold", VOL_THRESHOLD)),
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
    results["volatility_gated"] = dict(
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

    print("\n=== 3. Leakage review for volatility-gated momentum ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = volatility_gated_momentum_signals(closes)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, volatility-gated momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: gated signal at bar t uses closes[:t] only; "
          "neutral bars before the vol window carry the equity forward.")

    print("\n=== 4. Volatility-gated momentum on REGIME_DEPENDENT assets ===")
    print("Volatility-gated momentum: long the previous {}-day return, hold 1 day, daily "
          "rebalanced, but ONLY in regimes with realized vol < {} annualized. "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, 4 contiguous volatility blocks\n".format(
              MOMENTUM_LOOKBACK, VOL_THRESHOLD, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:14s} {:28s} {:12s}  {:12s}  verdict".format(
            "variant", "segments", "cand_disp", "null_disp"))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    name, segs, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
        print("")

    # Full-universe generalization: volatility-gated momentum across collected universe
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=volatility_gated_momentum_signals_fn,
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("vol_window", VOL_WINDOW), ("vol_threshold", VOL_THRESHOLD)),
        param_grid=[(("vol_window", VOL_WINDOW), ("vol_threshold", VOL_THRESHOLD))],
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

    # Write the artifact that the independent verification file recomputes
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        vol_window=VOL_WINDOW,
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
    print("NOTE: research/simulation only. Volatility-gated momentum: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())