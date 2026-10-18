"""Test a past-only volatility gate on lookback-5 momentum.

Background: The base MA(20/60) is REGIME_DEPENDENT on AMZN and JPM because their
walk-forward edge lived in the first turbulent block (2009-2013) and reversed in the
second (2022-2026); the calm block showed no edge. A "turbulent-only" filter (signal
active only in turbulent segments) does not remove the dependence: the swing sits
entireLY WITHIN the turbulent regime. The question now: does a past-only volatility
gate help the momentum mechanism (lookback=5)? Unlike the regime-adaptive or
segment-based filters that use actual segment labels, this gate uses a past-only
volatility classifier (realized vol > 25% annualized) applied at signal emission.

Predeclared gate: momentum signals only when early (past-only) realized volatility
> 0.25 annualized, neutral otherwise. This is a single threshold rule declared before
execution; we do not search or optimize after seeing results.

Method: For each asset, run the existing momentum signal and a volatility-gated
variant through the same stress_segments_across_tickers pipeline (same parameters,
including the coin-flip null) so verdicts are comparable. The universe includes the
10 assets from the Yahoo Finance OHLCV collection.

Verification: Both variants are run deterministically; the script asserts identical
output across independent re-runs.
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


def run_asset(asset, tickers):
    """Run both regular momentum and volatility-gated momentum on one asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    baseline = (("lookback", LOOKBACK),)
    results = {}

    # 1. Regular momentum signal (baseline for comparison)
    for name, sig, params in [
        ("momentum_regular", lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK), {"lookback": LOOKBACK}),
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

    # 2. Volatility-gated momentum: conditional_signal(momentum_signals, active_when_turbulent)
    # This is the gate we are testing: signal active only when early vol > 0.25 annualized, neutral otherwise.
    gated_fn = bt.conditional_signal(
        base_signals_fn=bt.momentum_signals,
        condition_fn=bt.active_when_turbulent,
    )
    
    # Wrap the gated function to accept lookback parameter like momentum_signals
    def wrapped_gated_fn(closes, lookback=LOOKBACK):
        return gated_fn(closes, lookback=lookback)
    
    gated_signals = wrapped_gated_fn(closes)
    if len(gated_signals) != len(closes):
        raise ValueError(f"{asset} momentum_gated: signals {len(gated_signals)} != bars {len(closes)}")

    # Use the same wrapped function for stress_segments
    res_gated = bt.stress_segments(
        signals_fn=wrapped_gated_fn,
        bars=list(bars),
        signals=gated_signals,
        param_grid=[{"lookback": LOOKBACK}],
        baseline=baseline,
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    names_g = [s.name for s in res_gated.scenarios]
    medians_g = [round(s.baseline_median_log_return, 3) for s in res_gated.scenarios]
    results["momentum_gated"] = dict(
        segments=names_g,
        medians=medians_g,
        candidate_dispersion=round(res_gated.candidate_dispersion, 3),
        null_dispersion=round(res_gated.null_dispersion, 3),
        verdict=res_gated.overall_verdict,
        n_folds=res_gated.n_folds,
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

    print("\n=== 4. Past-only volatility gate on lookback-5 momentum ===")
    print("Method: past-only volatility classifier > 0.25 annualized (active_when_turbulent).")
    print("Gate rule: signal active only when early realized volatility > 0.25; neutral otherwise.")
    print("Same walk-forward settings: train={}d/test={}d/warmup={}d/overlap={}d, min {} bars/segment, vs coin-flip null.\n".format(
        TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:18s} {:28s} {:28s} {:12s}  {:12s}  verdict".format(
            "variant", "segments", "candidate_medians", "cand_disp", "null_disp"))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:18s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    name, segs, meds, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
        print("")

    # Full-universe generalization for the gated variant
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    # Create the gated signals function for the full universe
    gated_fn_full = bt.conditional_signal(
        base_signals_fn=bt.momentum_signals,
        condition_fn=bt.active_when_turbulent,
    )

    # Wrap the gated function so it accepts lookback parameter like momentum_signals
    def gated_signals_full(closes, **params):
        return gated_fn_full(closes, **params)

    summary_gated = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=gated_signals_full,
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
    print("  " + summary_gated.inspect().replace("\n", "\n  "))
    print("")

    # Determinism: recompute the 2-asset comparison and assert identical output
    results_gated = {a: run_asset(a, tickers) for a in target}
    out2_lines = []
    for asset in target:
        out2_lines.append("Asset: {}".format(asset))
        for name, r in results_gated[asset].items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            out2_lines.append(
                "  {:18s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    name, segs, meds, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(results_gated, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact that the independent verification file recomputes
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=results_all,
        universe_gated=dict(
            verdict_counts=summary_gated.verdict_counts,
            ticker_order=summary_gated.ticker_order,
            per_asset=dict(
                (t, dict(
                    segments=[s.name for s in r.scenarios],
                    medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                    candidate_dispersion=round(r.candidate_dispersion, 3),
                    null_dispersion=round(r.null_dispersion, 3),
                    verdict=r.overall_verdict,
                ))
                for t, r in summary_gated.assets.items()
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. Past-only volatility gate on lookback-5 momentum: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())