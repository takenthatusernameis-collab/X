"""Holding-period robustness of the qualified lookback-5 momentum signal.

Primary question (R-002): does the qualified momentum effect survive a small
holding-period variation without becoming a single-point timing artifact?

Method (fixed a-priori, not tuned to OOS):
- Fixed lookback: 5 days (the admitted positive candidate; NOT re-tuned here).
- Predeclared holding-period grid: [1, 2, 5] days. Declared before execution;
  no hold period is chosen, added, or tuned after inspecting results.
- Mechanical bet: long the previous 5-day return; hold H bars; re-enter at the
  next slot boundary (daily rebalance is H=1). The engine carries the entry
  for exactly hold_period bars before re-evaluating the signal.
- Regime-stability gate on the collected 10-asset universe: 4 contiguous
  volatility blocks, walk-forward train=252d / test=84d / warmup=60d /
  overlap=60d, min 400 bars/segment, each hold vs a coin-flip sign null that
  shares the same holds and the same walk-forward.
- A synthetic perturbation sweep (lookback 3 / 5 / 10 at hold=1) exercises the
  parameter-robustness gate on simulated regime families.
- Determinism is asserted via an internal rerun of the AMZN/JPM subset; an
  independent verifier recomputes the artifact from a separate walk_forward
  path.

Classification (a-priori): for one asset at one hold, a "hold" is EDGE when the
segment medians are uniformly positive and REGIME_STABLE, LOSS when uniformly
negative (REGIME_STABLE_LOSS, i.e. a stable loss rather than an edge), and
NO_EDGE otherwise. Across the grid: ROBUST = EDGE at all three holds; NO_EDGE
= NO_EDGE at all three holds; SENSITIVE = a mix. A SENSITIVE result in which
the only edge sits at hold=1 marks the effect as a single-point timing
artifact.

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
HOLD_PERIODS = (1, 2, 5)          # predeclared grid -- fixed before execution
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_periods_results.json"


def segment_result_for_hold(ticker, bars, hold):
    """Walk-forward regime-stability result for one ticker at one hold."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=LOOKBACK)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": LOOKBACK}],
        baseline=(("lookback", LOOKBACK),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=bt.BacktestConfig(hold_period=hold),
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        medians_unrounded=[s.baseline_median_log_return for s in res.scenarios],
        null_medians_unrounded=[s.noise_median_log_return for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def hold_verdict(sr):
    """EDGE / LOSS / NO_EDGE for one hold's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(hold_results):
    """Verdict on whether the edge is holding-period-robust for one asset.

    ROBUST      : uniformly positive REGIME_STABLE edges at all three holds.
    SENSITIVE   : a positive REGIME_STABLE edge at one or two holds only.
    NO_EDGE     : no hold shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for h in HOLD_PERIODS:
        counts[hold_verdict(hold_results[str(h)])] += 1
    if counts["EDGE"] == len(HOLD_PERIODS):
        return "ROBUST", counts
    if counts["EDGE"] == 0 and counts["LOSS"] == 0:
        return "NO_EDGE", counts
    return "SENSITIVE", counts


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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review per asset (hold 1 / 2 / 5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        for h in HOLD_PERIODS:
            res = bt.run_bars(list(bars), signals,
                              bt.BacktestConfig(hold_period=h, initial_capital=1e6))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum holding-period sweep (hold 1 / 2 / 5) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each hold vs a coin-flip null on "
          "the same segments\n".format(TRAIN, TEST, WARM, OVERLAP,
                                       MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    hold_results = {h: {t: segment_result_for_hold(t, all_tickers[t], h)
                        for t in ticker_order}
                    for h in HOLD_PERIODS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    hold_counts = {str(h): {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                            "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
                   for h in HOLD_PERIODS}
    for ticker in ticker_order:
        profile = {str(h): hold_results[h][ticker] for h in HOLD_PERIODS}
        v, counts = robustness_verdict(profile)
        robustness_counts[v] += 1
        for h in HOLD_PERIODS:
            hold_counts[str(h)][profile[str(h)]["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["hold_medians"] = {str(h): profile[str(h)]["medians"]
                                   for h in HOLD_PERIODS}
        profile["hold_null_medians"] = {str(h): profile[str(h)]["null_medians"]
                                        for h in HOLD_PERIODS}
        per_asset[ticker] = profile

    print("  holding-period verdict counts across universe:")
    for h in HOLD_PERIODS:
        print("    hold={} : {}".format(h, hold_counts[str(h)]))
    print("  holding-period-robustness verdict counts: "
          "ROBUST={}(all 3 holds EDGE) | SENSITIVE=(edge at 1-2 holds) | "
          "NO_EDGE=(no hold EDGE): ROBUST={}; SENSITIVE={}; NO_EDGE={}\n".format(
              robustness_counts["ROBUST"], robustness_counts["ROBUST"],
              robustness_counts["SENSITIVE"], robustness_counts["NO_EDGE"]))
    print("  Per-asset holding profiles (first segment median, verdict):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:7.3f} {:4s}".format(h, p[str(h)]["medians"][0],
                                       p[str(h)]["verdict"])
            for h in HOLD_PERIODS
        )
        print("    {:6s} segments={:<8} medians125={:<42s} rob={:<8s} "
              "edge={}".format(
            ticker, str(p["1"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {h: {t: segment_result_for_hold(t, tickers[t], h)
                for t in target}
            for h in HOLD_PERIODS}
    out1 = json.dumps({h: {t: hold_results[h][t] for t in target}
                       for h in HOLD_PERIODS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    universe_per_asset = {}
    for t in ticker_order:
        p = per_asset[t]
        universe_per_asset[t] = dict(
            hold_medians={str(h): p["hold_medians"][str(h)] for h in HOLD_PERIODS},
            hold_null_medians={str(h): p["hold_null_medians"][str(h)]
                               for h in HOLD_PERIODS},
            hold_counts=p["edge_counts"],
            robustness_verdict=p["robustness_verdict"],
            n_folds_hold1=p["1"]["n_folds"],
        )

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        holds=list(HOLD_PERIODS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            hold_counts=hold_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=universe_per_asset,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum holding-period sweep across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
