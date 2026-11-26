"""Real-data holding-period sensitivity of the short-horizon momentum class.

Frontier cell R-002 (campaign slot 04). The momentum class was admitted as
candidate positive evidence (activations 37318950814 / 37361000967) and has
since been qualified on lookback (3/5/10), horizon (5/20/60) and transaction
cost (R-001). The one remaining temporal qualification is the *holding period*:
all admitted evidence is at a 1-day hold with daily rebalance. This check asks
whether the positive edge survives a small predeclared variation in the holding
period, or whether it is a single-point (1-day) timing artifact.

Primary question: does the qualified momentum effect survive a small
predeclared holding-period variation without becoming a single-point timing
artifact?

Design (fixed a-priori, NOT tuned to any holding-period result):
- Momentum = long the previous ``lookback``-day return with a fixed holding
  period ``hold`` (``bt.momentum_hold_signals``), lookback FIXED at 5.
- Predeclared holding-period set HOLD_PERIODS = (1, 5, 10, 20). hold=1 is the
  reference and reproduces the admitted lookback-5 momentum artifact exactly
  (``momentum_hold_signals(hold=1) == momentum_signals``).
- Same walk-forward and 4-block volatility regime family as the admitted
  momentum artifacts (train=252d/test=84d/warmup=60d/overlap=60d, min 400
  bars/segment), each hold vs the framework's coin-flip null on the same
  segments (``bt.stress_segments``).
- Per-asset holding-period robustness (analogous to the lookback sweep):
    EDGE at a hold  : REGIME_STABLE with uniformly positive medians.
    HOLD_ROBUST     : EDGE at >= 2 of the predeclared holds.
    HOLD_SENSITIVE  : EDGE at exactly 1 hold (a single-point timing warning).
    NO_EDGE         : no hold shows a positive REGIME_STABLE edge.
  An asset that is EDGE at hold=1 but at NO hold>1 is flagged as a
  single-point (1-day) timing artifact for that asset.

A-priori prediction (recorded before execution): the momentum edge is driven by
genuine short-horizon positive return autocorrelation, so if it is real the
positive REGIME_STABLE edge should persist across the predeclared holds for the
same robust asset set (AAPL, MSFT, GOOGL, AMZN, META, TSLA), with at most mild
degradation at longer holds. Falsification: if the positive edge exists ONLY at
hold=1 and vanishes (NO_EDGE or negative) at every hold>1 for those assets, the
effect is a single-point (1-day) timing artifact and the temporal-robustness
claim is REJECTED.

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACK = 5
HOLD_PERIODS = (1, 5, 10, 20)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_hold_period_results.json"


def segment_result(ticker, bars, hold):
    """Walk-forward regime-stability result for one ticker and one hold period."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_hold_signals(closes, lookback=LOOKBACK, hold=hold)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_hold_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": LOOKBACK, "hold": hold}],
        baseline=(("lookback", LOOKBACK), ("hold", hold)),
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
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def hold_verdict(sr):
    """EDGE / LOSS / NO_EDGE for one hold period's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(hold_results):
    """Holding-period robustness verdict for one asset.

    HOLD_ROBUST     : uniformly positive REGIME_STABLE edges at >= 2 holds.
    HOLD_SENSITIVE  : a positive REGIME_STABLE edge at exactly one hold.
    NO_EDGE         : no hold shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for h in HOLD_PERIODS:
        counts[hold_verdict(hold_results[str(h)])] += 1
    if counts["EDGE"] >= 2:
        return "HOLD_ROBUST", counts
    if counts["EDGE"] == 1:
        return "HOLD_SENSITIVE", counts
    return "NO_EDGE", counts


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
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

    print("\n=== 3. Leakage review per asset (lookback=5, hold=5) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_hold_signals(closes, lookback=LOOKBACK, hold=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: held weight at bar i uses closes[:i] only "
          "(rebalance bar j <= i); neutral bars before the lookback carry equity.")

    print("\n=== 4. Holding-period sweep (hold 1 / 5 / 10 / 20; lookback fixed 5) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each hold vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  Predeclared HOLD_PERIODS = {} (hold=1 is the reference / admitted "
          "1-day hold)\n".format(list(HOLD_PERIODS)))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [hold][ticker].
    hold_results = {h: {t: segment_result(t, all_tickers[t], h) for t in ticker_order}
                    for h in HOLD_PERIODS}

    per_asset = {}
    robustness_counts = {"HOLD_ROBUST": 0, "HOLD_SENSITIVE": 0, "NO_EDGE": 0}
    hold_1_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                     "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(h): hold_results[h][ticker] for h in HOLD_PERIODS}
        v, counts = robustness_verdict({str(h): profile[str(h)] for h in HOLD_PERIODS})
        robustness_counts[v] += 1
        r1 = profile["1"]
        hold_1_counts[r1["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["hold_medians"] = {str(h): profile[str(h)]["medians"] for h in HOLD_PERIODS}
        profile["hold_null_medians"] = {str(h): profile[str(h)]["null_medians"]
                                        for h in HOLD_PERIODS}
        # Single-point (1-day) timing artifact flag: EDGE at hold=1 but no
        # positive edge at any hold>1.
        edge_gt1 = sum(1 for h in HOLD_PERIODS if h > 1
                       and hold_verdict(profile[str(h)]) == "EDGE")
        profile["edge_at_hold_gt1"] = edge_gt1
        profile["single_point_timing_artifact"] = (
            hold_verdict(profile["1"]) == "EDGE" and edge_gt1 == 0)
        per_asset[ticker] = profile

    print("  hold=1 (reference) verdict counts across universe: {}".format(hold_1_counts))
    print("  holding-period robustness verdict counts: "
          "HOLD_ROBUST={} (>=2 holds positive edge) | "
          "HOLD_SENSITIVE={} (edge at exactly one hold) | "
          "NO_EDGE={}\n".format(
              robustness_counts["HOLD_ROBUST"],
              robustness_counts["HOLD_SENSITIVE"],
              robustness_counts["NO_EDGE"]))
    print("  Per-asset holding-period profiles (medians per hold + verdict):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "h{}:{:6.3f} {:4s}".format(h, p[str(h)]["medians"][0],
                                       hold_verdict(p[str(h)]))
            for h in HOLD_PERIODS
        )
        print("    {:6s} segments={:<8} {} rob={:<13s} edge_counts={} "
              "single_point={}".format(
                  ticker, str(p["1"]["segments"]), med, p["robustness_verdict"],
                  p["edge_counts"], p["single_point_timing_artifact"]))
    print("")

    # Universe temporal-robustness classification over the hold=1 EDGE set.
    hold1_edge_assets = [t for t in ticker_order
                         if hold_verdict(per_asset[t]["1"]) == "EDGE"]
    robust_of_edge = [t for t in hold1_edge_assets
                      if per_asset[t]["robustness_verdict"] == "HOLD_ROBUST"]
    sensitive_of_edge = [t for t in hold1_edge_assets
                         if per_asset[t]["robustness_verdict"] == "HOLD_SENSITIVE"]
    artifact_assets = [t for t in ticker_order
                       if per_asset[t]["single_point_timing_artifact"]]

    n_edge = len(hold1_edge_assets)
    n_robust = len(robust_of_edge)
    if n_edge == 0:
        universe_class = "NO_EDGE"
    elif n_robust >= 4:
        universe_class = "TEMPORALLY_ROBUST"
    elif n_robust >= 1:
        universe_class = "TEMPORALLY_FRAGILE"
    else:
        universe_class = "SINGLE_POINT_ARTIFACT"

    print("  hold=1 EDGE assets ({}): {}".format(n_edge, hold1_edge_assets))
    print("  HOLD_ROBUST among them ({}): {}".format(n_robust, robust_of_edge))
    print("  HOLD_SENSITIVE among them ({}): {}".format(len(sensitive_of_edge),
                                                       sensitive_of_edge))
    print("  single-point (1-day) timing artifacts ({}): {}".format(
        len(artifact_assets), artifact_assets))
    print("  universe temporal classification: {}".format(universe_class))

    # Determinism: rerun the target subset and assert identical output.
    res2 = {h: {t: segment_result(t, tickers[t], h) for t in target}
            for h in HOLD_PERIODS}
    out1 = json.dumps({h: {t: hold_results[h][t] for t in target}
                       for h in HOLD_PERIODS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("\ndeterminism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        hold_periods=list(HOLD_PERIODS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            hold_1_counts=hold_1_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            hold1_edge_assets=hold1_edge_assets,
            hold_robust_assets=robust_of_edge,
            hold_sensitive_assets=sensitive_of_edge,
            single_point_timing_artifacts=artifact_assets,
            classification=universe_class,
            per_asset=dict(
                (t, dict(
                    hold_medians={str(h): per_asset[t]["hold_medians"][str(h)]
                                  for h in HOLD_PERIODS},
                    hold_null_medians={str(h): per_asset[t]["hold_null_medians"][str(h)]
                                       for h in HOLD_PERIODS},
                    verdicts={str(h): per_asset[t][str(h)]["verdict"] for h in HOLD_PERIODS},
                    edge_counts=per_asset[t]["edge_counts"],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    single_point_timing_artifact=per_asset[t]["single_point_timing_artifact"],
                    n_folds_1=per_asset[t]["1"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
        a_priori_prediction=(
            "Positive REGIME_STABLE edge persists across holds {1,5,10,20} for "
            "the robust asset set (>=4 of the hold=1 EDGE assets HOLD_ROBUST); "
            "edge only at hold=1 with NO_EDGE at all hold>1 => single-point "
            "(1-day) timing artifact (REJECTED)."
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum holding-period sensitivity on the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())