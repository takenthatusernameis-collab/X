"""Test longer-horizon momentum (lookback 20 / 60) on the collected universe.

Objective: the momentum cell (lookback=5) was admitted as candidate positive
evidence in activation 37318950814, showing a REGIME_STABLE positive edge in
7/10 assets, and the lookback-sweep check (activation 37361000967) showed the
edge is robust across lookback 3/5/10 in 6/10 assets. The frontier question is
whether the edge generalizes to the classic momentum formation windows —
lookback 20 and 60 (roughly one and three months) — or whether it is confined
to the short-horizon window.

Method (fixed a-priori, not tuned to OOS): long the previous 20- or 60-day
return (hold 1 day, daily rebalance) — the same momentum construction as
`research/checks/momentum.py`, only a longer formation window. Lookbacks 20
and 60. Same walk-forward windows (train=252d/test=84d/warmup=60d/overlap=60d)
and 4 contiguous volatility blocks as the lookback-sweep check. Each lookback's
per-segment median log return is compared against a coin-flip sign null
benchmark run on the same segments. The synthetic generator contains no
momentum structure, so real data supplies the null.

A lookback is EDGE if it shows a uniformly-positive REGIME_STABLE edge in an
asset; a horizon verdict is ROBUST if a positive REGIME_STABLE edge appears at
>= 2 lookbacks, SENSITIVE if it appears at exactly one lookback, and NO_EDGE
otherwise.

A-priori falsification prediction: if the positive autocorrelation is genuinely
short-horizon, the edge may decay as the formation window lengthens — lookback
20 could still show a positive REGIME_STABLE edge in some assets (edge
persists) while lookback 60 weakens toward CONSISTENT_WITH_NOISE (edge
degrades). If the edge generalizes, lookback 20 and 60 show ROBUST positive
edges in the same 6/10 assets as lookback 3/5/10. Either outcome closes the
cell.

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
LOOKBACKS = (20, 60)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_longer_horizon_results.json"


def segment_result(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def horizon_verdict(sr):
    """Verdict for one lookback's scenario result on one asset.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge on this asset; REGIME_STABLE_LOSS is a stable loss,
    not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def horizon_robustness(lookback_results):
    """Horizon-robustness verdict for one asset across lookbacks 20/60.

    ROBUST: a positive REGIME_STABLE edge at >= 2 lookbacks.
    SENSITIVE: a positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE: no lookback shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[horizon_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
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

    print("\n=== 3. Leakage review per asset ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            signals = bt.momentum_signals(closes, lookback=lb)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), bt.momentum_signals(closes, lookback=20),
                          bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals (20/60) {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Longer-horizon momentum (20 / 60) across collected universe ===")
    print("  Long the previous N-day return, hold 1 day, daily rebalanced. "
          "Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP,
                                          MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index].
    lookback_results = {lb: {t: segment_result(t, all_tickers[t], lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    horizon_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_counts = {lb: {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                            "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
                       for lb in LOOKBACKS}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = horizon_robustness(profile)
        horizon_counts[v] += 1
        for lb in LOOKBACKS:
            lookback_counts[lb][profile[str(lb)]["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["horizon_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"]
                                            for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  Lookahead-20 regime gate verdict counts across universe: "
          f"{lookback_counts[20]}")
    print("  Lookahead-60 regime gate verdict counts across universe: "
          f"{lookback_counts[60]}")
    print("  Horizon-robustness verdict counts: "
          f"ROBUST={horizon_counts['ROBUST']} "
          f"(positive edge at both lookbacks) | "
          f"SENSITIVE={horizon_counts['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={horizon_counts['NO_EDGE']}\n")
    print("  Per-asset horizon profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:6s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments3560={:<40s} hz={:<8s} "
              "edge_counts={}".format(
            ticker, str(p["20"]["segments"]), med, p["horizon_verdict"],
            p["edge_counts"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result(t, tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            lookback_20_counts=lookback_counts[20],
            lookback_60_counts=lookback_counts[60],
            horizon_counts=horizon_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                           for lb in LOOKBACKS},
                    verdicts_20=per_asset[t]["20"]["verdict"],
                    verdicts_60=per_asset[t]["60"]["verdict"],
                    horizon_verdict=per_asset[t]["horizon_verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    n_folds_20=per_asset[t]["20"]["n_folds"],
                    n_folds_60=per_asset[t]["60"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Longer-horizon momentum (20/60) across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
