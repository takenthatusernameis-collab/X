"""Test the 20-day breakout continuation signal on the collected universe.

Objective: the momentum frontier (long the previous N-day return) is admitted as
candidate positive evidence with a short-horizon REGIME_STABLE edge. A
directionally-biased variant — the breakout continuation rule (long when the
close exceeds the highest close of the preceding N trading days, hold 1 day) —
is a genuinely different signal class: it enters only on new highs rather than
whenever the lookback return is positive. This check tests whether the breakout
filter produces reproducible evidence beyond the momentum family.

Method (fixed a-priori, not tuned to OOS): breakout continuation with lookback
in {5, 10, 20} (20 is the canonical breakout window; 5/10 are a bounded
sensitivity set). Same walk-forward windows
(train=252d/test=84d/warmup=60d/overlap=60d) and 4 contiguous volatility blocks
as the momentum checks. Each lookback's per-segment median log return is
compared against a coin-flip sign null benchmark run on the same segments.

A lookback is EDGE on an asset if it shows a uniformly-positive REGIME_STABLE
edge in the asset. A per-lookback sensitivity verdict is NO_EDGE if no lookback
shows a positive edge in that asset, SENSITIVE if it appears at exactly one
lookback, and ROBUST if it appears at >= 2 lookbacks.

A-priori falsification prediction (NOT observed): the breakout filter is a
directionally-biased selection of the momentum long leg; without its own price
structure it should add little — the rule is expected to show
CONSISTENT_WITH_NOISE in most assets (small positive medians below the
framework's noise band), with possible small REGIME_STABLE positive edges only
where the momentum edge already exists (AAPL, MSFT, GOOGL, AMZN, META, TSLA).
Either outcome closes the cell.

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
LOOKBACKS = (5, 10, 20)
CANONICAL_LOOKBACK = 20
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "breakout_20day_cont_results.json"


def segment_result(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.breakout_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.breakout_signals,
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
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def per_lookback_edge(sr):
    """Whether this lookback shows a uniformly-positive REGIME_STABLE edge."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def sensitivity_verdict(lookback_results):
    """Sensitivity verdict for one asset across lookbacks {5, 10, 20}."""
    counts = {"EDGE": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        if per_lookback_edge(lookback_results[str(lb)]):
            counts["EDGE"] += 1
        else:
            counts["NO_EDGE"] += 1
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
            sig = bt.breakout_signals(closes, lookback=lb)
            bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars),
                          bt.breakout_signals(closes, lookback=CANONICAL_LOOKBACK),
                          bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, breakout signals (5/10/20) "
              f"{len(bt.breakout_signals(closes, lookback=20))}, leakage [PASS], "
              f"total_return {total_return:+.3f}")
    print("  no-look-ahead: breakout flag at bar t uses closes[:t] only; "
          "neutral bars carry the equity forward.")

    print("\n=== 4. Breakout continuation across collected universe ===")
    print("  Long when close > max(close of prior N days), hold 1 day, daily "
          "rebalanced. Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, lookbacks {lb} vs coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP,
                                           MIN_SEGMENT_BARS, lb=" / ".join(map(str, LOOKBACKS))))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    lookback_results = {lb: {t: segment_result(t, all_tickers[t], lb)
                             for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    sensitivity_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_counts = {lb: {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                            "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
                       for lb in LOOKBACKS}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = sensitivity_verdict(profile)
        sensitivity_counts[v] += 1
        for lb in LOOKBACKS:
            lookback_counts[lb][profile[str(lb)]["verdict"]] += 1
        profile["sensitivity_verdict"] = v
        profile["sensitivity_counts"] = counts
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"]
                                            for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  Lookback regime-stability verdict counts across universe:")
    for lb in LOOKBACKS:
        print(f"    lookback {lb}: {lookback_counts[lb]}")
    print(f"  Sensitivity verdict counts: "
          f"ROBUST={sensitivity_counts['ROBUST']} (edge at >=2 lookbacks) | "
          f"SENSITIVE={sensitivity_counts['SENSITIVE']} (edge at exactly 1 lookback) | "
          f"NO_EDGE={sensitivity_counts['NO_EDGE']}\n")
    print("  Per-asset profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        details = ", ".join(
            "{}:[{:6.3f},{:6.3f},{:6.3f}]{:9s}".format(
                lb,
                p[str(lb)]["medians"][0],
                p[str(lb)]["medians"][1] if len(p[str(lb)]["medians"]) > 1 else p[str(lb)]["medians"][0],
                p[str(lb)]["medians"][2] if len(p[str(lb)]["medians"]) > 2 else p[str(lb)]["medians"][0],
                p[str(lb)]["verdict"],
            )
            for lb in LOOKBACKS
        )
        print("    {:6s} {} hz={:<9s} counts={}".format(
            ticker, details, p["sensitivity_verdict"], p["sensitivity_counts"]))
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
        canonical_lookback=CANONICAL_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            lookback_counts=lookback_counts,
            sensitivity_counts=sensitivity_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                           for lb in LOOKBACKS},
                    verdicts={str(lb): per_asset[t][str(lb)]["verdict"] for lb in LOOKBACKS},
                    sensitivity_verdict=per_asset[t]["sensitivity_verdict"],
                    sensitivity_counts=per_asset[t]["sensitivity_counts"],
                    n_folds={str(lb): per_asset[t][str(lb)]["n_folds"] for lb in LOOKBACKS},
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Breakout continuation (5/10/20-day) across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
