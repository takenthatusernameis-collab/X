"""Test the 20-day breakout continuation signal on the collected universe.

Objective: The momentum frontier establishes a known signal class. A
directionally-biased variant — the breakout continuation rule (long when the
close exceeds the highest close of the preceding N trading days, hold 1 day) —
is a genuinely different signal class: it enters only on new highs rather than
whenever the lookback return is positive. This check tests whether the breakout
filter produces reproducible evidence beyond the momentum family, using only
the canonical 20-day lookback to avoid redundant parameter sweeps.

Method (fixed a-priori, not tuned to OOS): breakout continuation with lookback
= 20 (canonical breakout window). Same walk-forward windows
(train=252d/test=84d/warmup=60d/overlap=60d) and 4 contiguous volatility blocks
as the momentum checks. Each segment's median log return is compared against a
coin-flip sign null benchmark run on the same segments.

An asset shows an EDGE if it displays a uniformly-positive REGIME_STABLE
edge across all segments. A per-asset verdict is REGIME_STABLE if the candidate
dispersion exceeds the null dispersion by >= 2x; CONSISTENT_WITH_NOISE if
medians are all within 0.05 of zero; REGIME_DEPENDENT if candidate dispersion
> 2x null dispersion but not uniformly positive; REGIME_STABLE_LOSS if uniformly
negative.

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
CANONICAL_LOOKBACK = 20
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "breakout_20day_fixed_results.json"


def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    signals = bt.breakout_signals(closes, lookback=CANONICAL_LOOKBACK)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=bt.breakout_signals,
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": CANONICAL_LOOKBACK}],
        baseline=(("lookback", CANONICAL_LOOKBACK),),
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
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )


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
        sig = bt.breakout_signals(closes, lookback=CANONICAL_LOOKBACK)
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars),
                          bt.breakout_signals(closes, lookback=CANONICAL_LOOKBACK),
                          bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, breakout signals "
              f"{len(bt.breakout_signals(closes, lookback=CANONICAL_LOOKBACK))}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: breakout flag at bar t uses closes[:t] only; "
          "neutral bars carry the equity forward.")

    print("\n=== 4. 20-day breakout continuation across collected universe ===")
    print("  Long when close > max(close of prior 20 days), hold 1 day, daily "
          "rebalanced. Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, lookback=20 vs coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    all_results = {}
    for asset in target:
        all_results[asset] = run_asset(asset, tickers)

    print("  20y breakout continuation regime-stability across universe:")
    for asset in target:
        r = all_results[asset]
        print(f"    {asset}: verdict {r['verdict']} "
              f"(candidate {r['candidate_dispersion']:.3f}, null {r['null_dispersion']:.3f}) "
              f"segments {r['segments']} medians {r['medians']}")

    print("\n  Full-universe generalization:")
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: bt.breakout_signals(c, lookback=CANONICAL_LOOKBACK),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", CANONICAL_LOOKBACK),),
        param_grid=[{"lookback": CANONICAL_LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("  " + summary.inspect().replace("\n", "\n  "))

    print("\n=== 5. Determinism check ===")
    res2 = {a: run_asset(a, tickers) for a in target}
    out1 = json.dumps(all_results, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=CANONICAL_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=all_results,
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
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. 20-day breakout continuation across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())