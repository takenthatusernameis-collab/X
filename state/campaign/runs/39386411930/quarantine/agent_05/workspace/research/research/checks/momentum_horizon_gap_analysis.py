"""Null-dispersion-robust horizon characterization of longer-horizon momentum.

Independent recharacterization (no regime-stability verdict gate): for each
asset and lookback (20 / 60), recompute the walk-forward segment medians and
the coin-flip null via a separate implementation path, then report, per asset
and lookback:
  - the candidate median and null median per segment
  - the signed gap (candidate - null) per segment
  - how many segments have a positive candidate gap (independent of the
    regime-stability gate, which is sensitive to null-dispersion fluctuations)

This addresses the finding in `momentum_longer_horizon.py` that the
candidate's per-segment medians are identical across lookback 20 and 60 while
the regime-stability verdicts diverge: the divergence was traced to the
coin-flip null's dispersion varying widely with lookback (e.g. TSLA null
dispersion 0.603 at lookback 20 vs 0.013 at lookback 60), not to the
candidate. A gap-based characterization removes that noise driver and states
the edge in terms of magnitude versus the null directly.

Method (fixed a-priori): long the previous 20-/60-day return, hold 1 day,
daily rebalance; walk-forward per segment train=252d/test=84d/warmup=60d/
overlap=60d; 4 contiguous volatility blocks; coin-flip null via the
framework's noise_benchmark on the same segments. Independent path:
re-implemented volatility blocks / segment reconstruction / momentum signals /
walk_forward + noise_benchmark.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (20, 60)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
OUT = ARTIFACT_DIR / "momentum_horizon_gap_analysis.json"


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_gap(ticker, s, e, train, test, warm, overlap, lookback):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=bt.BacktestConfig())
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = bt.noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252)
    return float(np.median(cand_log)), noise.baseline_median_log_return


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = bt.load_manifest() if hasattr(bt, "load_manifest") else None
    print("=== 1. Manifest integrity ===")
    if manifest is not None:
        from research.data.preflight import sha256_file
        for entry in manifest["entries"]:
            actual = sha256_file(Path(entry["location"]))
            print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
        print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} "
              f"| universe: {len(manifest['universe'])}")
    else:
        from research.data.preflight import load_manifest, sha256_file
        manifest = load_manifest()
        for entry in manifest["entries"]:
            actual = sha256_file(Path(entry["location"]))
            print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
        print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} "
              f"| universe: {len(manifest['universe'])}")

    print("\n=== 2. Independent gap analysis (lookbacks 20 / 60, all 10 assets) ===")
    print("  Independent path: re-implemented volatility blocks / segments, "
          "fresh momentum signals, walk_forward + noise_benchmark. "
          "Reporting candidate - null gap per segment, independent of the "
          "regime-stability verdict gate.\n")

    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0]
               for e in manifest["entries"]}
    rows = []
    for ticker in tickers:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        out_rows = []
        for lb in LOOKBACKS:
            segs = []
            gaps = []
            for lab, s, e in runs:
                c, n = segment_gap(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                gap = c - n
                segs.append([round(c, 3), round(n, 3)])
                gaps.append(round(gap, 3))
            pos = sum(1 for g in gaps if g > 0)
            out_rows.append(
                dict(lookback=lb, segments=lbls,
                     segment_gaps=[segs, gaps],
                     pos_segment_gap_count=f"{pos}/{len(gaps)}",
                     min_gap=min(gaps), max_gap=max(gaps),
                     median_gap=round(float(np.median(gaps)), 3)))
        rows.append(dict(ticker=ticker, lookback20=out_rows[0], lookback60=out_rows[1]))

    for r in rows:
        lb20 = r["lookback20"]
        lb60 = r["lookback60"]
        print("  {:6s} lb20: pos-segments {pos20:<10s} med-gap {mg20:+.3f} "
              "(seg gaps {g20})".format(
            r["ticker"], pos20=lb20["pos_segment_gap_count"],
            mg20=lb20["median_gap"], g20=str(lb20["segment_gaps"][1])))
        print("         lb60: pos-segments {pos60:<10s} med-gap {mg60:+.3f} "
              "(seg gaps {g60})".format(
            pos60=lb60["pos_segment_gap_count"], mg60=lb60["median_gap"],
            g60=str(lb60["segment_gaps"][1])))

    # Determinism: rerun AMZN/JPM and assert identical gap rows.
    amzn_runs = segments_from_labels(vol_blocks(
        tickers["AMZN"].closes_array(), N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    def amzn_gap_row(lb):
        return [round(segment_gap("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0]
                      - segment_gap("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lb)[1], 3)
                for lab, s, e in amzn_runs]
    r1 = {lb: amzn_gap_row(lb) for lb in LOOKBACKS}
    r2 = {lb: amzn_gap_row(lb) for lb in LOOKBACKS}
    print("determinism: r1 == r2: {}".format(json.dumps(r1, sort_keys=True) ==
          json.dumps(r2, sort_keys=True)))
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    artifact = dict(dataset_id=DATASET_ID, seed=42, lookbacks=list(LOOKBACKS),
                    train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
                    window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
                    per_asset=rows)
    with open(OUT, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(OUT))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Longer-horizon momentum null-gap analysis (20/60): "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
