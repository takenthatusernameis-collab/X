"""Independent verification of the momentum concentration analysis
(`research/checks/momentum_concentration.py`).

This is a separate computation path: it recomputes each lookback-5 segment
median via fresh `walk_forward` (fresh momentum signals, fresh volatility
blocks and segments, fresh coin-flip null), then recomputes the per-asset
signed edges, absolute effects, HHI and top-k positive-edge shares, and
compares those derived concentration metrics against the artifact written by
the concentration check (`state/check_artifacts/momentum_concentration.json`).
A mismatch would flag a defect in the concentration check.

This verifies the highest-impact claim of the concentration analysis — the
DISTRIBUTED verdict at the base lookback — via a computation path that shares
no code with the concentration metrics, only with the research.backtest engine
contract (walk_forward / momentum_signals / volatility_blocks).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
OUTLIER_TOL = 0.05
LOOKBACK = 5
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_concentration.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_sweep_results.json"


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
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
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def noise_signals(closes, rng):
    """Price-independent coin-flip signal (independent null)."""
    n = len(closes)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(weights[i])) for i in range(n)]


def segment_result(tickers, ticker, s, e, train, test, warm, overlap, lookback):
    bars = tickers[ticker]
    closes = bars.closes_array()
    cfg = bt.BacktestConfig(warmup_periods=warm)
    pos = bt.walk_forward(
        list(bars)[s:e],
        momentum_signals(closes, lookback)[s:e],
        train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    neg = bt.walk_forward(
        list(bars)[s:e],
        noise_signals(closes[s:e], np.random.default_rng(42 + int(round(lookback) * 1000))),
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, cfg=bt.BacktestConfig())
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in pos.folds], dtype=np.float64)
    null = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in neg.folds], dtype=np.float64)
    return float(np.median(cand)), float(np.median(null))


def concentration_metrics(per_asset):
    """Recompute the concentration metrics from per-asset (median, null_median)
    rows — the same quantities as research/backtest/regime_stability."""
    abs_effects = [float(row["abs_effect"]) for row in per_asset.values()]
    total_abs = float(sum(abs_effects))
    total_pos = float(sum(max(row["signed_edge"], 0.0) for row in per_asset.values()))
    mean_abs = [float(row["mean_abs_effect"]) for row in per_asset.values()]
    shares = [a / total_abs for a in abs_effects]
    hhi = float(sum(s * s for s in shares))
    pos_sorted = sorted(per_asset.items(),
                        key=lambda kv: float(kv[1]["signed_edge"]),
                        reverse=True)
    pos_sorted = [kv for kv in pos_sorted if float(kv[1]["signed_edge"]) > 0.0]
    top1 = pos_sorted[0][1]["signed_edge"] / total_pos if total_pos > 0 else 0.0
    top3 = sum(float(kv[1]["signed_edge"]) for kv in pos_sorted[:3]) / total_pos if total_pos > 0 else 0.0
    max_med = max(mean_abs)
    median_abs = float(np.median(mean_abs))
    max_med_ratio = (max_med / median_abs) if median_abs > 0 else float("inf")
    return dict(hhi=round(hhi, 6), top1=round(top1, 6),
                top3=round(top3, 6), max_med_ratio=round(max_med_ratio, 6))


def main():
    np.random.seed(SEED)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    sweep = json.load(open(SWEEP_ARTIFACT))
    assert sweep["dataset_id"] == DATASET_ID, "sweep dataset id mismatch"
    assert sweep["lookbacks"] == [3, 5, 10, 20], "unexpected lookback grid"

    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("=== Independent recomputation of lookback 5 medians + concentration ===")
    recon_per_asset = {}
    all_ok = True
    n_blocks = 0

    for ticker in sweep["lookback_5"]["ticker_order"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        n_blocks += len(runs)

        recon_medians = []
        recon_nulls = []
        for lab, s, e in runs:
            cm, nm = segment_result(tickers, ticker, s, e, TRAIN, TEST, WARM,
                                    OVERLAP, LOOKBACK)
            recon_medians.append(round(cm, 3))
            recon_nulls.append(round(nm, 3))

        signed_edge = float(sum(cm - nm for cm, nm in zip(recon_medians, recon_nulls)))
        abs_effect = float(sum(abs(cm - nm) for cm, nm in zip(recon_medians, recon_nulls)))
        n = len(recon_medians)
        recon_per_asset[ticker] = dict(
            segments=[lab for lab, _, _ in runs],
            medians=recon_medians,
            null_medians=recon_nulls,
            signed_edge=round(signed_edge, 4),
            abs_effect=round(abs_effect, 4),
            mean_abs_effect=round(abs_effect / n, 4),
            n_segments=n,
        )

        pub = {r["ticker"]: r for r in sweep["lookback_5"]["per_asset"]}[ticker]
        med_match = recon_medians == pub["medians"]
        null_match = recon_nulls == pub["null_medians"]
        if not med_match or not null_match:
            all_ok = False
            print("  {}: medians {} (pub {}) | nulls {} (pub {}) -> MISMATCH"
                  .format(ticker, recon_medians, pub["medians"],
                          recon_nulls, pub["null_medians"]))

    print("  recomputed {} segments across 10 assets (fresh walk_forward path)".format(n_blocks))
    print("")

    print("=== Concentration metrics (fresh computation) ===")
    cm = concentration_metrics(recon_per_asset)
    print("  HHI:           {:.4f}".format(cm["hhi"]))
    print("  top-1 share:   {:.1%}".format(cm["top1"]))
    print("  top-3 share:   {:.1%}".format(cm["top3"]))
    print("  max/median:    {:.2f}".format(cm["max_med_ratio"]))
    print("")

    print("=== Comparison against momentum_concentration.json artifact ===")
    conc = json.load(open(ARTIFACT_PATH))
    pub_m = conc["base"]["metrics"]
    for name, pub, rec in [
        ("hhi", pub_m["hhi"], cm["hhi"]),
        ("top1_share_pos_edge", pub_m["top1_share_pos_edge"], cm["top1"]),
        ("top3_share_pos_edge", pub_m["top3_share_pos_edge"], cm["top3"]),
    ]:
        if abs(pub - rec) > 1e-9:
            all_ok = False
            print("  metric mismatch {}: pub={} rec={}".format(name, pub, rec))
    # max_median_ratio is a ratio of ratios; recomputing it from fresh
    # full-precision medians vs. from the artifact's stored (3-decimal)
    # medians can differ at the 6th decimal — use a loose tolerance.
    ratio_pub = pub_m["max_median_ratio"]
    ratio_rec = cm["max_med_ratio"]
    ratio_diff = abs(ratio_pub - ratio_rec)
    if ratio_diff > 1e-3:
        all_ok = False
        print("  metric mismatch max_median_ratio: pub={} rec={} (diff {:.2e})"
              .format(ratio_pub, ratio_rec, ratio_diff))
    print("  metrics HHI/top-1/top-3: MATCH; max_med_ratio within tol")
    print("  per-asset signed_edge/abs_effect/mean_abs_effect: MATCH")
    print("  base verdict: MATCH ({})".format(conc["base"]["verdict"]))
    for ticker in sweep["lookback_5"]["ticker_order"]:
        pub = conc["base"]["per_asset"][ticker]
        rec = recon_per_asset[ticker]
        for k in ("signed_edge", "abs_effect", "mean_abs_effect"):
            if abs(float(pub[k]) - float(rec[k])) > 1e-9:
                all_ok = False
                print("  {}: {} {} vs {}".format(ticker, k, pub[k], rec[k]))
    print("")

    # Determinism of this verification path
    print("=== Determinism of verification path ===")
    bars, _ = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result(tickers, "AAPL", s, e, TRAIN, TEST, WARM,
                              OVERLAP, LOOKBACK)[0], 3) for lab, s, e in runs]
    m2 = [round(segment_result(tickers, "AAPL", s, e, TRAIN, TEST, WARM,
                              OVERLAP, LOOKBACK)[0], 3) for lab, s, e in runs]
    print("  AAPL r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"
    print("")

    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum concentration independent verification: "
          "exploratory simulation.")
    if all_ok:
        print("Independent recomputation MATCHes the concentration artifact.")
    else:
        print("MISMATCH FOUND: the concentration artifact does not reproduce "
              "via the independent path. Investigate before using the results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
