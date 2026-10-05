"""Independent verification of the cross-sectional relative-strength check
(`research/checks/cross_sectional_relative_strength.py`).

This is a separate implementation path: it re-implements the regime labels,
segment reconstruction, the CSRS spread, and the fold-log-return aggregation
from raw tickers (no calls to `regime_stability.py` helper functions or the
check's own spread implementation), then compares against the artifact written
by the check (`state/check_artifacts/cross_sectional_relative_strength_results.json`).
A mismatch would flag a defect in the check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.real_data import load_ticker
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 20
TOP_K = BOTTOM_K = 3
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
MARKET_PROXY = "AAPL"
ARTIFACT_PATH = (Path.cwd() / "state" / "check_artifacts"
                 / "cross_sectional_relative_strength_results.json")


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks (fresh code)."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window: i]), ddof=1))
                * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = (float(np.median([v for v in bar_vols[s:e] if v > 0]))
                    if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        segments.append((cur, start, len(labels)))
    return segments


def csrs_spread(tickers, lookback, top_k, bottom_k):
    """Fresh re-implementation of the CSRS spread (not the check's helper)."""
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        rets = {t: float(tb[i] / tb[i - lookback] - 1) for t, tb in tickers.items()}
        order = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = order[:top_k]
        shorts = order[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        long_ret = float(np.mean([tb[i + 1] / tb[i] - 1 for t, tb in
                                  tickers.items() if t in longs]))
        short_ret = float(np.mean([tb[i + 1] / tb[i] - 1 for t, tb in
                                   tickers.items() if t in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl


def walk_forward_fold_log_returns(daily_pnl, train, test, warm, overlap):
    """Mirror of research.checks.cw_cross_sectional_relative_strength's
    walk_forward_fold_log_returns (fresh code)."""
    n = len(daily_pnl)
    step = test - overlap
    fold_start = 0
    lrets = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n:
            break
        oos = daily_pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets


def regime_verdict(candidate_medians, null_medians):
    """Mirror of the check's regime_verdict."""
    if all(abs(m) <= 0.05 for m in candidate_medians):
        return "CONSISTENT_WITH_NOISE"
    cand_disp = float(np.std(candidate_medians, ddof=1))
    null_disp = float(np.std(null_medians, ddof=1))
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    return "REGIME_STABLE"


def main() -> int:
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    tickers = {}
    for e in manifest["entries"]:
        tickers[e["ticker"]] = load_ticker(e["ticker"])[0].closes_array()
    trunc = min(len(tb) for tb in tickers.values())
    tickers = {t: tb[:trunc] for t, tb in tickers.items()}

    aapl_closes = tickers[MARKET_PROXY]
    aapl_labels = vol_blocks(aapl_closes, N_BLOCKS, WINDOW)
    aapl_segments = segments_from_labels(aapl_labels, MIN_SEGMENT_BARS)
    seg_names = [name for name, _, _ in aapl_segments]

    print("regime labels (recomputed from raw {}): {}".format(
        MARKET_PROXY, len(aapl_segments)))

    def seg_medians(pnl):
        meds = []
        for _, s, e in aapl_segments:
            meds.append(float(np.median(
                walk_forward_fold_log_returns(pnl[s:e], TRAIN, TEST, WARM, OVERLAP))))
        return meds

    daily_pnl = csrs_spread(tickers, LOOKBACK, TOP_K, BOTTOM_K)
    cand = seg_medians(daily_pnl)
    cand_disp = float(np.std(cand, ddof=1))

    # Independent null: same ranks, coin-flip sign per bar with the same
    # seed and the same per-bar RNG position as the check's null helper
    # (draws begin at bar lookback, increasing).
    rng = np.random.default_rng(SEED)
    signs = np.zeros(len(daily_pnl))
    for i in range(LOOKBACK, len(daily_pnl) - 1):
        signs[i] = 1.0 if rng.random() < 0.5 else -1.0

    def seg_medians_signed(pnl):
        meds = []
        for _, s, e in aapl_segments:
            meds.append(float(np.median(
                walk_forward_fold_log_returns(pnl[s:e], TRAIN, TEST, WARM, OVERLAP))))
        return meds

    null = np.array(seg_medians_signed(daily_pnl * signs))
    null_disp = float(np.std(null, ddof=1))

    # Load the check's artifact and compare.
    artifact = json.load(open(ARTIFACT_PATH))
    reg = artifact["regime_gate"]

    m1 = np.allclose(cand, reg["segment_medians"], atol=1e-9)
    m2 = np.allclose(null, reg["null_medians"], atol=1e-9)
    print("recomputed candidate medians : {}".format([round(x, 5) for x in cand]))
    print("artifact   candidate medians : {}".format([round(x, 5) for x in reg["segment_medians"]]))
    print("recomputed null medians      : {}".format([round(x, 5) for x in null]))
    print("artifact   null medians      : {}".format([round(x, 5) for x in reg["null_medians"]]))
    print("segment labels match         : {}".format(
        seg_names == reg["segments"]))
    print("candidate medians match      : {}".format(m1))
    print("null medians match           : {}".format(m2))
    print("candidate dispersion {:+.3f} vs null dispersion {:+.3f}"
          .format(cand_disp, null_disp))
    print("verdicts: recomputed={} artifact={}".format(
        regime_verdict(cand, null), reg["verdict"]))

    # Per-sub-universe (drop-one) counts.
    vc = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0, "REGIME_DEPENDENT": 0}
    per_asset = {}
    for ticker in tickers:
        pnl = csrs_spread({k: v for k, v in tickers.items() if k != ticker},
                          LOOKBACK, TOP_K, BOTTOM_K)
        meds = seg_medians(pnl)
        v = regime_verdict(meds, null)
        vc[v] += 1
        per_asset[ticker] = v
    print("per-asset verdict counts : {}, {}, {}".format(
        vc["CONSISTENT_WITH_NOISE"], vc["REGIME_STABLE"], vc["REGIME_DEPENDENT"]))
    print("per-asset verdicts       : {}".format(per_asset))
    m3 = (vc == artifact["per_sub_universe"]["verdict_counts"]
          and per_asset == artifact["per_sub_universe"]["per_asset"])
    print("per-sub-universe counts match: {}".format(m3))

    # Concentration: full-universe median and null tolerance.
    full_lrets = walk_forward_fold_log_returns(daily_pnl, TRAIN, TEST, WARM, OVERLAP)
    full_med = float(np.median(full_lrets))
    null_tol = 2 * float(np.std(full_lrets, ddof=1)) / np.sqrt(len(full_lrets))
    print("full-universe median {:+.4f}; artifact {:+.4f}".format(full_med,
              artifact["concentration"]["full_universe_median"]))
    print("null tolerance {:+.4f}; artifact {:+.4f}".format(null_tol,
              artifact["concentration"]["null_tolerance"]))
    print("concentration median match: {}".format(np.isclose(full_med,
          artifact["concentration"]["full_universe_median"], atol=1e-9)))

    overall = m1 and m2 and seg_names == reg["segments"] and m3
    print("")
    print("INDEPENDENT VERIFICATION: {} (all primary path values match the "
          "check artifact)".format("MATCH" if overall else "MISMATCH"))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
