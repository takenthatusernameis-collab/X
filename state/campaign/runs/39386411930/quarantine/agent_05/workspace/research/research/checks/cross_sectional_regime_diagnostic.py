"""Diagnose whether a cross-sectional dispersion classifier separates the
2009-2013 "good turbulent" regime from the 2022-2026 "bad turbulent" regime.

The MA-crossover class was REGIME_DEPENDENT on AMZN/JPM because its edge lived
in the 2009-2013 block; the base volatility classifier could not separate the
two turbulent regimes (both got "turbulent"). This diagnostic tests a richer,
cross-sectional regime feature: at each bar, the cross-sectional dispersion of
same-day simple returns across the collected universe (past-only by
construction). Question: does this label the 2009-2013 block differently from
2022-2026?

Method:
- Collect the 10-ticker adjusted-close universe.
- Compute per-bar (a) AAPL trailing-60d realized volatility and (b)
  cross-sectional dispersion = std of same-day returns across tickers.
- Classify each bar as turbulent when the feature exceeds its series median
  (past-only).
- Report the fraction of "turbulent" bars in the 2009-2013 and 2022-2026
  blocks under each classifier, using actual trading dates in the CSV.
- Independently recompute cross-sectional dispersion via a pairwise-absolute
  returns mean and confirm the block-label conclusion is unchanged.
- Optionally run MA(20/60) through the framework's stress_segments with the
  cross-sectional labels on AMZN and JPM and report the verdict.

No OOS tuning; fully deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

SEED = 42
WINDOW = 60
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
MIN_SEGMENT_BARS = 400
BLOCK_A = ("2009-01-01", "2014-01-01")   # "good turbulent" window
BLOCK_B = ("2022-01-01", "2026-10-03")   # "bad turbulent" window


def load_full_tickers() -> dict:
    manifest = load_manifest()
    tickers = {}
    for e in manifest["entries"]:
        tb = bt.load_ticker(e["ticker"])[0]
        tickers[e["ticker"]] = np.asarray(tb.closes_array())
    return tickers


def cross_sectional_dispersion(tickers):
    """Past-only cross-sectional dispersion of same-day simple returns.

    For bar i, the return closes[i]/closes[i-1] - 1 is realized between
    bars i-1 and i, so it is available at bar i (no look-ahead).
    """
    n = len(next(iter(tickers.values())))
    disp = np.zeros(n)
    for i in range(1, n):
        rets = [float(tb[i] / tb[i - 1] - 1) for tb in tickers.values()]
        if len(rets) == len(tickers):
            disp[i] = float(np.std(rets, ddof=1))
    return disp


def pairwise_dispersion(tickers):
    """Independent recomputation: mean of pairwise |same-day return|
    differences."""
    n = len(next(iter(tickers.values())))
    disp = np.zeros(n)
    for i in range(1, n):
        rets = [float(tb[i] / tb[i - 1] - 1) for tb in tickers.values()]
        if len(rets) == len(tickers):
            vals = [abs(rets[j] - rets[k])
                    for j in range(len(rets)) for k in range(j + 1, len(rets))]
            disp[i] = float(np.mean(vals))
    return disp


def smooth(series, window=WINDOW):
    """Rolling-window mean of `series` (past-only: window ends at i)."""
    n = len(series)
    out = np.zeros(n)
    for i in range(window - 1, n):
        out[i] = float(np.mean(series[i - window + 1:i + 1]))
    return out


def cs_volatility_blocks(closes, tickers, n_blocks=4, smooth_window=60):
    """Contiguous-block regime classification using smoothed cross-sectional
    dispersion instead of single-asset realized volatility.

    Same contract as research.backtest.volatility_blocks: split into
    ``n_blocks`` contiguous blocks and label each 'turbulent' when its
    (smoothed cross-sectional dispersion) median exceeds the series-wide
    median, else 'calm'. Bars before the smoothing window are 'insufficient'
    and dropped as a short leading segment.
    """
    n = len(closes)
    block_size = n // n_blocks
    cs = cross_sectional_dispersion(tickers)
    cs_smooth = smooth(cs, smooth_window)
    active = cs_smooth[cs_smooth > 0]
    if len(active) == 0:
        raise ValueError("not enough bars to compute smoothed dispersion")
    series_med = float(np.median(active))
    labels: List[str] = []
    for i in range(n):
        if i < smooth_window:
            labels.append("insufficient")
        else:
            block = i // block_size
            s, e = block * block_size, (block + 1) * block_size
            bmed = float(np.median(cs_smooth[s:e])) if e - s > smooth_window else 0.0
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def realized_vol(closes, window=WINDOW):
    rv = np.zeros(len(closes))
    for i in range(window, len(closes)):
        rv[i] = float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0)
    return rv


def frac_turbulent(feature, threshold, i0, i1):
    seg = feature[i0:i1]
    return float(np.mean(seg > threshold)), int((seg > threshold).sum()), int(len(seg))


def find_block_indices(dates):
    """Return (i0, i1) for [start, end) using actual trading dates."""

    def first_at_or_after(target):
        for j, d in enumerate(dates):
            if d >= target:
                return j
        return len(dates)

    return first_at_or_after(BLOCK_A[0]), first_at_or_after(BLOCK_A[1]), \
           first_at_or_after(BLOCK_B[0]), first_at_or_after(BLOCK_B[1])


def load_dates():
    dates_path = Path("research/data/raw/AAPL_daily.csv")
    dates = []
    with open(dates_path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            dates.append(parts[0] if len(parts) >= 1 else "")
    return dates


def main() -> int:
    tickers = load_full_tickers()
    all_n = min(len(tb) for tb in tickers.values())
    tickers = {t: tb[:all_n] for t, tb in tickers.items()}

    # --- feature series (all past-only) ---
    aapl_closes = tickers["AAPL"]
    aapl_rv = realized_vol(aapl_closes)
    cs_disp = cross_sectional_dispersion(tickers)
    cs_disp_pairwise = pairwise_dispersion(tickers)

    aapl_rv_active = aapl_rv[aapl_rv > 0]
    series_med_rv = float(np.median(aapl_rv_active))
    series_med_cs = float(np.median(cs_disp[cs_disp > 0]))
    series_med_cs_pa = float(np.median(cs_disp_pairwise[cs_disp_pairwise > 0])) \
        if (cs_disp_pairwise > 0).any() else 0.0
    cs_disp_smooth = smooth(cs_disp)
    cs_disp_pa_smooth = smooth(cs_disp_pairwise)
    series_med_cs_smooth = float(np.median(cs_disp_smooth[cs_disp_smooth > 0]))
    series_med_cs_pa_smooth = float(np.median(cs_disp_pa_smooth[cs_disp_pa_smooth > 0])) \
        if (cs_disp_pa_smooth > 0).any() else 0.0

    print("=== 1. Feature distributions (window length = {})".format(all_n))
    print("  AAPL trailing-60d realized vol: active bars={}, "
          "series median = {:.3f}".format(len(aapl_rv_active), series_med_rv))
    print("  Cross-sectional dispersion (std of same-day returns): "
          "series median = {:.4f}".format(series_med_cs))
    print("  Cross-sectional dispersion (pairwise |diff| mean, "
          "independent): series median = {:.4f}".format(series_med_cs_pa))
    print("  Smoothed cross-sec dispersion (60-bar rolling mean): "
          "series median = {:.4f}".format(series_med_cs_smooth))
    print("  Smoothed pairwise dispersion (60-bar rolling mean): "
          "series median = {:.4f}".format(series_med_cs_pa_smooth))
    print()

    dates = load_dates()
    i0a, i1a, i0b, i1b = find_block_indices(dates)
    print("  block A indices: {} (date {}..{}), block B indices: "
          "{} (date {}..{})".format(
              i0a, dates[i0a], dates[i1a - 1] if i1a > i0a else "-",
              i0b, dates[i0b], dates[i1b - 1] if i1b > i0b else "-"))
    print()

    print("=== 2. Block classification: fraction of bars labeled 'turbulent' ===")
    print("  Block 2009-2013 (bar {}..{}):".format(i0a, i1a - 1))
    fr_a_rv = frac_turbulent(aapl_rv, series_med_rv, i0a, i1a)
    fr_a_cs = frac_turbulent(cs_disp, series_med_cs, i0a, i1a)
    fr_a_cs_pa = frac_turbulent(cs_disp_pairwise, series_med_cs_pa, i0a, i1a)
    print("    volatility classifier : {:.2%} turbulent ({} / {})".format(
        fr_a_rv[0], fr_a_rv[1], fr_a_rv[2]))
    print("    cross-sec dispersion  : {:.2%} turbulent ({} / {})".format(
        fr_a_cs[0], fr_a_cs[1], fr_a_cs[2]))
    print("    pairwise dispersion   : {:.2%} turbulent ({} / {})".format(
        fr_a_cs_pa[0], fr_a_cs_pa[1], fr_a_cs_pa[2]))
    print("  Block 2022-2026 (bar {}..{}):".format(i0b, i1b - 1))
    fr_b_rv = frac_turbulent(aapl_rv, series_med_rv, i0b, i1b)
    fr_b_cs = frac_turbulent(cs_disp, series_med_cs, i0b, i1b)
    fr_b_cs_pa = frac_turbulent(cs_disp_pairwise, series_med_cs_pa, i0b, i1b)
    print("    volatility classifier : {:.2%} turbulent ({} / {})".format(
        fr_b_rv[0], fr_b_rv[1], fr_b_rv[2]))
    print("    cross-sec dispersion  : {:.2%} turbulent ({} / {})".format(
        fr_b_cs[0], fr_b_cs[1], fr_b_cs[2]))
    print("    pairwise dispersion   : {:.2%} turbulent ({} / {})".format(
        fr_b_cs_pa[0], fr_b_cs_pa[1], fr_b_cs_pa[2]))
    print("  Block 2009-2013 (smoothed cs labels):")
    fr_a_cs_s = frac_turbulent(cs_disp_smooth, series_med_cs_smooth, i0a, i1a)
    fr_a_cs_pa_s = frac_turbulent(cs_disp_pa_smooth, series_med_cs_pa_smooth, i0a, i1a)
    print("    smoothed cross-sec disp : {:.2%} turbulent ({} / {})".format(
        fr_a_cs_s[0], fr_a_cs_s[1], fr_a_cs_s[2]))
    print("    smoothed pairwise disp  : {:.2%} turbulent ({} / {})".format(
        fr_a_cs_pa_s[0], fr_a_cs_pa_s[1], fr_a_cs_pa_s[2]))
    print("  Block 2022-2026 (smoothed cs labels):")
    fr_b_cs_s = frac_turbulent(cs_disp_smooth, series_med_cs_smooth, i0b, i1b)
    fr_b_cs_pa_s = frac_turbulent(cs_disp_pa_smooth, series_med_cs_pa_smooth, i0b, i1b)
    print("    smoothed cross-sec disp : {:.2%} turbulent ({} / {})".format(
        fr_b_cs_s[0], fr_b_cs_s[1], fr_b_cs_s[2]))
    print("    smoothed pairwise disp  : {:.2%} turbulent ({} / {})".format(
        fr_b_cs_pa_s[0], fr_b_cs_pa_s[1], fr_b_cs_pa_s[2]))
    print()

    sep_vol = abs(fr_a_rv[0] - fr_b_rv[0]) < 0.01
    sep_cs = abs(fr_a_cs[0] - fr_b_cs[0]) >= 0.01
    sep_cs_pa = abs(fr_a_cs_pa[0] - fr_b_cs_pa[0]) >= 0.01
    sep_cs_s = abs(fr_a_cs_s[0] - fr_b_cs_s[0]) >= 0.01
    sep_cs_pa_s = abs(fr_a_cs_pa_s[0] - fr_b_cs_pa_s[0]) >= 0.01
    print("=== 3. Can the classifiers separate the two turbulent blocks? ===")
    print("  volatility classifier separates blocks: {}".format(not sep_vol))
    print("  cross-sec dispersion separates blocks:  {}".format(sep_cs))
    print("  (independent pairwise measure agrees):  {}".format(sep_cs_pa))
    print("  smoothed cross-sec dispersion separates: {}".format(sep_cs_s))
    print("  (independent pairwise agrees):          {}".format(sep_cs_pa_s))
    print()

    # --- MA(20/60) through stress_segments with block-based cs dispersion labels ---
    print("=== 4. MA(20/60) through stress_segments on AMZN/JPM (cs-block labels) ===")
    for asset in ["AMZN", "JPM"]:
        bars_full = bt.load_ticker(asset)[0]
        closes_a = np.asarray(bars_full.closes_array()[:all_n])
        bars = bt.BarSequence(
            bars_full.dates[:all_n], bars_full.opens[:all_n], bars_full.highs[:all_n],
            bars_full.lows[:all_n], bars_full.closes[:all_n], bars_full.volumes[:all_n])
        labels = cs_volatility_blocks(closes_a, tickers, n_blocks=4, smooth_window=60)
        seg_fn = bt.segment_fn_from_labels(labels)
        signals = bt.ma_crossover_signals(closes_a, 20, 60)
        grid = [{"fast": 20, "slow": 60}]
        res = bt.stress_segments(
            bt.ma_crossover_signals, bars, signals, grid,
            (("fast", 20.0), ("slow", 60.0)), seg_fn,
            train_window=252, test_window=84, warmup=60,
            overlap_window=60, min_segment_bars=MIN_SEGMENT_BARS)
        print("  {}: segments=[{}]".format(
            asset, ", ".join(s.name for s in res.scenarios)))
        print("    candidate medians: [{}]  verdict: {}".format(
            ", ".join("{:+.3f}".format(s.baseline_median_log_return)
                      for s in res.scenarios), res.overall_verdict))

    print()
    print("research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
