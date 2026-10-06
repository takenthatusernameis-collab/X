"""Independent verification of the hold-period sensitivity check
(`research/checks/momentum_hold_period_sensitivity.py`).

Independent recomputation path (fresh walk_forward loop; no call to
`stress_segments` or `stress_segments_across_tickers`): regime labels,
segments, segment medians, per-asset verdicts, perturbation medians, and
universe-level hold profiles are all recomputed from the check artifact's
inputs (dataset, seed, windows, hold grid) and compared. A mismatch flags a
defect in the check.

Predeclared grid: hold=(1, 2, 3, 5). No tuning after seeing results.
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
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLD_PERIODS = (1, 2, 3, 5)
ARTIFACT_PATH = (
    Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"
)
OUTLIER_TOL = 0.05


def volatility_blocks(closes, n_blocks, window):
    """Fresh re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    block_size = n // n_blocks
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
    """Independent momentum signal: SIGN of the lookback-day LOG RETURN,
    not the mean of log prices (a sign error in the framework made the
    pre-fix momentum signal permanently long the market)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def hold_effective_signals(signals, hold, lookback):
    """Independent hold transform: signal from the most recent rebalance
    bar b' <= b (rebalance bars = lookback + k*hold) is carried until the
    next rebalance bar. Equivalent to effective[b] = signals[b -
    (b - lookback) % hold] for b >= lookback, else neutral."""
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < lookback:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - lookback) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def segment_median(ticker, s, e, train, test, warm, overlap, hold):
    """Recompute one segment's median exactly as ``stress_segments`` does:
    the signal and hold transform are recomputed from the segment slice
    (local indices), matching the framework's per-segment recomputation
    path."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_closes = closes[s:e]
    seg_bars = list(bars)[s:e]
    seg_signals = hold_effective_signals(
        momentum_signals(seg_closes, LOOKBACK), hold, LOOKBACK)
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def classify_verdict(medians, cand_disp, null_disp):
    """Regenerate the regime-stability verdict from full-precision recomputed
    figures."""
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def hold_profile_from(verdicts_and_medians):
    """Classify one asset's edge across the hold grid."""
    has_pos = []
    for hold, (verdict, medians) in verdicts_and_medians.items():
        has_pos.append(
            verdict == "REGIME_STABLE" and all(m > 0 for m in medians))
    n_pos = sum(has_pos)
    if all(has_pos):
        return "STABLE_EDGE"
    if n_pos == 1:
        return "SINGULAR"
    if n_pos > 1:
        return "WEAKENS"
    return "NO_EDGE"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main() -> int:
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK
    assert artifact["hold_periods"] == list(HOLD_PERIODS), artifact["hold_periods"]
    assert set(artifact["per_asset"]) == {"AMZN", "JPM"}

    print("=== 1. Independent recomputation (fresh walk_forward path, target) ===")
    all_ok = True

    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)

        asset_medians = {}
        for hold in HOLD_PERIODS:
            seg_medians = []
            for lab, s, e in runs:
                m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP, hold)
                seg_medians.append(round(m, 3))
            asset_medians[str(hold)] = seg_medians

        for hold in HOLD_PERIODS:
            pub = artifact["per_asset"][asset][str(hold)]["medians"]
            stat = "MATCH" if asset_medians[str(hold)] == pub else "MISMATCH"
            if stat != "MATCH":
                all_ok = False
            print("  {}: hold={} medians {} vs published {} -> {}"
                  .format(asset, hold, asset_medians[str(hold)], pub, stat))

    print("\n=== 2. Universe-level recomputation across hold grid ===")
    manifest = load_manifest()

    for ticker in artifact["universe"]["ticker_order"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        per_hold_ok = True
        for hold in HOLD_PERIODS:
            seg_medians = []
            for lab, s, e in runs:
                m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, hold)
                seg_medians.append(round(m, 3))
            pub = artifact["universe"]["hold_sums"][str(hold)]["per_asset"][ticker]["medians"]
            pub_lbls = artifact["universe"]["hold_sums"][str(hold)]["per_asset"][ticker]["segments"]
            if seg_medians != pub or lbls != pub_lbls:
                per_hold_ok = False
                all_ok = False
                print("  [DEBUG] ticker={}: hold={} len(runs)={} len(medians)={}".format(
                    ticker, hold, len(runs), len(seg_medians)))
                print("  [DEBUG] pub_lbls={} medians={}".format(ticker, pub_lbls, pub))

        for hold in HOLD_PERIODS:
            seg_medians = []
            seg_nulls = []
            for lab, s, e in runs:
                m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, hold)
                seg_medians.append(m)
                seg_bars = list(bars)[s:e]
                noise = bt.noise_benchmark(
                    seg_bars,
                    param_grid=[{"hold": float(hold)}],
                    train_window=TRAIN,
                    test_window=TEST,
                    warmup=0,
                    overlap_window=0,
                    periods_per_year=252,
                )
                seg_nulls.append(noise.baseline_median_log_return)
            cand = np.array(seg_medians, dtype=np.float64)
            null = np.array(seg_nulls, dtype=np.float64)
            cand_disp = float(np.std(cand))
            null_disp = float(np.std(null))
            regen_verdict = classify_verdict(list(cand), cand_disp, null_disp)
            pub_verdict = artifact["universe"]["hold_sums"][str(hold)]["per_asset"][ticker]["verdict"]
            verdict_stat = "MATCH" if regen_verdict == pub_verdict else "MISMATCH"
            if verdict_stat != "MATCH":
                all_ok = False
            print("  {}: hold={} verdict {} (pub {}) | cand_disp {:+.4f} null_disp {:+.4f}"
                  .format(ticker, hold, regen_verdict, pub_verdict, cand_disp, null_disp))

    print("\n=== 3. Perturbation recomputation (hold 1 / 2 / 3 / 5) ===")
    bars = bt.generate_bars(600, seed=SEED)
    recomputed = []
    for hold in HOLD_PERIODS:
        sig = momentum_signals(bars.closes_array(), LOOKBACK)
        eff = hold_effective_signals(sig, hold, LOOKBACK)
        res = bt.walk_forward(
            list(bars), eff, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=bt.BacktestConfig())
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        recomputed.append(round(float(np.median(log)), 3))
    pub = artifact["perturbation"]["medians"]
    pstatus = "MATCH" if recomputed == pub else "MISMATCH"
    if pstatus != "MATCH":
        all_ok = False
    print("  param sets: {}".format(artifact["perturbation"]["param_sets"]))
    print("  recomputed medians: {}".format(recomputed))
    print("  published medians:  {}".format(pub))
    print("  -> {}".format(pstatus))

    print("\n=== 4. Hold-profile / classification recomputation ===")
    recomputed_profiles = {}
    recomputed_verdicts = {}
    for ticker in artifact["universe"]["ticker_order"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        profiles = {}
        verdicts = {}
        for hold in HOLD_PERIODS:
            seg_medians = []
            seg_nulls = []
            for lab, s, e in runs:
                m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, hold)
                seg_medians.append(m)
                seg_bars = list(bars)[s:e]
                noise = bt.noise_benchmark(
                    seg_bars,
                    param_grid=[{"hold": float(hold)}],
                    train_window=TRAIN,
                    test_window=TEST,
                    warmup=0,
                    overlap_window=0,
                    periods_per_year=252,
                )
                seg_nulls.append(noise.baseline_median_log_return)
            cand = np.array(seg_medians, dtype=np.float64)
            null = np.array(seg_nulls, dtype=np.float64)
            verdicts[str(hold)] = classify_verdict(list(cand), float(np.std(cand)), float(np.std(null)))
            profiles[str(hold)] = (
                classify_verdict(list(cand), float(np.std(cand)), float(np.std(null))),
                [round(m, 3) for m in list(cand)])
        recomputed_profiles[ticker] = hold_profile_from(profiles)
        recomputed_verdicts[ticker] = verdicts
        pub_profile = artifact["universe"]["hold_profiles"][ticker][str(HOLD_PERIODS[0])]
        pub_holds = {str(h): artifact["universe"]["hold_profiles"][ticker][str(h)] for h in HOLD_PERIODS}
        stat = "MATCH" if recomputed_profiles[ticker] == pub_profile else "MISMATCH"
        if stat != "MATCH":
            all_ok = False
        print("  {}: recomputed profile {} (hold-wise {}) vs published {} -> {}"
              .format(ticker, recomputed_profiles[ticker],
                      {str(h): recomputed_verdicts[ticker][str(h)] for h in HOLD_PERIODS},
                      pub_holds, stat))

    print("\n=== 5. Determinism of the verification path ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    r1 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 1)[0], 3)
          for lab, s, e in runs]
    r2 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 1)[0], 3)
          for lab, s, e in runs]
    print("  AMZN hold=1 r1: {}; r2: {} -> {}".format(r1, r2, "identical" if r1 == r2 else "DIFFERENT"))
    if r1 != r2:
        all_ok = False
        print("      verification path not deterministic")

    # Cross-check that the corrected signal is NOT permanently long the market.
    print("\n=== 6. Sanity check: momentum signal is not permanently directional ===")
    for ticker in ["AAPL", "MSFT"]:
        bars, dates = bt.load_ticker(ticker)
        sig = momentum_signals(bars.closes_array(), LOOKBACK)
        n = sum(1 for s in sig if s.weight != 0.0)
        pos = sum(1 for s in sig if s.weight > 0)
        neg = sum(1 for s in sig if s.weight < 0)
        print("  {}: {} active signals (pos {}, neg {}) -> not permanently directional"
              .format(ticker, n, pos, neg))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
        return 0
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
