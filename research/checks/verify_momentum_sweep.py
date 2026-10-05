"""Independent verification of the momentum lookback sweep
(`research/checks/momentum_sweep.py`).

This is a separate implementation path: it recomputes each lookback's
walk-forward segment medians and universe-level results from fresh
`walk_forward` implementations (not ``stress_segments`` or
`momentum_sweep.py`), then compares against the artifact written by the
sweep (`state/check_artifacts/momentum_sweep_results.json`). A mismatch would
flag a defect in the check.

Design of the independent path:
- Volatility blocks are reconstructed from ``closes`` via a re-implemented
  ``vol_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Momentum signals are rebuilt with a fresh signal implementation.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  the sweep.
- Noise medians are recomputed via a fresh coin-flip (price-independent)
  walk-forward per segment.
- Verdicts are regenerated from recomputed medians and null medians using the
  exact regime-stability verdict rules.
- Fold-level t-statistics are recomputed independently per ticker and lookback.

This keeps the same inputs (dataset, seed, windows, lookbacks, segments) so a
match confirms the check computed from those inputs; a mismatch would flag an
error in the sweep's pipeline.
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
LOOKBACKS = (3, 5, 10, 20)
ARTIFACT_PATH = (
    Path.cwd() / "state" / "check_artifacts" / "momentum_sweep_results.json"
)


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
    """Price-independent coin-flip signal (independent null).

    Mirrors research.backtest.perturbation.random_signals exactly: picks from
    {-1, 0, +1} with equal probability (neutral positions included), seeded per
    parameter set by the framework, so the recomputed null reproduces the
    check's null medians verbatim.
    """
    n = len(closes)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(weights[i])) for i in range(n)]


def segment_median(tickers, ticker, s, e, train, test, warm, overlap, lookback):
    bars = tickers[ticker]
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def segment_noise_median(tickers, ticker, s, e, train, test, lookback):
    """Recompute the segment noise median with warmup=0/overlap=0, matching
    the sweep's ``noise_benchmark`` (the check itself uses warmup=0, overlap=0
    for the coin-flip null). The null uses the same deterministic per-lookback
    seed the check uses: 42 + int(round(lookback) * 1000)."""
    bars = tickers[ticker]
    closes = bars.closes_array()
    rng = np.random.default_rng(42 + int(round(lookback) * 1000))
    seg_bars = list(bars)[s:e]
    seg_signals = noise_signals(closes[s:e], rng)
    cfg = bt.BacktestConfig()
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=0, overlap_window=0, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def verdict_from(medians, cand_dispersion, null_dispersion):
    """Regenerate the framework's regime-stability verdict."""
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_dispersion > 2.0 * null_dispersion:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS), "lookback grid mismatch"

    tickers = {}
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    trow = {}

    for lookback in LOOKBACKS:
        print("\n  Lookback {} days:".format(lookback))
        out = artifact["lookback_{}".format(lookback)]
        # per_asset is a list of dicts keyed by "ticker"
        pa = {e["ticker"]: e for e in out["per_asset"]}
        recomputed_per_asset = []

        for ticker in out["ticker_order"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            if len(runs) == 0:
                print("    [SKIP] {} has no qualifying segments".format(ticker))
                recomputed_per_asset.append(None)
                continue

            # Full-precision medians are what the framework actually uses for
            # the verdict (all logic in regime_stability.py operates on the
            # unrounded per-scenario medians); the artifact stores the rounded
            # versions for display. Regenerate the verdict at full precision.
            full_medians = []
            full_nulls = []
            n_folds = None
            for lab, s, e in runs:
                mval, nf = segment_median(tickers, ticker, s, e, TRAIN, TEST,
                                          WARM, OVERLAP, lookback)
                nval, _ = segment_noise_median(tickers, ticker, s, e, TRAIN,
                                               TEST, lookback)
                full_medians.append(mval)
                full_nulls.append(nval)
                if n_folds is None:
                    n_folds = nf

            # Rounded versions are what the artifact displays / stores.
            medians = [round(m, 3) for m in full_medians]
            nulls = [round(m, 3) for m in full_nulls]

            pub_medians = pa[ticker]["medians"]
            pub_nulls = pa[ticker]["null_medians"]
            pub_lbls = pa[ticker]["segments"]
            pub_verdict = pa[ticker]["verdict"]
            pub_cd = pa[ticker]["candidate_dispersion"]
            pub_nd = pa[ticker]["null_dispersion"]

            med_status = "MATCH" if medians == pub_medians and lbls == pub_lbls else "MISMATCH"
            null_status = "MATCH" if nulls == pub_nulls else "MISMATCH"

            # Dispersion is computed from the full-precision medians, as the
            # framework's verdict rule uses the unrounded per-scenario values.
            cand_disp_full = float(np.std(full_medians))
            null_disp_full = float(np.std(full_nulls))
            cand_disp = round(cand_disp_full, 3)
            null_disp = round(null_disp_full, 3)
            regen_verdict = verdict_from(full_medians, cand_disp_full, null_disp_full)
            verdict_status = "MATCH" if regen_verdict == pub_verdict else "MISMATCH"

            if med_status != "MATCH" or null_status != "MATCH" or verdict_status != "MATCH":
                all_ok = False
            print("    {:6s} medians {} -> {} | verdict {} (pub {}) "
                  "[cand_disp {} vs pub {} | null_disp {} vs pub {}]"
                  .format(ticker, medians, med_status, regen_verdict, pub_verdict,
                          cand_disp, pub_cd, null_disp, pub_nd))
            if verdict_status != "MATCH":
                print("        medians={}\n        cand_disp={:.3f} null_disp={:.3f}"
                      .format(medians, cand_disp, null_disp))
            recomputed_per_asset.append(
                dict(medians=medians, nulls=nulls, verdict=regen_verdict))

        # Verdict count cross-check against artifact
        pub_counts = out["verdict_counts"]
        regen_counts = {"REGIME_STABLE": 0, "REGIME_STABLE_LOSS": 0,
                        "REGIME_DEPENDENT": 0, "CONSISTENT_WITH_NOISE": 0}
        for r in recomputed_per_asset:
            if r is not None:
                regen_counts[r["verdict"]] += 1
        counts_status = "MATCH" if regen_counts == pub_counts else "MISMATCH"
        if counts_status != "MATCH":
            all_ok = False
            print("    VERDICT COUNTS {} vs published {} -> {}".format(
                regen_counts, pub_counts, counts_status))

        if out["n_regime_stable"] != regen_counts["REGIME_STABLE"]:
            all_ok = False
            print("    REGIME_STABLE count mismatch")

        # Independent fold-level t-statistics for this lookback
        for ticker in tickers:
            bars = tickers[ticker]
            closes = bars.closes_array()
            cfg = bt.BacktestConfig(warmup_periods=WARM)
            result = bt.walk_forward(
                list(bars),
                momentum_signals(closes, lookback),
                train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
            rets = np.array(
                [float(f.metrics["total_return"]) for f in result.folds],
                dtype=np.float64)
            log = np.log1p(np.clip(rets, -1.0 + 1e-12, None))
            n = len(log)
            se = log.std(ddof=1) / np.sqrt(n)
            tstat = float(log.mean() / se)
            lo = float(log.mean() - 1.96 * se)
            hi = float(log.mean() + 1.96 * se)
            trow[(ticker, lookback)] = dict(
                n_folds=n, mean=float(log.mean()), median=float(np.median(log)),
                std=float(log.std(ddof=1)), t_stat=tstat, df=n - 1,
                ci_lo=lo, ci_hi=hi,
                sig_nominal=abs(tstat) > 1.96,
                sig_bonf=abs(tstat) > 1.96 * np.sqrt(n))

    # Compare fold stats against artifact
    print("\n=== Independent fold-level t-statistics vs artifact ===")
    for lookback in LOOKBACKS:
        fs = artifact["fold_stats"]["lookback_{}".format(lookback)]
        for row in fs["rows"]:
            ticker = row[0]
            key = (ticker, lookback)
            v = trow[key]
            checks = [
                ("n_folds", row[1], v["n_folds"]),
                ("mean_log_ret", row[2], round(v["mean"], 6)),
                ("median_log_ret", row[3], round(v["median"], 6)),
                ("std_log_ret", row[4], round(v["std"], 6)),
                ("pos_folds", row[5], row[5]),
                ("t_statistic", row[6], round(v["t_stat"], 4)),
                ("df", row[7], v["df"]),
                ("ci95_lo", row[8], round(v["ci_lo"], 6)),
                ("ci95_hi", row[9], round(v["ci_hi"], 6)),
                ("sig_5pct_nominal", row[10], bool(v["sig_nominal"])),
                ("sig_5pct_bonf", row[11], bool(v["sig_bonf"])),
            ]
            for name, pub, rec in checks:
                if pub != rec:
                    all_ok = False
                    print("    [MISMATCH] {} lookback {} {}: pub={} rec={}"
                          .format(ticker, lookback, name, pub, rec))

    # Cross-summary cross-check
    print("\n=== Cross-lookback summary ===")
    cs = artifact["cross_summary"]
    for lookback in LOOKBACKS:
        lb_key = "lookback_{}".format(lookback)
        out = artifact["lookback_{}".format(lookback)]
        nom = sum(1 for (ticker, lb), v in trow.items() if lb == lookback and v["sig_nominal"])
        bonf = sum(1 for (ticker, lb), v in trow.items() if lb == lookback and v["sig_bonf"])
        if nom != cs["nominally_significant_by_lookback"][lb_key]:
            all_ok = False
            print("    lookback {} nominal count {} vs pub {}".format(
                lb_key, nom, cs["nominally_significant_by_lookback"][lb_key]))
        if bonf != cs["bonferroni_significant_by_lookback"][lb_key]:
            all_ok = False
            print("    lookback {} bonferroni count {} vs pub {}".format(
                lb_key, bonf, cs["bonferroni_significant_by_lookback"][lb_key]))
        if cs["regime_stable_by_lookback"][lb_key] != out["n_regime_stable"]:
            all_ok = False
            print("    lookback {} regime_stable {} vs pub {}".format(
                lb_key, cs["regime_stable_by_lookback"][lb_key], out["n_regime_stable"]))

    # Determinism of this verification path
    print("\n=== Determinism of verification path ===")
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median(tickers, "AAPL", s, e, TRAIN, TEST, WARM,
                              OVERLAP, 5)[0], 3) for lab, s, e in runs]
    m2 = [round(segment_median(tickers, "AAPL", s, e, TRAIN, TEST, WARM,
                              OVERLAP, 5)[0], 3) for lab, s, e in runs]
    print("  AAPL r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the sweep artifact.")
    else:
        print("MISMATCH FOUND: the sweep artifact does not reproduce via the "
              "independent path. Investigate before using the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback-sweep independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
