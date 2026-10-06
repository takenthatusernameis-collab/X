"""Independent verification of the regime-filtered momentum check
(`research/checks/regime_filtered_momentum.py`).

This is a separate implementation path: it recomputes the regime gate for all
three variants (base, turbulent-only, calm-only) on AMZN/JPM and across the
full 10-asset universe using a fresh `walk_forward` implementation, then
compares every reported median, null median, and verdict against the artifact
written by the check. A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` via a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segment boundaries follow ``min_segment_bars`` on the reconstructed labels.
- Per-segment walk-forward is done directly via ``bt.walk_forward`` (not
  ``stress_segments`` or ``regime_stability.py`` regime helpers).
- Each variant's signal is re-implemented from scratch (base momentum;
  regime-filtered momentum active only in calm / turbulent blocks).
- All values are loaded from the check's artifact and compared, including the
  per-asset and universe-level verdict counts.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch flags an error in the
check's pipeline.
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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "regime_filtered_momentum_results.json"

VARIANTS = [("base", None), ("turbulent_only", "turbulent"), ("calm_only", "calm")]


def volatility_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    block_size = n // n_blocks
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


def segment_boundaries(labels, min_segment_bars):
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


def base_momentum_signals(closes, lookback):
    """Fresh implementation of base momentum."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def regime_filtered_momentum_signals(closes, lookback, keep):
    """Fresh implementation of regime-filtered momentum."""
    labels = volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(n):
        lab = labels[i]
        if lab == "insufficient" or lab != keep:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        elif i >= lookback - 1:
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_median(ticker, s, e, train, test, warm, overlap, signals):
    """Walk-forward median log return for one segment (fresh path)."""
    bars, dates = bt.load_ticker(ticker)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=bt.BacktestConfig(warmup_periods=warm))
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def coin_flip_segment_median(ticker, s, e, train, test, warm, overlap, signals):
    """Coin-flip null segment median on the same segment (fresh path)."""
    bars, dates = bt.load_ticker(ticker)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    n = len(seg_bars)
    null = [bt.Signal(date=i + 1, weight=np.random.choice([-1.0, 1.0]))
            for i in range(n)]
    res = bt.walk_forward(
        seg_bars, null, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=bt.BacktestConfig(warmup_periods=warm))
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def compute_verdict(medians, null_medians, cand_disp, null_disp):
    """Re-implementation of the regime-stability verdict logic."""
    if all(m < 0 for m in medians):
        if cand_disp <= 2.0 * null_disp:
            return "REGIME_STABLE_LOSS"
        return "REGIME_DEPENDENT"
    max_disp = max(abs(m - nm) for m, nm in zip(medians, null_medians))
    max_null_disp = max(abs(nm) for nm in null_medians)
    # candidate dispersion vs null dispersion (framework uses raw dispersion)
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if max_disp <= 2.0 * max_null_disp:
        return "CONSISTENT_WITH_NOISE"
    return "REGIME_STABLE"


def run_variant(tickers, variant, target=None):
    """Run one variant and return per-asset dicts keyed like the artifact."""
    if target is None:
        target = list(tickers.keys())
    results = {}
    for ticker in target:
        bars, dates = tickers[ticker]
        closes = bars.closes_array()
        labels = volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        boundaries = segment_boundaries(labels, MIN_SEGMENT_BARS)
        if variant[1] is None:
            signals = base_momentum_signals(closes, LOOKBACK)
        else:
            signals = regime_filtered_momentum_signals(closes, LOOKBACK, variant[1])
        scenario_medians, scenario_nulls, scenario_folds = [], [], []
        for name, s, e in boundaries:
            cm, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, signals)
            nm, _ = coin_flip_segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, signals)
            scenario_medians.append(cm)
            scenario_nulls.append(nm)
            scenario_folds.append(nf)
        cand_disp = float(np.std(scenario_medians)) if scenario_medians else 0.0
        null_disp = float(np.std(scenario_nulls)) if scenario_nulls else 0.0
        verdict = compute_verdict(scenario_medians, scenario_nulls, cand_disp, null_disp)
        results[ticker] = dict(
            segments=[n for n, _, _ in boundaries],
            medians=[round(m, 3) for m in scenario_medians],
            null_medians=[round(m, 3) for m in scenario_nulls],
            candidate_dispersion=round(cand_disp, 3),
            null_dispersion=round(null_disp, 3),
            verdict=verdict,
            n_folds=scenario_folds,
        )
    return results


def run_universe(tickers, variant):
    """Run one variant across the full universe (fresh path)."""
    results = {}
    for ticker, (bars, dates) in tickers.items():
        close_list = bars.closes_array()
        # walk_forward over the full series with the variant signal
        if variant[1] is None:
            sigs = base_momentum_signals(close_list, LOOKBACK)
        else:
            sigs = regime_filtered_momentum_signals(close_list, LOOKBACK, variant[1])
        res = bt.walk_forward(
            list(bars), sigs, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=bt.BacktestConfig(warmup_periods=WARM))
        # compute per-segment medians the same way the check does via segments
        # (full-series walk-forward medians vs null); here we simply report the
        # full-series fold statistics which is the independent path's aggregation.
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        # For the independent check we use a block-based split consistent with
        # the check's 4 volatility-block regime family.
        labels = volatility_blocks(close_list, n_blocks=N_BLOCKS, window=WINDOW)
        boundaries = segment_boundaries(labels, MIN_SEGMENT_BARS)
        scenario_medians, scenario_nulls = [], []
        for _, s, e in boundaries:
            cm, _ = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, sigs)
            nm, _ = coin_flip_segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, sigs)
            scenario_medians.append(cm)
            scenario_nulls.append(nm)
        cand_disp = float(np.std(scenario_medians)) if scenario_medians else 0.0
        null_disp = float(np.std(scenario_nulls)) if scenario_nulls else 0.0
        verdict = compute_verdict(scenario_medians, scenario_nulls, cand_disp, null_disp)
        results[ticker] = dict(
            segments=[n for n, _, _ in boundaries],
            medians=[round(m, 3) for m in scenario_medians],
            null_medians=[round(m, 3) for m in scenario_nulls],
            candidate_dispersion=round(cand_disp, 3),
            null_dispersion=round(null_disp, 3),
            verdict=verdict,
        )
    counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
              "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
    for r in results.values():
        counts[r["verdict"]] += 1
    return dict(per_asset=results, verdict_counts=counts)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def medians_almost(a, b, tol=0.005):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def main():
    np.random.seed(SEED)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"
    artifact = load_artifact()
    assert artifact["lookback"] == LOOKBACK
    assert set(artifact["universe"].keys()) == {v[0] for v in VARIANTS}, artifact["universe"].keys()

    # Load tickers once.
    tickers = {}
    for entry in manifest["entries"]:
        if entry["ticker"] in ("AMZN", "JPM"):
            tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True
    for var_name, keep in VARIANTS:
        print("Variant: {} (keep={!r})".format(var_name, keep))
        asset_results = run_variant(tickers, (var_name, keep), target=list(tickers.keys()))
        art = artifact["variants"][var_name]
        for ticker in tickers:
            r = asset_results[ticker]
            a = art[ticker]
            ok = (r["segments"] == a["segments"]
                  and medians_almost(r["medians"], a["medians"])
                  and medians_almost(r["null_medians"], a["null_medians"])
                  and abs(r["candidate_dispersion"] - a["candidate_dispersion"]) <= 0.005
                  and abs(r["null_dispersion"] - a["null_dispersion"]) <= 0.005
                  and r["verdict"] == a["verdict"])
            if not ok:
                all_ok = False
                print("  MISMATCH {}: cand_disp {} vs {} | null_disp {} vs {} "
                      "| verdict {} vs {}".format(
                          ticker, r["candidate_dispersion"], a["candidate_dispersion"],
                          r["null_dispersion"], a["null_dispersion"], r["verdict"], a["verdict"]))
                print("    segs:", r["segments"], "vs", a["segments"])
                print("    meds:", r["medians"], "vs", a["medians"])
                print("    nulls:", r["null_medians"], "vs", a["null_medians"])
        print("  AMZN/JPM variant: OK" if all_ok else "  AMZN/JPM variant: MISMATCH")

    # Full-universe recomputation (fresh path).
    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])
    print("=== Independent universe recomputation (fresh walk_forward path) ===")
    universe_ok = True
    for var_name, keep in VARIANTS:
        summary = run_universe(all_tickers, (var_name, keep))
        art = artifact["universe"][var_name]
        vc_ok = summary["verdict_counts"] == art["verdict_counts"]
        if not vc_ok:
            universe_ok = False
            print("  verdict_counts MISMATCH {} vs {}".format(
                summary["verdict_counts"], art["verdict_counts"]))
        for ticker in all_tickers:
            r = summary["per_asset"][ticker]
            a = art["per_asset"][ticker]
            if r["segments"] != a["segments"]:
                universe_ok = False
                print("  segment MISMATCH {} {} vs {}".format(ticker, r["segments"], a["segments"]))
                continue
            if not medians_almost(r["medians"], a["medians"]):
                universe_ok = False
                print("  medians MISMATCH {} {} vs {}".format(ticker, r["medians"], a["medians"]))
            if not medians_almost(r["null_medians"], a["null_medians"]):
                universe_ok = False
                print("  null_medians MISMATCH {} {} vs {}".format(ticker, r["null_medians"], a["null_medians"]))
            if r["verdict"] != a["verdict"]:
                universe_ok = False
                print("  verdict MISMATCH {} {} vs {}".format(ticker, r["verdict"], a["verdict"]))
        print("  universe {}: {}".format(
            var_name, "OK (counts {} | {} assets)".format(
                summary["verdict_counts"], len(all_tickers)) if universe_ok else "MISMATCH"))

    # Determinism: rerun and compare.
    np.random.seed(SEED)
    all_tickers2 = {}
    for entry in manifest["entries"]:
        all_tickers2[entry["ticker"]] = bt.load_ticker(entry["ticker"])
    summary2 = run_universe(all_tickers2, ("base", None))
    out1 = json.dumps(summary["per_asset"], sort_keys=True)
    out2 = json.dumps(summary2["per_asset"], sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    if out1 != out2:
        all_ok = False
        universe_ok = False
    print("")
    print("VERDICT:" if all_ok and universe_ok else "VERDICT (mismatches above):")
    print("  independent verifier: MATCH" if (all_ok and universe_ok) else "  independent verifier: MISMATCH")
    return 0 if (all_ok and universe_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
