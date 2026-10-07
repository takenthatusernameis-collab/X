"""Independent verification of `research/checks/breakout_20day_fixed.py`.

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the breakout fixed check
(`state/check_artifacts/breakout_20day_fixed_results.json`). A mismatch would
flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- The coin-flip null is recomputed through the framework's own
  ``noise_benchmark`` (a separate code path from ``stress_segments``).
- Verdicts are regenerated from the FRESH UNROUNDED medians and nulls via the
  documented rules, so verdicts decided on unrounded numbers are not confused
  by a 3-dp rounding round-trip.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch would flag an error
in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
CANONICAL_LOOKBACK = 20
SWEEP_VERDICT_TOL = 0.05
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "breakout_20day_fixed_results.json"


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


def breakout_signals(closes, lookback):
    """Fresh implementation of the breakout signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        if float(closes[i]) > float(np.max(closes[i - lookback : i])):
            out[i] = bt.Signal(date=i + 1, weight=1.0)
    return out


def segment_result(ticker, s, e, train, test, warm, overlap, lookback):
    """Walk-forward regime-stability result for one asset segment and lookback,
    computed with fresh components: volatility_blocks, segment slicing,
    breakout signals, walk_forward, and the framework noise_benchmark. No call
    to stress_segments."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = breakout_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=bt.BacktestConfig())
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252)
    null_log = noise.baseline_median_log_return
    return (float(np.median(cand_log)), null_log, len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def regime_verdict(medians, null_medians):
    """Regenerate the regime-stability verdict from medians and null medians
    using the same rule as
    research.backtest.stress_segments / stress_segments_across_tickers."""
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= SWEEP_VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def main():
    np.random.seed(42)
    artifact = load_artifact(ARTIFACT_PATH)
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == CANONICAL_LOOKBACK, f"lookback mismatch: {artifact['lookback']} vs {CANONICAL_LOOKBACK}"

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        medians = []
        nulls = []
        for lab, s, e in runs:
            m, n, _ = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, CANONICAL_LOOKBACK)
            medians.append(round(m, 3))
            nulls.append(round(n, 3))
        pub_medians = artifact["per_asset"][asset]["medians"]
        pub_nulls = artifact["per_asset"][asset]["null_medians"]
        status = ("MATCH"
                  if medians == pub_medians
                  and nulls == pub_nulls
                  and lbls == artifact["per_asset"][asset]["segments"]
                  else "MISMATCH")
        if status != "MATCH":
            mismatch = True
            all_ok = False
        print("  {}: segments={} -> {}".format(
            asset, lbls, status))
        if status != "MATCH":
            print("    medians recomputed {} vs published {}".format(medians, pub_medians))
            print("    nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [artifact["per_asset"][asset]["segments"]]))

    print("\n=== Gate 2: fresh medians/nulls for all 10 assets + verdict "
          "regeneration from fresh unrounded values ===")
    print("  This gate recomputes every asset's segment medians and null medians "
          "via the independent walk_forward path, then regenerates the regime "
          "and sensitivity verdicts from the FRESH UNROUNDED values (matching how "
          "the check itself computes verdicts). This avoids a 3-dp rounding "
          "boundary from confusing a verdict that was decided on unrounded "
          "numbers.")
    for ticker in artifact["universe"]["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        medians_ok = True
        verdict_ok = True
        fresh_profiles = {}
        for lb in [CANONICAL_LOOKBACK]:
            pm = []
            pn = []
            fresh_medians = []
            fresh_nulls = []
            for lab, s, e in runs:
                m, n, _ = segment_result(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                pm.append(round(m, 3))
                pn.append(round(n, 3))
                fresh_medians.append(m)
                fresh_nulls.append(n)
            pub_pm = artifact["universe"]["per_asset"][ticker]["medians"]
            pub_pn = artifact["universe"]["per_asset"][ticker]["null_medians"]
            if pm != pub_pm or pn != pub_pn:
                medians_ok = False
                all_ok = False
                print("  [DEBUG] {} lookback {}: fresh rounded medians={} nulls={} vs published medians={} nulls={}"
                      .format(ticker, lb, pm, pn, pub_pm, pub_pn))
            regen = regime_verdict(fresh_medians, fresh_nulls)
            pub = artifact["universe"]["per_asset"][ticker]["verdict"]
            if regen != pub:
                verdict_ok = False
                all_ok = False
                print("  [DEBUG] {} lookback {}: fresh unrounded medians={} nulls={} -> regenerated {} vs published {}"
                      .format(ticker, lb, fresh_medians, fresh_nulls, regen, pub))
            fresh_profiles[str(lb)] = {"fresh_medians": fresh_medians,
                                       "fresh_nulls": fresh_nulls,
                                       "regime_verdict": regen,
                                       "published_verdict": pub,
                                       "medians_ok": pm == pub_pm}
        edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb, p in fresh_profiles.items():
            if (p["regime_verdict"] == "REGIME_STABLE"
                    and all(m > 0 for m in p["fresh_medians"])):
                edge_counts["EDGE"] += 1
            elif p["regime_verdict"] == "REGIME_STABLE_LOSS":
                edge_counts["LOSS"] += 1
            else:
                edge_counts["NO_EDGE"] += 1
        robust = "ROBUST" if edge_counts["EDGE"] >= 2 else (
            "SENSITIVE" if edge_counts["EDGE"] == 1 else "NO_EDGE")
        # No sensitivity_verdict field in our artifact, skip this comparison
        status = "MATCH" if (medians_ok and verdict_ok) else "MISMATCH"
        if not (medians_ok and verdict_ok):
            all_ok = False
        print("  {}: fresh medians/nulls -> {} | regime verdicts -> {} | edge_counts={} -> {}"
              .format(ticker, status, "MATCH" if verdict_ok else "MISMATCH",
                      edge_counts, robust))

    print("\n=== Gate 3: determinism of the verification path ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, CANONICAL_LOOKBACK)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, CANONICAL_LOOKBACK)[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback 20 r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, fresh "
              "regenerated verdicts match the published ones, and the "
              "verification path is deterministic.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. 20-day breakout continuation (fixed) independent "
          "verification on AMZN/JPM + all-asset fresh recomputation + verdict "
          "reproducibility: exploratory simulation.")


if __name__ == "__main__":
    main()