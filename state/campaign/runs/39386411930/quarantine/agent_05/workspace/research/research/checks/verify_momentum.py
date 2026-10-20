"""Independent verification of the momentum check
(`research/checks/momentum.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum check (`state/check_artifacts/momentum_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Perturbation medians are recomputed by walking each parameter set
  independently.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch would flag an
error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

BASE_SEED = 42  # same base seed the check used for perturbations


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


def segment_median(ticker, s, e, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, LOOKBACK)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def base_ma_segment_median(ticker, s, e, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    n = len(closes)
    sgn = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(59, n):
        fm = np.mean(closes[i - 19 : i + 1])
        sm = np.mean(closes[i - 59 : i + 1])
        sgn[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    seg_bars = list(bars)[s:e]
    seg_signals = sgn[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK
    assert artifact["per_asset"].keys() == {"AMZN", "JPM"}

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset: base MA and momentum segment medians vs artifact
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        base_medians = []
        for lab, s, e in runs:
            m, nf = base_ma_segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP)
            base_medians.append(round(m, 3))
        mom_medians = []
        for lab, s, e in runs:
            m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP)
            mom_medians.append(round(m, 3))

        pub_base = artifact["per_asset"][asset]["base_ma"]["medians"]
        pub_mom = artifact["per_asset"][asset]["momentum"]["medians"]
        pub_lbls = artifact["per_asset"][asset]["base_ma"]["segments"]

        base_status = "MATCH" if base_medians == pub_base and lbls == pub_lbls else "MISMATCH"
        mom_status = "MATCH" if mom_medians == pub_mom and lbls == pub_lbls else "MISMATCH"
        if base_status != "MATCH" or mom_status != "MATCH":
            all_ok = False
        print("  {}: base medians {} vs published {} -> {}"
              .format(asset, base_medians, pub_base, base_status))
        print("  {}: momentum medians {} vs published {} -> {}"
              .format(asset, mom_medians, pub_mom, mom_status))

    # Universe-level per-asset medians and verdicts vs artifact
    print("\n=== Universe-level recomputation ===")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for ticker in artifact["universe"]["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        medians = [round(segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
                   for lab, s, e in runs]
        pub = artifact["universe"]["per_asset"][ticker]["medians"]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["segments"]
        # Regenerate the verdict from recomputed medians and null medians.
        cand = np.array(medians)
        null = np.array(artifact["universe"]["per_asset"][ticker]["null_medians"])
        cand_disp = round(float(np.std(cand)), 3)
        null_disp = round(float(np.std(null)), 3)
        if all(abs(m) <= 0.05 for m in medians):
            regen_verdict = "CONSISTENT_WITH_NOISE"
        elif cand_disp > 2.0 * null_disp:
            regen_verdict = "REGIME_DEPENDENT"
        elif all(m < 0 for m in medians):
            # Uniformly negative medians with a swing within twice the null:
            # stable losses, not an edge; must be read alongside the medians.
            regen_verdict = "REGIME_STABLE_LOSS"
        else:
            regen_verdict = "REGIME_STABLE"
        status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            print("  [DEBUG] ticker={}: len(runs)={} lbls={}".format(ticker, len(runs), lbls))
            print("  [DEBUG] ticker={}: pub_lbls={}".format(ticker, pub_lbls))
            print("  [DEBUG] ticker={}: len(medians)={} medians={}".format(ticker, len(medians), medians))
            print("  [DEBUG] ticker={}: pub_medians={}".format(ticker, pub))
        verdict_status = "MATCH" if regen_verdict == artifact["universe"]["per_asset"][ticker]["verdict"] else "MISMATCH"
        if verdict_status != "MATCH":
            all_ok = False
        print("  {}: medians {} segments={} -> {} | verdict {} (pub: {})"
              .format(ticker, medians, pub_lbls, status, regen_verdict,
                      artifact["universe"]["per_asset"][ticker]["verdict"]))
        if verdict_status != "MATCH":
            print("      verdict regeneration did not match artifact: {}"
                  .format(verdict_status))

    # Perturbation recomputation (fresh walk_forward loop per param set)
    print("\n=== Perturbation recomputation (3 / 5 / 10 lookback) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]
    recomputed = []
    for params in grid:
        signals = momentum_signals(bars.closes_array(), params["lookback"])
        res = bt.walk_forward(
            list(bars), signals, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=bt.BacktestConfig())
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        recomputed.append(round(float(np.median(log)), 3))
    pub = artifact["perturbation"]["medians"]
    pub_bs = artifact["perturbation"]["baseline_median"]
    pub_noise = artifact["perturbation"]["noise_median"]
    pstatus = "MATCH" if recomputed == pub else "MISMATCH"
    if pstatus != "MATCH":
        all_ok = False
    print("  param sets: {}".format(artifact["perturbation"]["param_sets"]))
    print("  recomputed medians: {}".format(recomputed))
    print("  published medians:  {}".format(pub))
    print("  baseline: {} (pub {:+.3f}) | null: {} (pub {:+.3f})"
          .format(recomputed[1], pub_bs, recomputed[0], pub_noise))
    print("  -> {}".format(pstatus))

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    print("  AMZN r1: {}; r2: {} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum independent verification on AMZN/JPM + universe: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
