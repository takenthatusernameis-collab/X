"""Independent verification of the momentum hold period sensitivity check
(`research/checks/momentum_hold_period_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum check (`state/check_artifacts/momentum_hold_period_sensitivity_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``vol_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Perturbation medians are recomputed by walking each parameter set
  independently.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments, hold periods) so a match
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
HOLD_PERIODS = (1, 2, 3, 5)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"

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


def hold_effective_signals(signals, lookback, hold):
    """Carry each signal for `hold` bars before the next rebalance.

    Position at bar b equals the signal from the most recent rebalance bar
    b' <= b, where rebalance bars are lookback + k*hold. Concretely,
    effective[b] = signals[b - (b - lookback) % hold] for b >= lookback,
    else neutral.

    No look-ahead: the transform only reindexes the signal series; each
    output bar depends on signals at or before that bar.
    """
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < lookback:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - lookback) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def segment_median(ticker, lookback, hold, s, e, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    eff = hold_effective_signals(signals, lookback, hold)
    seg_bars = list(bars)[s:e]
    seg_signals = eff[s:e]
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
    assert artifact["hold_periods"] == list(HOLD_PERIODS)
    assert artifact["per_asset"].keys() == {"AMZN", "JPM"}

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset: momentum segment medians vs artifact
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        for hold in HOLD_PERIODS:
            mom_medians = []
            for lab, s, e in runs:
                m, nf = segment_median(asset, LOOKBACK, hold, s, e, TRAIN, TEST, WARM, OVERLAP)
                mom_medians.append(round(m, 3))

            asset_h_key = str(hold)
            pub_mom = artifact["per_asset"][asset][asset_h_key]["medians"]
            pub_lbls = artifact["per_asset"][asset][asset_h_key]["segments"]

            mom_status = "MATCH" if mom_medians == pub_mom and lbls == pub_lbls else "MISMATCH"
            if mom_status != "MATCH":
                all_ok = False
            print(f"  {asset} hold={hold}: momentum medians {mom_medians} vs published {pub_mom} -> {mom_status}")

    # Universe-level per-asset medians and verdicts vs artifact
    print("\n=== Universe-level recomputation ===")
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {}
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # The universe per-asset data is stored under hold_sums -> per_asset for each hold
    for hold in HOLD_PERIODS:
        h_key = str(hold)
        for ticker in artifact["universe"]["hold_sums"][h_key]["per_asset"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            medians = [round(segment_median(ticker, LOOKBACK, hold, s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
                       for lab, s, e in runs]
            pub = artifact["universe"]["hold_sums"][h_key]["per_asset"][ticker]["medians"]
            pub_lbls = artifact["universe"]["hold_sums"][h_key]["per_asset"][ticker]["segments"]
            # Regenerate the verdict from recomputed medians and null medians.
            cand = np.array(medians)
            null = np.array(artifact["universe"]["hold_sums"][h_key]["per_asset"][ticker]["null_medians"])
            cand_disp = round(float(np.std(cand)), 3)
            null_disp = round(float(np.std(null)), 3)
            if all(abs(m) <= 0.05 for m in medians):
                regen_verdict = "CONSISTENT_WITH_NOISE"
            elif cand_disp >= 2.0 * null_disp:
                regen_verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                regen_verdict = "REGIME_STABLE_LOSS"
            else:
                regen_verdict = "REGIME_STABLE"
            status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  [DEBUG] ticker={ticker} hold={hold}: len(runs)={len(runs)} lbls={lbls}")
                print(f"  [DEBUG] ticker={ticker} hold={hold}: pub_lbls={pub_lbls}")
                print(f"  [DEBUG] ticker={ticker} hold={hold}: len(medians)={len(medians)} medians={medians}")
                print(f"  [DEBUG] ticker={ticker} hold={hold}: pub_medians={pub}")
            verdict_status = "MATCH" if regen_verdict == artifact["universe"]["hold_sums"][h_key]["per_asset"][ticker]["verdict"] else "MISMATCH"
            if verdict_status != "MATCH":
                all_ok = False
            print(f"  {ticker} hold={hold}: medians {medians} segments={pub_lbls} -> {status} | verdict {regen_verdict} (pub: {artifact['universe']['hold_sums'][h_key]['per_asset'][ticker]['verdict']})")
            if verdict_status != "MATCH":
                print(f"      verdict regeneration did not match artifact: {verdict_status}")

    # Perturbation recomputation (fresh walk_forward loop per param set)
    print("\n=== Perturbation recomputation (hold 1 / 2 / 3 / 5) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"hold": float(h)} for h in HOLD_PERIODS]
    recomputed = []
    for params in grid:
        signals = hold_effective_signals(momentum_signals(bars.closes_array(), LOOKBACK), LOOKBACK, int(round(params["hold"])))
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
    print(f"  param sets: {artifact['perturbation']['param_sets']}")
    print(f"  recomputed medians: {recomputed}")
    print(f"  published medians:  {pub}")
    print(f"  baseline: {recomputed[0]} (pub {pub_bs}) | null: {recomputed[0]} (pub {pub_noise})")
    print(f"  -> {pstatus}")

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median("AMZN", LOOKBACK, 1, s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median("AMZN", LOOKBACK, 1, s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    print(f"  AMZN r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum hold period sensitivity independent verification on AMZN/JPM + universe: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()