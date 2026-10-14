"""Independent verification of the momentum_hold_period_sensitivity check
(`research/checks/momentum_hold_period_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median for each holding period from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum_hold_period_sensitivity check
(`state/check_artifacts/momentum_hold_period_sensitivity_results.json`).
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


def hold_effective_signals(signals, hold):
    """Carry each signal for `hold` bars before the next rebalance.

    Position at bar b equals the signal from the most recent rebalance bar
    b' <= b, where rebalance bars are lookback + k*hold. Concretely,
    effective[b] = signals[b - (b - lookback) % hold] for b >= lookback,
    else neutral. This is the standard holding-period semantics: the bet
    taken at bar t is marked to market for bars t .. t+H-1 and exited
    at bar t+hold (or re-entered at the new level).

    No look-ahead: the transform only reindexes the signal series; each
    output bar depends on signals at or before that bar.
    """
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < LOOKBACK:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - LOOKBACK) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_median_hold(ticker, s, e, hold, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, LOOKBACK)
    eff = hold_effective_signals(signals, hold)
    seg_bars = list(bars)[s:e]
    seg_signals = eff[s:e]
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
    assert "AMZN" in artifact["per_asset"]
    assert "JPM" in artifact["per_asset"]
    assert str(HOLD_PERIODS[0]) in artifact["per_asset"]["AMZN"]

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset: segment medians for each hold period vs artifact
    target_assets = ["AMZN", "JPM"]
    for asset in target_assets:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        per_asset_results = {}
        for hold in HOLD_PERIODS:
            medians = []
            for lab, s, e in runs:
                m, nf = segment_median_hold(asset, s, e, hold, TRAIN, TEST, WARM, OVERLAP)
                medians.append(round(m, 3))
            per_asset_results[str(hold)] = medians

        # Compare against published medians
        print(f"\n  {asset} per-asset holding-period medians:")
        for hold in HOLD_PERIODS:
            key = str(hold)
            recomputed = per_asset_results[key]
            published = artifact["per_asset"][asset][key]["medians"]
            status = "MATCH" if recomputed == published else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print(f"    hold={hold}: recomputed {recomputed} vs published {published} -> {status}")

    # Universe-level recomputation
    print("\n=== Universe-level recomputation ===")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    universe_ledger = {}
    for ticker in artifact["universe"]["hold_sums"]:
        universe_ledger[ticker] = {}
        for hold in HOLD_PERIODS:
            hold_key = str(hold)
            medians = []
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]
            for lab, s, e in runs:
                m, nf = segment_median_hold(ticker, s, e, hold, TRAIN, TEST, WARM, OVERLAP)
                medians.append(round(m, 3))

            published_medians = artifact["universe"]["hold_sums"][hold_key]["per_asset"][ticker]["medians"]
            published_null_medians = artifact["universe"]["hold_sums"][hold_key]["per_asset"][ticker]["null_medians"]
            published_verdict = artifact["universe"]["hold_sums"][hold_key]["per_asset"][ticker]["verdict"]

            # Regenerate verdict
            cand = np.array(medians)
            null = np.array(published_null_medians)
            cand_disp = round(float(np.std(cand)), 3)
            null_disp = round(float(np.std(null)), 3)
            if all(abs(m) <= 0.05 for m in medians):
                regen_verdict = "CONSISTENT_WITH_NOISE"
            elif cand_disp > 2.0 * null_disp:
                regen_verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                regen_verdict = "REGIME_STABLE_LOSS"
            else:
                regen_verdict = "REGIME_STABLE"

            status = "MATCH" if medians == published_medians else "MISMATCH"
            verdict_status = "MATCH" if regen_verdict == published_verdict else "MISMATCH"
            if status != "MATCH" or verdict_status != "MATCH":
                all_ok = False
                print(f"  [DEBUG] ticker={ticker} hold={hold} len(runs)={len(runs)} lbls={lbls}")
                print(f"  [DEBUG] ticker={ticker} hold={hold} published_lbls={artifact['universe']['hold_sums'][hold_key]['per_asset'][ticker]['segments']}")
                print(f"  [DEBUG] ticker={ticker} hold={hold} len(medians)={len(medians)} medians={medians}")
                print(f"  [DEBUG] ticker={ticker} hold={hold} published_medians={published_medians}")

            universe_ledger[ticker][hold_key] = {
                "medians": medians,
                "verdict": regen_verdict,
                "status": status,
                "verdict_status": verdict_status
            }
            print(f"    {ticker} hold={hold}: medians {medians} segments={lbls} -> {status} | verdict {regen_verdict} (pub: {published_verdict}) -> {verdict_status}")

    # Perturbation recomputation (synthetic perturbation sweep over hold grid)
    print("\n=== Perturbation recomputation (hold 1 / 2 / 3 / 5) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"hold": 1.0}, {"hold": 2.0}, {"hold": 3.0}, {"hold": 5.0}]
    recomputed = []
    for params in grid:
        signals = hold_effective_signals(momentum_signals(bars.closes_array(), LOOKBACK), params["hold"])
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
    print(f"  param sets: {[('hold', h) for h in HOLD_PERIODS]}")
    print(f"  recomputed medians: {recomputed}")
    print(f"  published medians:  {pub}")
    print(f"  baseline: {recomputed[0]} (pub {pub_bs}) | null: {recomputed[1]} (pub {pub_noise})")
    print(f"  -> {pstatus}")

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    determinism_ok = True
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        m1 = []
        for hold in HOLD_PERIODS:
            medians = []
            for lab, s, e in runs:
                m, nf = segment_median_hold(asset, s, e, hold, TRAIN, TEST, WARM, OVERLAP)
                medians.append(round(m, 3))
            m1.append(medians)
        m2 = []
        for hold in HOLD_PERIODS:
            medians = []
            for lab, s, e in runs:
                m, nf = segment_median_hold(asset, s, e, hold, TRAIN, TEST, WARM, OVERLAP)
                medians.append(round(m, 3))
            m2.append(medians)
        print(f"  {asset} r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
        if m1 != m2:
            determinism_ok = False
    assert determinism_ok, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. Momentum holding-period sensitivity independent verification on AMZN/JPM + universe: exploratory simulation.")


if __name__ == "__main__":
    main()
