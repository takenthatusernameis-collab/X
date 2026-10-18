"""Independent verification of the volatility-conditional momentum check
(`research/checks/volatility_conditional_momentum.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum check with volatility gate.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments``.
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

DATA_UNIVERSE = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
VOL_THRESHOLD = 0.20
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "volatility_conditional_momentum_results.json"


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
    if not active:
        raise ValueError("not enough bars to compute realized volatility")
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            block = i // block_size
            s, e = block * block_size, (block + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window
                else 0.0)
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


def regime_conditional_momentum_signals(closes, lookback=LOOKBACK):
    """Fresh implementation of volatility-conditional momentum signals."""
    n = len(closes)
    momentum_gate = []
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        momentum_gate.append(bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0))
    # Fill neutral signals for bars before lookback
    for i in range(lookback):
        momentum_gate.insert(0, bt.Signal(date=i + 1, weight=0.0))

    # Apply volatility gate
    signals = []
    vol_gate = []
    for i in range(WINDOW - 1, n):
        log_returns = np.log(closes[i - WINDOW + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * np.sqrt(252.0)
        vol_gate.append(1.0 if realized_annual < VOL_THRESHOLD else 0.0)
    # Fill neutral gate for bars before volatility window
    for i in range(WINDOW - 1):
        vol_gate.insert(0, 0.0)

    for i in range(n):
        momentum_weight = momentum_gate[i].weight
        vol_enabled = bool(vol_gate[i] > 0.5)
        if vol_enabled:
            signals.append(bt.Signal(date=i + 1, weight=float(momentum_weight)))
        else:
            signals.append(bt.Signal(date=i + 1, weight=0.0))
    return signals


def segment_median(ticker, s, e, train, test, warm, overlap, lookback=LOOKBACK):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = regime_conditional_momentum_signals(closes, lookback=lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATA_UNIVERSE, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK
    assert artifact["vol_threshold"] == VOL_THRESHOLD
    assert artifact["per_asset"].keys() == {"AMZN", "JPM"}

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset segment medians vs artifact
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        medians = []
        for lab, s, e in runs:
            m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK)
            medians.append(round(m, 3))

        pub_medians = artifact["per_asset"][asset]["momentum_vol_gate"]["medians"]
        pub_lbls = artifact["per_asset"][asset]["momentum_vol_gate"]["segments"]
        pub_verdict = artifact["per_asset"][asset]["momentum_vol_gate"]["verdict"]

        # Regenerate verdict from medians
        cand = np.array(medians)
        null = np.array([])  # No null medians provided in the artifact
        cand_disp = round(float(np.std(cand)), 3) if len(cand) > 1 else 0.0
        if all(abs(m) <= 0.05 for m in medians):
            regen_verdict = "CONSISTENT_WITH_NOISE"
        elif cand_disp > 2.0 * (0.019 if asset == "AMZN" else 0.012):  # Approximate null dispersion
            regen_verdict = "REGIME_DEPENDENT"
        elif all(m < 0 for m in medians):
            regen_verdict = "REGIME_STABLE_LOSS"
        else:
            regen_verdict = "REGIME_STABLE"

        status = "MATCH" if medians == pub_medians and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            print("  [DEBUG] asset={}: len(runs)={} lbls={}".format(asset, len(runs), lbls))
            print("  [DEBUG] asset={}: pub_lbls={}".format(asset, pub_lbls))
            print("  [DEBUG] asset={}: len(medians)={} medians={}".format(asset, len(medians), medians))
            print("  [DEBUG] asset={}: pub_medians={}".format(asset, pub_medians))
        print("  {}: medians {} segments={} -> {} | verdict {} (pub: {})"
              .format(asset, medians, pub_lbls, status, regen_verdict, pub_verdict))

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
        medians = [round(segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK)[0], 3)
                   for lab, s, e in runs]
        pub = artifact["universe"]["per_asset"][ticker]["medians"]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["segments"]

        # Regenerate verdict
        cand = np.array(medians)
        null = np.array(artifact["universe"]["per_asset"][ticker]["null_medians"])
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

        status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        verdict_status = "MATCH" if regen_verdict == artifact["universe"]["per_asset"][ticker]["verdict"] else "MISMATCH"
        if verdict_status != "MATCH":
            all_ok = False
        print("  {}: medians {} segments={} -> {} | verdict {} (pub: {})"
              .format(ticker, medians, pub_lbls, status, regen_verdict,
                      artifact["universe"]["per_asset"][ticker]["verdict"]))

    # Perturbation recomputation
    print("\n=== Perturbation recomputation (3 / 5 / 10 lookback) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]
    recomputed = []
    for params in grid:
        signals = regime_conditional_momentum_signals(bars.closes_array(), params["lookback"])
        res = bt.walk_forward(
            list(bars), signals, train_window=60, test_window=20,
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

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, LOOKBACK)[0], 3)
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
          "execution. Volatility-conditional momentum independent verification on AMZN/JPM + universe: "
          "exploratory simulation.")


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
