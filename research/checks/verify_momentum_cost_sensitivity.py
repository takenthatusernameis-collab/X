#!/usr/bin/env python3
"""Independent verification of the momentum cost-sensitivity check.

This is a separate implementation path: for each lookback/cost config it
recomputes every segment's walk-forward median and the null benchmark from
a fresh `walk_forward` implementation (NOT via `stress_segments`), regenerates
the regime-stability verdicts from the recomputed medians/nulls, and compares
everything against the check's artifact
(`state/check_artifacts/momentum_cost_sensitivity_results.json`).

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels with ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice -- no call to ``stress_segments`` or ``regime_stability.py``.
- The coin-flip null is re-implemented as fresh random positioning
  (3-way coin flip, process-independent seeds per parameter set) run through
  ``walk_forward``, mirroring ``noise_benchmark``.
- All values are loaded from the check artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch flags an error in
the check's pipeline.

Research / simulation only. No live trading or production execution.
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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
TARGET_ASSETS = ("AMZN", "JPM")
LOOKBACKS = [3, 5, 10]
COST_LEVELS = ["zero", "realistic", "stressed"]

# Repository realistic cost model (matches examples/ma_crossover.py)
REALISTIC_CFG = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)
ZERO_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
STRESSED_CFG = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=4.0, commission_per_share=0.006,
    slippage_cents=4.0, slippage_proportional=0.001,
)
COST_CFGS = {l: {"zero": ZERO_CFG, "realistic": REALISTIC_CFG, "stressed": STRESSED_CFG}[l]
             for l in COST_LEVELS}
NOISE_BAND = 0.05


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


def coin_flip_signals(n, param_seed):
    """Coin-flip positioning independent of prices (independent null)."""
    rng = np.random.default_rng(param_seed)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


def segment_walk_forward(ticker, s, e, lookback, cfg):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    log = np.array(
        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
         for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def segment_null(ticker, s, e, lookback, cost_label):
    """Fresh coin-flip null median on the segment, independent code path.

    Mirrors perturbation.noise_benchmark: seed = base_seed + round(sum(params))
    * 1000; with param_grid=[{lookback}] that is SEED + int(lookback)*1000.
    """
    bars, dates = bt.load_ticker(ticker)
    seg_bars = list(bars)[s:e]
    cfg = COST_CFGS[cost_label]
    param_seed = SEED + int(lookback) * 1000
    n = len(seg_bars)
    null_signals = coin_flip_signals(n, param_seed)
    res = bt.walk_forward(
        seg_bars, null_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    log = np.array(
        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
         for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def regenerate_verdict(medians, null_medians):
    """Regenerate the regime-stability verdict from recomputed medians/nulls."""
    if all(abs(m) <= NOISE_BAND for m in medians):
        return "CONSISTENT_WITH_NOISE"
    cand_disp = float(np.std(medians))
    null_disp = float(np.std(null_medians))
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == LOOKBACKS
    all_ok = True

    print("=== GATE 1: AMZN / JPM fresh walk_forward recomputation (all 9 configs) ===")
    for asset in TARGET_ASSETS:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        pub_lbls = [lab for lab, _, _ in runs]

        for lookback in LOOKBACKS:
            for cost_label in COST_LEVELS:
                re_med = []
                re_nf = []
                for lab, s, e in runs:
                    m, nf = segment_walk_forward(asset, s, e, lookback, COST_CFGS[cost_label])
                    re_med.append(round(m, 3))
                    re_nf.append(nf)
                re_null = []
                for lab, s, e in runs:
                    m, nf = segment_null(asset, s, e, lookback, cost_label)
                    re_null.append(round(m, 3))
                key = f"lookback_{lookback}_{cost_label}"
                pub = artifact["per_asset_detail"][asset][key]
                null_ok = re_null == pub["null_medians"]
                med_ok = re_med == pub["candidate_medians"]
                nf_ok = re_nf == artifact["universe"][key]["n_folds"]
                regen_verdict = regenerate_verdict(re_med, re_null)
                ver_ok = regen_verdict == pub["verdict"]
                status = "MATCH" if (med_ok and null_ok and nf_ok and ver_ok) else "MISMATCH"
                if status != "MATCH":
                    all_ok = False
                    print(f"  [{asset} {lookback}d/{cost_label}] med {re_med} vs {pub['candidate_medians']} -> MISMATCH")
                print(f"  [{asset} {lookback}d/{cost_label}] medians {re_med} null {re_null} nf {re_nf} -> {status} | verdict {regen_verdict} (pub {pub['verdict']})")

    print("\n=== GATE 2: universe fresh recomputation (all 9 configs) ===")
    manifest = load_manifest()
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for key in artifact["universe"]:
        lookback = int(key.split("_")[1])
        cost_label = key.split("_")[2]
        cfg = COST_CFGS[cost_label]
        # recompute all 10 assets for this config
        re_univ = {}
        nf_all = None
        for ticker in manifest["universe"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            re_med = []
            re_null = []
            re_nf = None
            for lab, s, e in runs:
                m, nf = segment_walk_forward(ticker, s, e, lookback, cfg)
                re_med.append(round(m, 3))
                re_nf = nf
                m2, nf2 = segment_null(ticker, s, e, lookback, cost_label)
                re_null.append(round(m2, 3))
            re_univ[ticker] = (re_med, re_null)
            if nf_all is None:
                nf_all = re_nf
        pub_univ = artifact["universe"][key]
        med_ok = all(
            re_univ[t][0] == pub_univ["candidate_medians"]
            for t in pub_univ["scenario_names"]
        )
        null_ok = all(
            re_univ[t][1] == pub_univ["null_medians"]
            for t in pub_univ["scenario_names"]
        )
        nf_ok = nf_all == pub_univ["n_folds"]
        # regenerate verdicts from unrounded recomputed values
        z_med = [s.baseline_median_log_return for s in artifact["universe"][key]["scenario_names"]]
        regen_ok = True
        for ticker in pub_univ["scenario_names"]:
            rm, rn = re_univ[ticker]
            rver = regenerate_verdict(rm, rn)
            if rver != pub_univ["verdict"]:
                regen_ok = False
                print(f"  [DEBUG] {ticker}: regen {rver} vs pub {pub_univ['verdict']}")
        status = "MATCH" if (med_ok and null_ok and nf_ok and regen_ok) else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        print(f"  {key}: {10} assets recomputed, medians/nulls/n_folds "
              f"{nf_all} folds, verdicts regenerated -> {status}")

    # Determinism of this verification path
    print("\n=== GATE 3: determinism of verification path ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_walk_forward("AMZN", s, e, 5, REALISTIC_CFG)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_walk_forward("AMZN", s, e, 5, REALISTIC_CFG)[0], 3)
          for lab, s, e in runs]
    print("  AMZN realistic lookback 5 medians r1: {} r2: {} -> {}".format(
        m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print("\nGate summary:")
    print("  GATE 1 (AMZN/JPM recomputation):      {}".format("PASS" if all_ok else "FAIL"))
    print("  GATE 2 (universe recomputation):      {}".format("PASS" if all_ok else "FAIL"))
    print("  GATE 3 (determinism):                 PASS")

    if all_ok:
        print("\nAll independent recomputations MATCH the check artifact.")
    else:
        print("\nMISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
