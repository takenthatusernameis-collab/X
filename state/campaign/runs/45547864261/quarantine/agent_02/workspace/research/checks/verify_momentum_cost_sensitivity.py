"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates, all using a fresh code path that does NOT call
`stress_segments`:

1. Fresh recomputation of AMZN/JPM per-segment medians and null medians for
   lookback 5 at the zero and base cost levels, via re-implemented
   `volatility_blocks`, segment reconstruction, `momentum_signals`,
   `walk_forward` (with an explicit `BacktestConfig` cost cfg), and the
   framework's `noise_benchmark`. Compared to the artifact.

2. Magnitude classification reproducibility: the per-asset max absolute
   per-segment median change across all cost levels (vs zero) is recomputed
   from the artifact medians, and the COST_* classification is re-derived via
   the documented rule (material if > 0.05; fragile-verdict-only if a zero-cost
   edge is lost without material change).

3. Determinism: the AMZN lookback-5 base-cost medians are recomputed twice and
   asserted identical.

Research / simulation only. No live trading or production execution.
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
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
LOOKBACKS = (3, 5, 10)
MATERIALITY = 0.05

COST_LEVELS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "base": dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005),
    "double": dict(commission_per_trade=2.0, commission_per_share=0.003,
                   slippage_cents=2.0, slippage_proportional=0.0010),
    "quad": dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0020),
}
COST_ORDER = ("zero", "base", "double", "quad")

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


def vol_blocks(closes, n_blocks, window):
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
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def cfg(level):
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM,
                             **COST_LEVELS[level])


def segment_result(ticker, s, e, lookback, level):
    """Fresh path: walk_forward + noise_benchmark, no stress_segments."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg(level))
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg(level), periods_per_year=252)
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def is_edge(medians, verdict):
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def main():
    art = json.loads(ARTIFACT.read_text())
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, lb5, zero & base) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEG)
        lbls = [lab for lab, _, _ in runs]
        for level in ["zero", "base"]:
            medians, nulls = [], []
            for lab, s, e in runs:
                m, n, nf = segment_result(asset, s, e, 5, level)
                medians.append(round(m, 3))
                nulls.append(round(n, 3))
            pub = art["per_asset"][asset]["5"][level]
            status = ("MATCH"
                      if medians == [round(v, 3) for v in pub["medians"]]
                      and nulls == [round(v, 3) for v in pub["null_medians"]]
                      and lbls == pub["segments"]
                      else "MISMATCH")
            if status != "MATCH":
                all_ok = False
            print("  {} lb5 {}: segments={} -> {}".format(asset, level, lbls, status))
            if status != "MATCH":
                print("    medians recomputed {} vs published {}".format(medians, pub["medians"]))
                print("    nulls  recomputed {} vs published {}".format(nulls, pub["null_medians"]))

    print("\n=== Gate 2: magnitude classification reproducibility ===")
    per_asset = art["per_asset"]
    material = {}
    for t in per_asset:
        worst = 0.0
        for lb in per_asset[t]:
            z = per_asset[t][lb]["zero"]["medians"]
            for lvl in COST_ORDER:
                if lvl == "zero":
                    continue
                lev = per_asset[t][lb][lvl]["medians"]
                worst = max(worst, max(abs(a - b) for a, b in zip(z, lev)))
        material[t] = round(worst, 4)
    material_assets = [t for t, w in material.items() if w > MATERIALITY]
    zero_edges = [t for t in per_asset
                  if is_edge(per_asset[t]["5"]["zero"]["medians"],
                             per_asset[t]["5"]["zero"]["verdict"])]
    base_edges = [t for t in per_asset
                  if is_edge(per_asset[t]["5"]["base"]["medians"],
                             per_asset[t]["5"]["base"]["verdict"])]
    lost = [t for t in zero_edges if t not in base_edges]
    if not zero_edges:
        cls = "COST_INCONCLUSIVE"
    elif material_assets:
        cls = "COST_FRAGILE"
    elif lost:
        cls = "COST_FRAGILE_VERDICT_ONLY"
    else:
        cls = "COST_ROBUST"
    status = "MATCH" if cls == art["classification"] else "MISMATCH"
    if status != "MATCH":
        all_ok = False
    print("  recomputed classification={} vs published={} -> {}".format(
        cls, art["classification"], status))
    print("  material_assets={} lost={} max_median_change={}".format(
        material_assets, lost, material))

    print("\n=== Gate 3: determinism (AMZN lb5 base medians twice) ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEG)
    m1 = [round(segment_result("AMZN", s, e, 5, "base")[0], 3) for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, "base")[0], 3) for lab, s, e in runs]
    det = "identical" if m1 == m2 else "DIFFERENT"
    if m1 != m2:
        all_ok = False
    print("  r1={} r2={} -> {}".format(m1, m2, det))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact and the "
              "classification reproduces via the documented magnitude rule.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()