"""Independent verification of the momentum cost-sensitivity check
(`research/checks/momentum_cost_sensitivity.py`).

This is a separate implementation path. It does not import ``stress_segments``,
``parameter_sweep``, ``noise_benchmark``, ``momentum_signals``, or
``volatility_blocks``. It re-implements each of those from their documented
contracts and recomputes, for every ticker / lookback / cost level:

  * the per-segment walk-forward median log return of the candidate;
  * the per-segment coin-flip-null median log return under the same cost config;
  * the regime verdict regenerated from the recomputed medians;
  * the signal-construction degeneracy counts.

It then compares against the artifact written by the check
(`state/check_artifacts/momentum_cost_sensitivity_results.json`). A MATCH
confirms the check computed from the stated inputs; a MISMATCH flags a defect.

Independent-path notes:
- The candidate signal is re-implemented exactly as the check defines it
  (``mean(log(closes[i-lookback+1:i+1]))`` sign), so the verifier confirms what
  the check computed, not an alternative signal definition. The degeneracy
  section separately shows what a genuine log-return rule would produce.
- Signals are recomputed on each segment slice (as ``stress_segments`` does),
  not sliced from a global series.
- The null uses ``warmup=0, overlap_window=0`` (matching ``stress_segments``),
  while the candidate uses ``warmup=60, overlap_window=60``.
- The null coin-flip seed is ``42 + lookback*1000`` (matching
  ``perturbation.noise_benchmark``'s parameter-to-seed mapping).

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
LOOKBACKS = (3, 5, 10)
PRIMARY_LOOKBACK = 5
OUTLIER_TOL = 0.05
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

COST_PARAMS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
    "stress": dict(commission_per_trade=8.0, commission_per_share=0.012,
                   slippage_cents=8.0, slippage_proportional=0.0020),
}
COST_LEVELS = ("zero", "realistic", "conservative", "stress")


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


def momentum_signal_impl(closes, lookback):
    """Fresh implementation of the check's momentum signal (mean log price)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        v = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if v > 0 else -1.0)
    return out


def random_signals_impl(n, seed):
    """Fresh implementation of perturbation.random_signals (coin-flip null)."""
    rng = np.random.default_rng(seed)
    w = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(x)) for i, x in enumerate(w)]


def cost_cfg(level):
    p = COST_PARAMS[level]
    return bt.BacktestConfig(
        commission_per_trade=p["commission_per_trade"],
        commission_per_share=p["commission_per_share"],
        slippage_cents=p["slippage_cents"],
        slippage_proportional=p["slippage_proportional"],
    )


def fold_median_log_return(bars, signals, train, test, warm, overlap, cfg):
    res = bt.walk_forward(bars, signals, train_window=train, test_window=test,
                          warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def candidate_segment(ticker_bars, closes, s, e, lookback, level):
    seg_bars = list(ticker_bars)[s:e]
    seg_closes = closes[s:e]
    seg_signals = momentum_signal_impl(seg_closes, lookback)
    return fold_median_log_return(seg_bars, seg_signals, TRAIN, TEST, WARM, OVERLAP,
                                  cost_cfg(level))


def null_segment(ticker_bars, s, e, lookback, level):
    seg_bars = list(ticker_bars)[s:e]
    seed = SEED + int(lookback) * 1000
    sig = random_signals_impl(len(seg_bars), seed)
    return fold_median_log_return(seg_bars, sig, TRAIN, TEST, 0, 0, cost_cfg(level))


def regen_verdict(cand, null):
    cand = np.array(cand, dtype=np.float64)
    null = np.array(null, dtype=np.float64)
    if all(abs(m) <= OUTLIER_TOL for m in cand):
        return "CONSISTENT_WITH_NOISE"
    if float(np.std(cand)) > 2.0 * float(np.std(null)):
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in cand):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def main():
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    assert artifact["dataset_id"] == DATASET_ID, "dataset id mismatch"
    assert artifact["primary_lookback"] == PRIMARY_LOOKBACK
    assert artifact["cost_levels"] == list(COST_LEVELS)

    manifest = load_manifest()
    tickers = {}
    ticker_order = []
    for entry in manifest["entries"]:
        t = entry["ticker"]
        tickers[t] = bt.load_ticker(t)[0]
        ticker_order.append(t)

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True
    n_cells = 0
    n_cells_ok = 0
    max_med_diff = 0.0
    max_null_diff = 0.0

    for ticker in ticker_order:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        for lb in LOOKBACKS:
            for lvl in COST_LEVELS:
                n_cells += 1
                cand = []
                null = []
                for lab, s, e in runs:
                    cm, _ = candidate_segment(bars, closes, s, e, lb, lvl)
                    nm, _ = null_segment(bars, s, e, lb, lvl)
                    cand.append(cm)
                    null.append(nm)
                pub = artifact["per_asset"][ticker][str(lb)][lvl]
                pub_cand = pub["medians"]
                pub_null = pub["null_medians"]
                cand_r = [round(x, 3) for x in cand]
                null_r = [round(x, 3) for x in null]
                max_med_diff = max(max_med_diff, max(abs(a - b) for a, b in zip(cand_r, pub_cand)))
                max_null_diff = max(max_null_diff, max(abs(a - b) for a, b in zip(null_r, pub_null)))
                v = regen_verdict(cand, null)
                ok = (cand_r == pub_cand and null_r == pub_null and v == pub["verdict"])
                if ok:
                    n_cells_ok += 1
                else:
                    all_ok = False
                    print("  MISMATCH {} lb={} lvl={}".format(ticker, lb, lvl))
                    print("    cand  mine={} pub={}".format(cand_r, pub_cand))
                    print("    null  mine={} pub={}".format(null_r, pub_null))
                    print("    verdict mine={} pub={}".format(v, pub["verdict"]))
        print("  {:6s} recomputed 3 lookbacks x 4 cost levels".format(ticker))

    print("\n  cells recomputed: {}/{} MATCH; max |cand diff|={:.4f}; "
          "max |null diff|={:.4f}".format(n_cells_ok, n_cells, max_med_diff, max_null_diff))

    # Degeneracy diagnostic recomputation.
    print("\n=== Signal-degeneracy diagnostic recomputation ===")
    deg_ok = True
    for ticker in ticker_order:
        bars = tickers[ticker]
        closes = bars.closes_array()
        n = len(closes)
        lb = PRIMARY_LOOKBACK
        sig = momentum_signal_impl(closes, lb)
        w = np.array([s.weight for s in sig], dtype=np.float64)
        n_pos = int(np.sum(w > 0))
        n_neg = int(np.sum(w < 0))
        n_zero = int(np.sum(w == 0))
        active = w[n_zero:]
        flips = int(np.sum(np.diff(active) != 0)) if active.size > 1 else 0
        genuine = np.array([np.log(closes[i]) - np.log(closes[i - lb])
                            for i in range(lb, n)], dtype=np.float64)
        mine = dict(
            frac_bars_long=round(n_pos / n, 4),
            n_sign_flips=flips,
            genuine_frac_plus1=round(float(np.mean(genuine > 0)), 4),
            genuine_frac_minus1=round(float(np.mean(genuine < 0)), 4),
        )
        pub = artifact["signal_degeneracy"][ticker]
        ok = (mine["frac_bars_long"] == round(pub["frac_bars_long"], 4)
              and mine["n_sign_flips"] == pub["n_sign_flips"]
              and mine["genuine_frac_plus1"] == round(pub["genuine_frac_plus1"], 4)
              and mine["genuine_frac_minus1"] == round(pub["genuine_frac_minus1"], 4))
        if not ok:
            deg_ok = False
            all_ok = False
            print("  MISMATCH {}: mine={} pub={}".format(ticker, mine, pub))
    print("  degeneracy recomputation: {}".format("ALL MATCH" if deg_ok else "MISMATCH"))

    # Determinism of this verification path.
    print("\n=== Determinism of verification path ===")
    closes = tickers["AMZN"].closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    r1 = [round(candidate_segment(tickers["AMZN"], closes, s, e, 5, "realistic")[0], 3)
          for lab, s, e in runs]
    r2 = [round(candidate_segment(tickers["AMZN"], closes, s, e, 5, "realistic")[0], 3)
          for lab, s, e in runs]
    print("  AMZN realistic lookback5 r1={} r2={} -> {}".format(
        r1, r2, "identical" if r1 == r2 else "DIFFERENT"))
    assert r1 == r2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact "
              "({}/{} cells).".format(n_cells_ok, n_cells))
    else:
        print("MISMATCH FOUND: the cost-sensitivity artifact does not reproduce "
              "via the independent path. Investigate before admitting results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()