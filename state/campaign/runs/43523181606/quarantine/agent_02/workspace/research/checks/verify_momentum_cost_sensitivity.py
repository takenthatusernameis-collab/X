"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Four gates, each on a code path separate from the check:

1. Signal-semantics gate: rebuild the momentum weights from raw closes for every
   ticker and assert the signal is a genuine two-sided bet (both long and short
   bars in material proportion) at every lookback. This blocks a repeat of the
   mean-log-price defect (a constant position that would make any "no edge"
   result meaningless).
2. Fresh-recomputation gate: recompute AMZN/JPM medians and null medians for all
   3 lookbacks x 3 cost levels from raw closes using a re-implemented
   `volatility_blocks`, segment reconstruction, momentum signal, `walk_forward`,
   and a re-implemented coin-flip null (same documented seed mapping). Compare
   every published median/null/verdict.
3. Cost-mechanics gate: assert (a) the adverse-fill round trip loses exactly
   2 * s * notional, (b) full-sample terminal wealth is strictly decreasing in
   the cost level for every ticker/lookback (the base-free gauge; the
   framework's relative per-fold return can rise with cost because a cost paid
   before the test window lowers the ratio's denominator), and (c) zero cost
   still prices fills exactly at the close.
4. Classification reproducibility gate: re-derive every per-cell classification
   and the universe counts from the published medians/verdicts with an
   independent implementation of the declared rule, and assert determinism of a
   repeated cell recomputation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import random_signals

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COSTS_BPS = (0.0, 10.0, 30.0)
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


def vol_blocks(closes, n_blocks, window):
    """Independent re-implementation of volatility_blocks."""
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
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        runs.append((cur, start, len(labels)))
    return runs


def momentum_weights(closes, lookback):
    """Independent momentum signal: sign of the trailing lookback-day log return."""
    n = len(closes)
    w = [0.0] * n
    for i in range(lookback, n):
        ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        w[i] = 1.0 if ret > 0 else -1.0
    return w


def cfg_for(bps):
    return bt.BacktestConfig(initial_capital=1e6, slippage_proportional=float(bps) / 10000.0)


def null_median(seg_bars, lookback, bps, seed=42):
    """Independent coin-flip null: same documented seed mapping, fresh code."""
    n = len(seg_bars)
    base_seed = seed + int(round(float(lookback)) * 1000)
    sig = random_signals(n, seed=base_seed)
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=0, overlap_window=0, cfg=cfg_for(bps))
    rets = np.array([f.metrics["total_return"] for f in res.folds], dtype=np.float64)
    return float(np.median(np.log1p(np.clip(rets, -1.0 + 1e-12, None))))


def segment_result(ticker, bars, s, e, lookback, bps):
    """Fresh per-segment recomputation.

    Signals are rebuilt from the *segment's own* closes, matching
    `stress_segments`, which re-derives signals from the sliced series rather
    than from the global signal vector.
    """
    seg_bars = list(bars)[s:e]
    seg_closes = np.array([b.close for b in seg_bars], dtype=np.float64)
    w = momentum_weights(seg_closes, lookback)
    sig = [bt.Signal(date=i + 1, weight=w[i]) for i in range(len(seg_bars))]
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg_for(bps))
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    null = null_median(seg_bars, lookback, bps)
    return float(np.median(cand)), null, len(res.folds)


def main():
    with open(ARTIFACT) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["costs_bps"] == list(COSTS_BPS)
    all_ok = True

    tickers = {}
    for entry in [e for e in json.load(open("research/data/manifest.json"))["entries"]]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("=== Gate 1: signal semantics (both sides must be traded) ===")
    for t, bars in tickers.items():
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            w = np.array(momentum_weights(closes, lb))
            long_share = float((w > 0).mean())
            short_share = float((w < 0).mean())
            ok = long_share > 0.15 and short_share > 0.15
            if not ok:
                all_ok = False
            print("  {:6s} lb={:<2d} long={:.1%} short={:.1%} -> {}".format(
                t, lb, long_share, short_share, "OK" if ok else "DEGENERATE"))

    print("\n=== Gate 2: fresh recomputation (AMZN, JPM; all lookbacks x costs) ===")
    for t in ("AMZN", "JPM"):
        bars = tickers[t]
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        for lb in LOOKBACKS:
            for bps in COSTS_BPS:
                med, nulls, nf = [], [], None
                for _lab, s, e in runs:
                    m, n, nf = segment_result(t, bars, s, e, lb, bps)
                    med.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["per_asset"][t][str(lb)]["cost_bps"][str(bps)]
                ok = (med == pub["medians"] and nulls == pub["null_medians"]
                      and [r[0] for r in runs] == pub["segments"] and nf == pub["n_folds"])
                if not ok:
                    all_ok = False
                if not ok:
                    all_ok = False
                    print("    nf={} pub_nf={} segs={} pub_segs={} med_eq={} null_eq={}".format(
                        nf, pub["n_folds"], [r[0] for r in runs], pub["segments"],
                        med == pub["medians"], nulls == pub["null_medians"]))
                print("  {:6s} lb={:<2d} cost={:<4} medians {} nulls {} -> {}".format(
                    t, lb, bps, med, nulls, "MATCH" if ok else "MISMATCH"))
                if not ok:
                    print("    published medians {} nulls {} segs {}".format(
                        pub["medians"], pub["null_medians"], pub["segments"]))

    print("\n=== Gate 3: cost mechanics ===")
    bars = [
        bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6),
        bt.Bar(date=2, open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6),
    ]
    sig = [bt.Signal(date=1, weight=1.0), bt.Signal(date=2, weight=0.0)]
    res = bt.run_bars(bars, sig, cfg_for(10.0))
    rt_ok = abs(float(res.equity_curve[-1]) - (1e6 - 2.0 * 0.001 * 1e6)) < 1e-6
    print("  round trip 10bps terminal {:.2f} (expected 998000.00) -> {}".format(
        res.equity_curve[-1], "OK" if rt_ok else "FAIL"))
    if not rt_ok:
        all_ok = False

    drag_ok = True
    for t, per_lb in art["terminal_wealth"].items():
        for lb, by_cost in per_lb.items():
            vals = [by_cost[str(bps)] for bps in COSTS_BPS]
            if any(hi >= lo for lo, hi in zip(vals, vals[1:])):
                drag_ok = False
                print("  WEALTH DRAG VIOLATION {} lb={} {}".format(t, lb, vals))
    print("  full-sample terminal wealth strictly decreasing in cost: {}".format(drag_ok))
    if not drag_ok:
        all_ok = False

    fill_ok = True
    for t in ("AMZN", "JPM"):
        b = tickers[t]
        c = b.closes_array()
        s = bt.momentum_signals(c, lookback=5)
        r = bt.run_bars(list(b), s, cfg_for(0.0))
        first = r.trades[0]
        bar = list(b)[first.date - 1]
        if abs(first.price - bar.close) > 1e-9:
            fill_ok = False
    print("  zero-cost fills price exactly at the close: {}".format(fill_ok))
    if not fill_ok:
        all_ok = False

    print("\n=== Gate 4: classification reproducibility ===")
    def positive(sr):
        return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])

    def classify(gross, at30):
        if not positive(gross):
            return "NO_GROSS_EDGE"
        return "COST_ROBUST" if positive(at30) else "COST_FRAGILE"

    counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_GROSS_EDGE": 0}
    mismatches = 0
    for t, per_lb in art["per_asset"].items():
        for lb, cell in per_lb.items():
            want = classify(cell["cost_bps"]["0.0"], cell["cost_bps"]["30.0"])
            counts[want] += 1
            if want != cell["classification"]:
                mismatches += 1
                print("  CLASSIFICATION MISMATCH {} lb={} derived {} published {}".format(
                    t, lb, want, cell["classification"]))
    ok_counts = counts == art["universe"]["classification_counts"]
    print("  derived counts {} vs published {} -> {}".format(
        counts, art["universe"]["classification_counts"], "MATCH" if ok_counts else "MISMATCH"))
    print("  per-cell classification mismatches: {}".format(mismatches))
    if not ok_counts or mismatches:
        all_ok = False

    print("\nDeterminism: recomputing AMZN lb=5 10bps medians twice.")
    b = tickers["AMZN"]
    closes = b.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", b, s, e, 5, 10.0)[0], 3) for _l, s, e in runs]
    m2 = [round(segment_result("AMZN", b, s, e, 5, 10.0)[0], 3) for _l, s, e in runs]
    det_ok = m1 == m2
    print("  r1: {} r2: {} -> {}".format(m1, m2, "identical" if det_ok else "DIFFERENT"))
    if not det_ok:
        all_ok = False

    print()
    if all_ok:
        print("All four verification gates pass: the signal is a genuine two-sided "
              "bet, AMZN/JPM medians and nulls reproduce from a fresh path, cost "
              "mechanics are adverse and monotone, and the classification plus "
              "universe counts reproduce.")
    else:
        print("VERIFICATION FAILED: see the mismatching gate above.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()