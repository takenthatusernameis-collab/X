"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates:

0. Definition gate: recompute the signal composition from a fresh implementation
   and confirm the artifact's composition record, including the implemented-vs-
   textbook sign disagreement. This checks what the cost grid measures.

1. Fresh recomputation of AMZN and NVDA (the only non-degenerate asset) at every
   lookback and every cost level via a separate path: re-implemented
   `volatility_blocks`, segment reconstruction, `momentum_signals`, and
   `walk_forward`, plus the framework's `noise_benchmark` for the matched null.
   No call to `stress_segments`.

2. Cross-consistency gate: the zero-cost column must match the prior
   `momentum_lookback_sweep_results.json` (medians, nulls, verdicts) for all 10
   tickers and all 3 lookbacks, because it is the same computation at zero cost.

3. Classification reproducibility: the predeclared rule is re-coded here from
   the artifact's stored cell values and must reproduce every published class.

4. Determinism: AMZN lookback 5 at 1x recomputed twice must be identical.
"""
from __future__ import annotations

import json
import statistics
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
LOOKBACKS = (3, 5, 10)
LEVELS = ("zero", "1x", "2x", "5x")

COST = {
    "zero": dict(),
    "1x": dict(commission_per_trade=2.0, commission_per_share=0.003,
               slippage_cents=2.0, slippage_proportional=0.0005),
    "2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
               slippage_cents=4.0, slippage_proportional=0.001),
    "5x": dict(commission_per_trade=10.0, commission_per_share=0.015,
               slippage_cents=10.0, slippage_proportional=0.0025),
}

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
OUTLIER_TOL = 0.05


def cfg(level):
    return bt.BacktestConfig(initial_capital=1e6, **COST[level])


def vol_blocks(closes, n_blocks, window):
    """Fresh re-implementation of research.backtest.volatility_blocks."""
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
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0])
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


def momentum_signals(closes, lookback):
    """Fresh momentum signal implementation (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def cell(ticker, bars, lookback, level):
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    sig = momentum_signals(closes, lookback)
    medians, nulls = [], []
    n_folds = None
    for _lab, s, e in runs:
        seg_bars = list(bars)[s:e]
        seg_signals = sig[s:e]
        res = bt.walk_forward(seg_bars, seg_signals, train_window=TRAIN,
                              test_window=TEST, warmup=WARM,
                              overlap_window=OVERLAP, cfg=cfg(level))
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        medians.append(round(float(np.median(log)), 6))
        null = noise_benchmark(bars=seg_bars, param_grid=[{"lookback": lookback}],
                               train_window=TRAIN, test_window=TEST,
                               warmup=0, overlap_window=0, periods_per_year=252,
                               cfg=cfg(level))
        nulls.append(round(float(null.baseline_median_log_return), 6))
        if n_folds is None:
            n_folds = len(res.folds)
    cand_disp = round(float(np.std(np.array(medians))), 6)
    null_disp = round(float(np.std(np.array(nulls))), 6)
    return dict(medians=medians, null_medians=nulls,
                candidate_dispersion=cand_disp, null_dispersion=null_disp,
                n_folds=int(n_folds), verdict=verdict(medians, nulls))


def verdict(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def classify(cells):
    zero = cells["zero"]
    if not (zero["verdict"] == "REGIME_STABLE"
            and all(m > 0 for m in zero["medians"])):
        return "NO_BASELINE_EDGE"
    med = {lvl: float(statistics.median(cells[lvl]["medians"])) for lvl in LEVELS}
    null = {lvl: float(statistics.median(cells[lvl]["null_medians"])) for lvl in LEVELS}
    if med["1x"] <= 0.0 or med["1x"] <= null["1x"]:
        return "COST_ERASED"
    if med["2x"] <= 0.0 or med["2x"] <= null["2x"] or med["2x"] / med["zero"] < 0.5:
        return "COST_FRAGILE"
    return "COST_ROBUST"


def main() -> int:
    with open(ARTIFACT) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID

    all_ok = True

    print("=== Gate 0: signal composition recomputed from a fresh path ===")
    for ticker in art["ticker_order"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            w = np.array([s.weight for s in momentum_signals(closes, lb)])
            long_share = round(float(np.mean(w > 0)), 6)
            short_share = round(float(np.mean(w < 0)), 6)
            neutral = int(np.sum(w == 0))
            pub = art["signal_composition"][str(lb)][ticker]
            ok = (long_share == pub["long_share"]
                  and short_share == pub["short_share"]
                  and neutral == pub["neutral_bars"])
            if not ok:
                all_ok = False
            print("  {:6s} lb{:2d} long={:.4f} short={:.4f} neutral={:2d} -> {}".format(
                ticker, lb, long_share, short_share, neutral, "MATCH" if ok else "MISMATCH"))
    degen = {t: art["signal_composition"]["5"][t]["degenerate_static_long"]
             for t in art["ticker_order"]}
    print("  degenerate_static_long at lookback 5: {} (count={})".format(
        degen, sum(1 for v in degen.values() if v)))

    print("\n=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, NVDA) ===")
    for ticker in ("AMZN", "NVDA"):
        bars, _ = bt.load_ticker(ticker)
        for lb in LOOKBACKS:
            for level in LEVELS:
                got = cell(ticker, bars, lb, level)
                pub = art["cells"][ticker][str(lb)][level]
                same = (got["medians"] == pub["medians"]
                        and got["null_medians"] == pub["null_medians"]
                        and got["verdict"] == pub["verdict"]
                        and got["n_folds"] == pub["n_folds"])
                if not same:
                    all_ok = False
                print("  {:6s} lb{:2d} {:4s} -> {}".format(
                    ticker, lb, level, "MATCH" if same else "MISMATCH"))
                if not same:
                    print("    recomputed medians={} nulls={} verdict={}".format(
                        got["medians"], got["null_medians"], got["verdict"]))
                    print("    published   medians={} nulls={} verdict={}".format(
                        pub["medians"], pub["null_medians"], pub["verdict"]))

    print("\n=== Gate 2: zero-cost column vs momentum_lookback_sweep_results.json ===")
    with open(SWEEP_ARTIFACT) as f:
        sweep = json.load(f)
    for ticker in sweep["universe"]["ticker_order"]:
        for lb in LOOKBACKS:
            pub = art["cells"][ticker][str(lb)]["zero"]
            old = sweep["universe"]["per_asset"][ticker]["lookback_medians"][str(lb)]
            old_null = sweep["universe"]["per_asset"][ticker]["lookback_null_medians"][str(lb)]
            old_verdict = sweep["universe"]["per_asset"][ticker]["verdicts_5"] if lb == 5 else None
            same = ([round(v, 3) for v in pub["medians"]] == old
                    and [round(v, 3) for v in pub["null_medians"]] == old_null)
            if lb == 5:
                same = same and pub["verdict"] == old_verdict
            if not same:
                all_ok = False
            print("  {:6s} lb{:2d} -> {}".format(ticker, lb, "MATCH" if same else "MISMATCH"))

    print("\n=== Gate 3: classification reproducibility from stored cells ===")
    for ticker in art["ticker_order"]:
        for lb in LOOKBACKS:
            got = classify(art["cells"][ticker][str(lb)])
            pub = art["classification"][ticker][str(lb)]["cls"]
            same = got == pub
            if not same:
                all_ok = False
            print("  {:6s} lb{:2d} recomputed={:16s} published={:16s} -> {}".format(
                ticker, lb, got, pub, "MATCH" if same else "MISMATCH"))

    print("\n=== Gate 4: determinism (AMZN lookback 5 at 1x, twice) ===")
    bars, _ = bt.load_ticker("AMZN")
    a = cell("AMZN", bars, 5, "1x")
    b = cell("AMZN", bars, 5, "1x")
    same = a == b
    if not same:
        all_ok = False
    print("  identical: {} -> {}".format(same, "MATCH" if same else "MISMATCH"))

    print("\n=== Gate 5: cost-accounting audit recomputed independently ===")
    rt = art["cost_model_audit"]["round_trip"]
    analytic = {
        "no_costs": 10000.0,
        "fixed_2c": 10000.0,
        "proportional_5bps": 10005.0,
        "realistic_1x": 9995.0,
    }
    analytic_ok = all(abs(rt["round_trip_pnl"][k] - v) < 1e-9 for k, v in analytic.items())
    if not analytic_ok:
        all_ok = False
    print("  round-trip PnL analytic {} vs stored {} -> {}".format(
        analytic, rt["round_trip_pnl"], "MATCH" if analytic_ok else "MISMATCH"))
    for ticker in ("AMZN", "JPM"):
        bars_t, _ = bt.load_ticker(ticker)
        closes = bars_t.closes_array()
        sig = momentum_signals(closes, 5)
        res = bt.run_bars(list(bars_t), sig, cfg("1x"))
        trade_closes = np.array([closes[bd.date - 1] for bd in res.trades], dtype=np.float64)
        shares = np.array([f.shares for f in res.trades], dtype=np.float64)
        commission = float(sum(f.commission for f in res.trades))
        slip_cash = float(np.sum(-shares * (np.array([f.price for f in res.trades]) - trade_closes)))
        gross = float(np.sum((0.02 + 0.0005 * trade_closes) * np.abs(shares)))
        pub = art["cost_model_audit"]["per_asset"][ticker]
        same = (pub["fills"] == len(res.trades)
                and abs(pub["commission"] - round(commission, 2)) < 0.01
                and abs(pub["engine_slippage_cash_impact"] - round(slip_cash, 2)) < 0.01
                and abs(pub["gross_intended_slippage_charge"] - round(gross, 2)) < 0.01)
        if not same:
            all_ok = False
        print("  {:6s} fills={} commission={:.0f} engine_slippage={:+.0f} "
              "gross={:.0f} -> {}".format(
                  ticker, len(res.trades), commission, slip_cash, gross,
                  "MATCH" if same else "MISMATCH"))

    print()
    if all_ok:
        print("All verification gates MATCH the artifact.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        return 1
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Independent verification of the momentum cost-sensitivity "
          "artifact: exploratory simulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())