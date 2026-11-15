"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Four verification gates:

1. Fresh recomputation of AMZN/JPM per-segment medians and null medians for
   every cost level and lookback, through a separate implementation path
   (re-implemented volatility blocks, segment reconstruction, and momentum
   signals, aggregating fold log returns with ``walk_forward``). The
   coin-flip null is recomputed through the framework's ``noise_benchmark``
   with the SAME ``BacktestConfig`` (costs) as the candidate, a separate code
   path from ``stress_segments``.

2. Zero-cost cross-consistency: the zero-cost column of this artifact is an
   independent rerun of the momentum class, so its per-asset/per-lookback
   medians, null medians, and (lookback 5) verdicts must match
   ``momentum_lookback_sweep_results.json`` exactly.

3. Reproducibility of the verdicts and the cost-robustness classification from
   the artifact medians/nulls via the documented rule, independently of the
   check's own classification code.

4. Turnover recompute: the per-ticker traded notional, commission, and
   slippage cash impact are recomputed from a fresh ``run_bars`` call at the
   realistic cost level and compared to the artifact; then determinism.
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
LOOKBACKS = (3, 5, 10)
COST_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

VERDICT_TOL = 0.05


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


def segment_result(ticker, s, e, lookback, cfg):
    """Fresh candidate + null medians for one asset segment, lookback, cost cfg."""
    bars, _dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252)
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def derive_verdict(medians, null_medians):
    cand_disp = float(np.std(np.array(medians, dtype=np.float64)))
    null_disp = float(np.std(np.array(null_medians, dtype=np.float64)))
    if all(abs(m) <= VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def is_edge(entry):
    return entry["verdict"] == "REGIME_STABLE" and all(m > 0 for m in entry["medians"])


def main():
    art = json.load(open(COST_ARTIFACT))
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    cost_names = [c["name"] for c in art["cost_levels"]]
    cfg_by_name = {c["name"]: bt.BacktestConfig(
        commission_per_trade=float(c["commission_per_trade"]),
        commission_per_share=float(c["commission_per_share"]),
        slippage_cents=float(c["slippage_cents"]),
        slippage_proportional=float(c["slippage_proportional"]),
    ) for c in art["cost_levels"]}

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all costs/lookbacks) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for cost in cost_names:
            cfg = cfg_by_name[cost]
            for lb in LOOKBACKS:
                medians, nulls, nfolds = [], [], None
                for _lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, cfg)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf
                pub = art["results"][cost][str(lb)][asset]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                status = ("MATCH" if medians == pub_medians
                          and nulls == pub_nulls
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {}: cost {} lookback {} -> {}".format(asset, cost, lb, status))
                    print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                    print("    nulls   recomputed {} vs published {}".format(nulls, pub_nulls))
            print("  {}: cost {} all lookbacks recomputed".format(asset, cost))

    print("\n=== Gate 2: zero-cost column cross-consistency vs momentum_lookback_sweep_results.json ===")
    sweep = json.load(open(SWEEP_ARTIFACT))
    sweep_lp = sweep["universe"]["per_asset"]
    cmismatch = False
    for ticker in sweep_lp:
        for lb in LOOKBACKS:
            z = art["results"]["zero"][str(lb)][ticker]
            sw5 = sweep_lp[ticker]["lookback_medians"][str(lb)]
            sn5 = sweep_lp[ticker]["lookback_null_medians"][str(lb)]
            ok = ([round(v, 3) for v in z["medians"]] == [round(v, 3) for v in sw5]
                  and [round(v, 3) for v in z["null_medians"]] == [round(v, 3) for v in sn5])
            if lb == 5:
                ok = ok and z["verdict"] == sweep_lp[ticker]["verdicts_5"]
            if not ok:
                cmismatch = True
                all_ok = False
                print("  {}: lookback {} MISMATCH".format(ticker, lb))
                print("    zero medians {} vs prior {}".format(z["medians"], sw5))
                print("    zero nulls   {} vs prior {}".format(z["null_medians"], sn5))
                if lb == 5:
                    print("    zero verdict {} vs prior {}".format(
                        z["verdict"], sweep_lp[ticker]["verdicts_5"]))
    if not cmismatch:
        print("  all 10 tickers x 3 lookbacks cross-checked -> MATCH")

    print("\n=== Gate 3: verdicts and cost-robustness classification re-derived from artifact ===")
    rmismatch = False
    edges = {}
    for ticker in art["summary"]["ticker_order"]:
        edges[ticker] = {}
        for cost in cost_names:
            e = []
            for lb in LOOKBACKS:
                entry = art["results"][cost][str(lb)][ticker]
                derived = derive_verdict(entry["medians"], entry["null_medians"])
                if derived != entry["verdict"]:
                    rmismatch = True
                    all_ok = False
                    print("  {}: cost {} lookback {} verdict derived {} vs published {}".format(
                        ticker, cost, lb, derived, entry["verdict"]))
                if derived == "REGIME_STABLE" and all(m > 0 for m in entry["medians"]):
                    e.append(lb)
            edges[ticker][cost] = e
    for ticker in art["summary"]["ticker_order"]:
        z = len(edges[ticker]["zero"]) >= 2
        r = len(edges[ticker]["realistic"]) >= 2
        c = len(edges[ticker]["conservative"]) >= 2
        if not z:
            cls = "NO_EDGE"
        elif c:
            cls = "COST_ROBUST"
        elif r:
            cls = "COST_ERODED"
        else:
            cls = "COST_ELIMINATED"
        pub = art["classification"][ticker]["cost_robustness"]
        status = "MATCH" if cls == pub else "MISMATCH"
        if cls != pub:
            rmismatch = True
            all_ok = False
        print("  {:6s} derived={:14s} published={:14s} -> {}".format(ticker, cls, pub, status))
    if not rmismatch:
        print("  all verdicts and the cost-robustness classification reproduce")

    print("\n=== Gate 4: turnover recompute (realistic) + determinism ===")
    realistic_cfg = cfg_by_name["realistic"]
    tmismatch = False
    for ticker in art["summary"]["ticker_order"]:
        bars, _dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = momentum_signals(closes, 5)
        res = bt.run_bars(list(bars), sig, realistic_cfg)
        traded = round(float(sum(abs(f.shares) * closes[f.date - 1] for f in res.trades)), 1)
        comm = round(float(sum(f.commission for f in res.trades)), 1)
        slip = round(float(sum(-f.shares * (f.price - closes[f.date - 1]) for f in res.trades)), 1)
        pub = art["turnover"]["realistic"][ticker]
        ok = (traded == pub["traded_notional"] and comm == pub["commission"]
              and slip == pub["slippage_net_cash_impact"])
        if not ok:
            tmismatch = True
            all_ok = False
            print("  {}: turnover MISMATCH recomputed {} vs published {}".format(
                ticker, (traded, comm, slip),
                (pub["traded_notional"], pub["commission"], pub["slippage_net_cash_impact"])))
    if not tmismatch:
        print("  all 10 tickers' realistic-cost turnover recompute MATCH")

    bars, _dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    cfg = cfg_by_name["realistic"]
    m1 = [round(segment_result("AMZN", s, e, 5, cfg)[0], 3) for _lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, cfg)[0], 3) for _lab, s, e in runs]
    print("  AMZN lookback 5 realistic r1: {}; r2: {} -> {}".format(
        m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact: per-segment "
              "medians/nulls, the zero-cost cross-consistency, verdicts and "
              "cost-robustness classification, and turnover.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()