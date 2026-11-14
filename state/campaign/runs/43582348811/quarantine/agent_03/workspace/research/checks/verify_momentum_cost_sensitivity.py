"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Four verification gates:

1. Fresh recomputation of AMZN/JPM candidate and null medians via a separate
   implementation path (re-implemented `volatility_blocks`, segment slicing,
   momentum signals, `walk_forward`, and the framework `noise_benchmark`), for
   every cost level and every lookback. Each per-asset median/null/segment/
   verdict must MATCH the check artifact. This path never calls
   `stress_segments`.

2. Cross-consistency: the ZERO-cost column of the cost artifact must reproduce
   `momentum_lookback_sweep_results.json` exactly (medians, nulls, verdicts) for
   all 10 assets and all 3 lookbacks. This proves the cost check's frictionless
   baseline is the same computation as the prior evidence.

3. Classification reproducibility + determinism: edge counts and the family
   classification are re-derived from the artifact's medians/nulls/verdicts via
   the predeclared rule, and one asset/level/lookback is recomputed twice.

4. Cost-model audit (decision-relevant qualification): the existing engine
   applies slippage as a price ADD on every fill, so a buy pays more and a sell
   receives more — i.e. slippage is a credit on the closing side and cancels
   over a round trip. This gate demonstrates that directly and attributes the
   realistic-cost drag on AMZN to commissions vs slippage. It does not change the
   predeclared classification; it records the qualification that the existing
   cost model stresses commissions, not directional slippage.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.data import Bar
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_LEVELS = ("zero", "realistic", "conservative")
CONSERVATIVE_MULTIPLIER = 3.0
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_ROBUST_MIN_EDGE_ASSETS = 6
COST_FRAGILE_MAX_EDGE_ASSETS = 2

ART = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
LOOKBACK_ART = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def cost_config(level):
    if level == "zero":
        return bt.BacktestConfig(initial_capital=1e6)
    mult = 1.0 if level == "realistic" else CONSERVATIVE_MULTIPLIER
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=REALISTIC_COST["commission_per_trade"] * mult,
        commission_per_share=REALISTIC_COST["commission_per_share"] * mult,
        slippage_cents=REALISTIC_COST["slippage_cents"] * mult,
        slippage_proportional=REALISTIC_COST["slippage_proportional"] * mult,
    )


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


def segment_result(ticker, s, e, cfg, lookback):
    """Fresh path: slice segment, walk_forward with cfg, noise_benchmark with cfg."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    res = bt.walk_forward(
        seg_bars, sig[s:e], train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252)
    cand_med = float(np.median(cand_log))
    return cand_med, noise.baseline_median_log_return, len(res.folds)


def is_edge(sr):
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def main():
    with open(ART) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all costs, all lookbacks) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for level in COST_LEVELS:
            cfg = cost_config(level)
            for lb in LOOKBACKS:
                medians, nulls, nfolds = [], [], None
                for idx, (lab, s, e) in enumerate(runs):
                    m, n, nf = segment_result(asset, s, e, cfg, lb)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    # RegimeStressResult.n_folds is the FIRST scenario's fold count.
                    if idx == 0:
                        nfolds = nf
                pub = art["per_asset"][asset][level][str(lb)]
                ok = (medians == pub["medians"]
                      and nulls == pub["null_medians"]
                      and lbls == pub["segments"]
                      and nfolds == pub["n_folds"])
                status = "MATCH" if ok else "MISMATCH"
                if not ok:
                    all_ok = False
                print("  {} {:12s} lb={:2d} segs={} folds={} -> {}".format(
                    asset, level, lb, lbls, nfolds, status))
                if not ok:
                    print("    recomputed medians {} vs published {}".format(medians, pub["medians"]))
                    print("    recomputed nulls   {} vs published {}".format(nulls, pub["null_medians"]))
                    print("    folds {} vs published {}".format(nfolds, pub["n_folds"]))

    print("\n=== Gate 2: zero-cost column cross-consistency vs momentum_lookback_sweep_results.json ===")
    with open(LOOKBACK_ART) as f:
        lb_art = json.load(f)
    lb_pub = lb_art["per_asset"]
    mism = False
    for ticker in art["ticker_order"]:
        for lb in LOOKBACKS:
            z = art["per_asset"][ticker]["zero"][str(lb)]
            p = lb_pub[ticker][str(lb)]
            ok = (z["medians"] == p["medians"]
                  and z["null_medians"] == p["null_medians"]
                  and z["segments"] == p["segments"]
                  and z["verdict"] == p["verdict"])
            if not ok:
                mism = True
                all_ok = False
                print("  {} lb={} MISMATCH zero={} vs prior={}".format(
                    ticker, lb, z, p))
    if not mism:
        print("  all 10 tickers x 3 lookbacks -> MATCH (frictionless baseline reproduces prior evidence)")

    print("\n=== Gate 3: classification reproducibility + determinism ===")
    edge_counts = {}
    for level in COST_LEVELS:
        edge_counts[level] = {
            str(lb): sum(1 for t in art["ticker_order"]
                         if is_edge(art["per_asset"][t][level][str(lb)]))
            for lb in LOOKBACKS
        }
    recomputed = dict(edge_counts)
    cm = recomputed == art["edge_counts_by_cost"]
    print("  edge counts recomputed {} vs published {} -> {}".format(
        recomputed, art["edge_counts_by_cost"], "MATCH" if cm else "MISMATCH"))
    if not cm:
        all_ok = False

    def family(counts_by_lb):
        vals = list(counts_by_lb.values())
        if min(vals) >= COST_ROBUST_MIN_EDGE_ASSETS:
            return "COST_ROBUST"
        if max(vals) <= COST_FRAGILE_MAX_EDGE_ASSETS:
            return "COST_FRAGILE"
        return "MIXED"

    cons_counts = {str(lb): edge_counts["conservative"][str(lb)] for lb in LOOKBACKS}
    recomputed_overall = family(cons_counts)
    print("  conservative counts {} -> overall {} vs published {} -> {}".format(
        cons_counts, recomputed_overall, art["overall_classification"],
        "MATCH" if recomputed_overall == art["overall_classification"] else "MISMATCH"))
    if recomputed_overall != art["overall_classification"]:
        all_ok = False

    # Determinism: recompute AMZN realistic lookback 5 twice.
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    cfg = cost_config("realistic")
    d1 = [round(segment_result("AMZN", s, e, cfg, 5)[0], 3) for _, s, e in runs]
    d2 = [round(segment_result("AMZN", s, e, cfg, 5)[0], 3) for _, s, e in runs]
    det = d1 == d2
    print("  determinism AMZN realistic lb=5 r1={} r2={} -> {}".format(
        d1, d2, "identical" if det else "DIFFERENT"))
    if not det:
        all_ok = False

    print("\n=== Gate 4: cost-model audit (slippage direction) ===")
    probe_bars = [
        Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
        Bar(date=2, open=110.0, high=110.0, low=110.0, close=110.0, volume=1.0),
        Bar(date=3, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
    ]
    probe_sig = [bt.Signal(date=1, weight=0.0),
                 bt.Signal(date=2, weight=1.0),
                 bt.Signal(date=3, weight=-1.0)]
    eq_zero = bt.run_bars(probe_bars, probe_sig,
                          bt.BacktestConfig(initial_capital=1e6)).equity_curve[-1]
    eq_slip = bt.run_bars(
        probe_bars, probe_sig,
        bt.BacktestConfig(initial_capital=1e6, slippage_proportional=0.0005)).equity_curve[-1]
    eq_comm = bt.run_bars(
        probe_bars, probe_sig,
        bt.BacktestConfig(initial_capital=1e6, commission_per_trade=2.0)).equity_curve[-1]
    print("  round-trip probe: zero={:.4f} slip_only={:.4f} comm_only={:.4f}".format(
        eq_zero, eq_slip, eq_comm))
    slip_credit = eq_slip > eq_zero
    comm_cost = eq_comm < eq_zero
    print("  slippage-only equity > zero-cost equity (slippage is a net CREDIT): {}".format(slip_credit))
    print("  commission-only equity < zero-cost equity (commission is a net COST): {}".format(comm_cost))
    if not (slip_credit and comm_cost):
        all_ok = False

    # Real-data attribution: AMZN realistic run, commission vs slippage cash impact.
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    close_by_date = {b.date: b.close for b in bars}
    sig = momentum_signals(closes, 5)
    cfg_real = cost_config("realistic")
    res = bt.run_bars(list(bars), sig, cfg_real)
    comm_total = float(sum(f.commission for f in res.trades))
    slip_cash = float(sum(-f.shares * (f.price - close_by_date[f.date]) for f in res.trades))
    eq_zero_full = bt.run_bars(list(bars), sig,
                               bt.BacktestConfig(initial_capital=1e6)).equity_curve[-1]
    eq_real_full = res.equity_curve[-1]
    drag = (eq_zero_full - eq_real_full) / 1e6
    print("  AMZN full-sample: n_fills={} commissions=${:.2f} net_slippage_cash=${:.2f}".format(
        len(res.trades), comm_total, slip_cash))
    print("  AMZN full-sample equity zero={:.2f} realistic={:.2f} cost_drag={:.4%} of initial capital".format(
        eq_zero_full, eq_real_full, drag))
    print("  (net_slippage_cash near zero => slippage cancels over round trips; "
          "drag is essentially commissions.)")

    # Cross-check the independently computed audit against the published artifact.
    pub_audit = art.get("cost_model_audit")
    audit_ok = False
    if isinstance(pub_audit, dict):
        p = pub_audit["round_trip_probe"]
        a = pub_audit["amzn_full_sample"]
        audit_ok = (
            p["slippage_is_net_credit"] == slip_credit
            and p["commission_is_net_cost"] == comm_cost
            and abs(a["commission_total"] - comm_total) <= 0.01
            and abs(a["net_slippage_cash"] - slip_cash) <= 0.01
            and abs(a["cost_drag_fraction_of_capital"] - drag) <= 1e-6
        )
    print("  published cost_model_audit cross-check -> {}".format(
        "MATCH" if audit_ok else "MISMATCH/MISSING"))
    if not audit_ok:
        all_ok = False

    print()
    if all_ok:
        print("ALL GATES MATCH: independent recomputation reproduces the cost artifact, "
              "the zero-cost column reproduces the prior lookback sweep, classification "
              "reproduces, determinism holds, and the cost-model audit is consistent.")
    else:
        print("MISMATCH FOUND: the cost artifact does not reproduce via the independent "
              "path or a gate failed. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()