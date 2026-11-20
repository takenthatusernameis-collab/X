"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

The verifier deliberately avoids the check's own helpers and recomputes every
quantity through a separate implementation path:

  Gate 1  Fresh walk_forward + noise_benchmark for AMZN and JPM at every cost
          level and every lookback, using a re-implemented volatility-block
          classifier, segment reconstruction, and momentum signal. Candidate
          and null medians are recomputed and compared to the artifact; verdicts
          are re-derived from the fresh medians with the documented rule.

  Gate 2  Fresh recomputation for all 10 assets at the two decision-relevant
          cost levels (repo_realistic and conservative_2x): medians, null
          medians, verdicts, edge counts, and robustness verdicts.

  Gate 3  Headline cost-robustness classification re-derived from the artifact's
          own per-asset medians/nulls/verdicts using the documented rule,
          independent of the check's `classify()` computation.

  Gate 4  Determinism: one asset/cost/lookback recomputed twice.

  Gate 5  Zero-cost cross-consistency: the artifact's zero-cost column must
          reproduce `momentum_lookback_sweep_results.json` exactly.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
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
SWEEP_VERDICT_TOL = 0.05

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
PRIOR = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

_REPO_REALISTIC = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_NAMES = ["zero", "repo_realistic", "conservative_2x", "severe_4x", "extreme_10x"]
COST_MULTIPLIER = {
    "zero": 0.0,
    "repo_realistic": 1.0,
    "conservative_2x": 2.0,
    "severe_4x": 4.0,
    "extreme_10x": 10.0,
}


def cost_kwargs(name):
    m = COST_MULTIPLIER[name]
    return {k: v * m for k, v in _REPO_REALISTIC.items()}


def make_cfg(name):
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=0,
                             periods_per_year=252, **cost_kwargs(name))


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


def fresh_cell(ticker, s, e, lookback, cost_name):
    """Fresh candidate + matched-null medians for one segment/lookback/cost."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=make_cfg(cost_name))
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=make_cfg(cost_name),
        periods_per_year=252)
    return (float(np.median(cand_log)), float(noise.baseline_median_log_return),
            len(res.folds))


def verdict_from(medians, null_medians):
    cand_disp = float(np.std(np.array(medians, dtype=np.float64)))
    null_disp = float(np.std(np.array(null_medians, dtype=np.float64)))
    if all(abs(m) <= SWEEP_VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def edge_of(verdict, medians):
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness(edge_count):
    if edge_count >= 2:
        return "ROBUST"
    if edge_count == 1:
        return "SENSITIVE"
    return "NO_EDGE"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    started = datetime.now(timezone.utc).isoformat()
    print("verify started_at_utc: {}".format(started))
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all costs/lookbacks) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for cost_name in COST_NAMES:
            for lb in LOOKBACKS:
                medians, nulls = [], []
                # The artifact stores `stress_segments`' n_folds, which is the
                # FIRST scenario's fold count (different segments have
                # different lengths and therefore different fold counts).
                nfolds = None
                for lab, s, e in runs:
                    m, n, nf = fresh_cell(asset, s, e, lb, cost_name)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    if nfolds is None:
                        nfolds = nf
                pub = art["per_asset"][cost_name][asset]["lookbacks"][str(lb)]
                v = verdict_from(medians, nulls)
                ok = (medians == pub["medians"]
                      and nulls == pub["null_medians"]
                      and v == pub["verdict"]
                      and lbls == pub["segments"]
                      and nfolds == pub["n_folds"])
                if not ok:
                    all_ok = False
                print("  {:6s} {:16s} lb={:>2} -> {}".format(asset, cost_name, lb,
                                                             "MATCH" if ok else "MISMATCH"))
                if not ok:
                    print("    recomputed medians {} nulls {} verdict {} folds {}".format(
                        medians, nulls, v, nfolds))
                    print("    published  medians {} nulls {} verdict {} folds {}".format(
                        pub["medians"], pub["null_medians"], pub["verdict"], pub["n_folds"]))

    print("\n=== Gate 2: all 10 assets at repo_realistic and conservative_2x ===")
    for cost_name in ["repo_realistic", "conservative_2x"]:
        for ticker in art["universe"]["ticker_order"]:
            bars, _ = bt.load_ticker(ticker)
            closes = bars.closes_array()
            runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
            edge_count = 0
            ok_asset = True
            for lb in LOOKBACKS:
                medians, nulls = [], []
                for lab, s, e in runs:
                    m, n, nf = fresh_cell(ticker, s, e, lb, cost_name)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["per_asset"][cost_name][ticker]["lookbacks"][str(lb)]
                v = verdict_from(medians, nulls)
                same = (medians == pub["medians"] and nulls == pub["null_medians"]
                        and v == pub["verdict"])
                if not same:
                    ok_asset = False
                    all_ok = False
                    print("  MISMATCH {} {} lb={}: recomputed {} / {} / {} vs published {} / {} / {}".format(
                        ticker, cost_name, lb, medians, nulls, v,
                        pub["medians"], pub["null_medians"], pub["verdict"]))
                if edge_of(v, medians) == "EDGE":
                    edge_count += 1
            rob = robustness(edge_count)
            pub_rob = art["per_asset"][cost_name][ticker]["robustness_verdict"]
            if rob != pub_rob or edge_count != art["per_asset"][cost_name][ticker]["edge_count"]:
                ok_asset = False
                all_ok = False
                print("  MISMATCH {} {} robustness {} vs {} (edge_count {} vs {})".format(
                    ticker, cost_name, rob, pub_rob, edge_count,
                    art["per_asset"][cost_name][ticker]["edge_count"]))
            if ok_asset:
                print("  {:6s} {:16s} -> MATCH (robustness {})".format(ticker, cost_name, rob))

    print("\n=== Gate 3: headline classification re-derived from artifact ===")
    def summarize(cost_name):
        robust, edge = [], []
        all_cand, all_null = [], []
        for t in art["universe"]["ticker_order"]:
            prof = art["per_asset"][cost_name][t]
            ec = 0
            for lb in LOOKBACKS:
                lb_res = prof["lookbacks"][str(lb)]
                all_cand.extend(lb_res["medians"])
                all_null.extend(lb_res["null_medians"])
                if edge_of(lb_res["verdict"], lb_res["medians"]) == "EDGE":
                    ec += 1
            if robustness(ec) == "ROBUST":
                robust.append(t)
            if ec >= 1:
                edge.append(t)
        gap = float(np.median(all_cand)) - float(np.median(all_null))
        return dict(robust=robust, edge=edge, gap=gap)

    summ = {c: summarize(c) for c in COST_NAMES}
    e0 = len(summ["zero"]["robust"])
    e1 = len(summ["conservative_2x"]["robust"])
    threshold = max(2, int(math.ceil(e0 / 2.0)))
    gap1 = summ["conservative_2x"]["gap"]
    if e1 >= threshold and gap1 > 0:
        headline = "COST_ROBUST"
    elif e1 >= 1:
        headline = "COST_FRAGILE"
    else:
        headline = "COST_ERASED"
    pub_headline = art["universe"]["headline"]["headline"]
    ok = (headline == pub_headline)
    if not ok:
        all_ok = False
    print("  re-derived: {} (E0={}, E1={}, thr={}, gap={:+.3f})".format(
        headline, e0, e1, threshold, gap1))
    print("  published : {} -> {}".format(pub_headline, "MATCH" if ok else "MISMATCH"))
    # Also cross-check each cost level's published robust/edge asset lists.
    for c in COST_NAMES:
        pub = art["universe"]["cost_summary"][c]
        same = (sorted(summ[c]["robust"]) == pub["robust_assets"]
                and sorted(summ[c]["edge"]) == pub["edge_assets"]
                and round(summ[c]["gap"], 3) == pub["median_gap"])
        if not same:
            all_ok = False
        print("  {:16s} robust/edge/gap -> {}".format(c, "MATCH" if same else "MISMATCH"))

    print("\n=== Gate 4: determinism (AMZN repo_realistic lookback 5 twice) ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    m1 = [round(fresh_cell("AMZN", s, e, 5, "repo_realistic")[0], 3) for _, s, e in runs]
    m2 = [round(fresh_cell("AMZN", s, e, 5, "repo_realistic")[0], 3) for _, s, e in runs]
    det = (m1 == m2)
    if not det:
        all_ok = False
    print("  r1={} r2={} -> {}".format(m1, m2, "identical" if det else "DIFFERENT"))

    print("\n=== Gate 5: zero-cost column vs prior lookback sweep ===")
    prior = load(PRIOR)
    xok = True
    for ticker in art["universe"]["ticker_order"]:
        for lb in LOOKBACKS:
            pub = prior["per_asset"][ticker][str(lb)]
            got = art["per_asset"]["zero"][ticker]["lookbacks"][str(lb)]
            same = (got["medians"] == pub["medians"]
                    and got["null_medians"] == pub["null_medians"]
                    and got["verdict"] == pub["verdict"])
            if not same:
                xok = False
                all_ok = False
                print("  MISMATCH {} lb={}".format(ticker, lb))
    print("  zero-cost column reproduces prior lookback sweep: {}".format(xok))

    print()
    if all_ok:
        print("All independent gates PASS: fresh recomputation MATCHes the artifact, "
              "the headline classification reproduces, and determinism holds.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("verify finished_at_utc: {}".format(datetime.now(timezone.utc).isoformat()))
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()