"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Five verification gates, all using a fresh code path (re-implemented
volatility_blocks / segment reconstruction / momentum signals, aggregating fold
log returns with ``walk_forward`` and the framework ``noise_benchmark`` — never
calling ``stress_segments``):

1. AMZN/JPM, all lookbacks {3,5,10}, all cost levels: fresh per-segment medians,
   null medians, segment labels, and regenerated verdicts MATCH the artifact.
2. All 10 assets, all lookbacks, conservative cost: fresh medians/nulls MATCH and
   regenerated verdicts MATCH (the decision-relevant cost level).
3. Zero-cost column cross-consistency: the artifact's zero-cost medians equal the
   prior `momentum_lookback_sweep_results.json` zero-cost medians for all assets
   and lookbacks (independent anchor that the zero-cost reference is unchanged).
4. Classification reproducibility: per-asset cost-robustness and the overall
   classification are re-derived from the artifact medians/verdicts via the
   documented rule and MATCH.
5. Determinism: a fresh conservative-cost AMZN lookback-5 recomputation is
   identical across two runs.
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

COST_LEVELS = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
}

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
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


def recomputed_segment(asset, s, e, lookback, cost):
    """Fresh per-segment candidate median, null median, and fold count."""
    bars, _ = bt.load_ticker(asset)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = bt.BacktestConfig(initial_capital=1e6, **COST_LEVELS[cost])
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252, cfg=cfg)
    return float(np.median(cand_log)), float(noise.baseline_median_log_return), len(res.folds)


def verdict_from(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def is_positive_edge(medians, verdict):
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def classify(per_lb_zero, per_lb_cons):
    zero_edges = [lb for lb in LOOKBACKS
                  if is_positive_edge(per_lb_zero[lb][0], per_lb_zero[lb][1])]
    survivors = [lb for lb in zero_edges
                 if is_positive_edge(per_lb_cons[lb][0], per_lb_cons[lb][1])]
    if len(zero_edges) >= 2 and len(survivors) == len(zero_edges):
        return "ROBUST"
    if len(zero_edges) >= 1 and len(survivors) == 0:
        return "FRAGILE"
    if len(zero_edges) == 0:
        return "NO_EDGE"
    return "PARTIAL"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks x costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for cost in ("zero", "realistic", "conservative"):
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = recomputed_segment(asset, s, e, lb, cost)
                    meds.append(round(m, 6))
                    nulls.append(round(n, 6))
                pub = art["per_asset"][asset]["per_lookback"][str(lb)][cost]
                pub_med = [round(v, 6) for v in pub["medians"]]
                pub_null = [round(v, 6) for v in pub["null_medians"]]
                v = verdict_from(pub["medians"], pub["null_medians"])
                status = ("MATCH" if meds == pub_med and nulls == pub_null
                          and lbls == pub["segments"] and v == pub["verdict"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {} lb={} cost={} -> {} (meds {} vs {}, nulls {} vs {})".format(
                        asset, lb, cost, status, meds, pub_med, nulls, pub_null))
        print("  {}: all lookbacks x costs checked".format(asset))

    print("\n=== Gate 2: all 10 assets, conservative cost, all lookbacks ===")
    for asset in art["ticker_order"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        for lb in LOOKBACKS:
            meds, nulls = [], []
            for lab, s, e in runs:
                m, n, _ = recomputed_segment(asset, s, e, lb, "conservative")
                meds.append(round(m, 6))
                nulls.append(round(n, 6))
            pub = art["per_asset"][asset]["per_lookback"][str(lb)]["conservative"]
            pub_med = [round(v, 6) for v in pub["medians"]]
            pub_null = [round(v, 6) for v in pub["null_medians"]]
            v = verdict_from(pub["medians"], pub["null_medians"])
            status = ("MATCH" if meds == pub_med and nulls == pub_null
                      and v == pub["verdict"] else "MISMATCH")
            if status != "MATCH":
                all_ok = False
                print("  {} lb={} -> {} (meds {} vs {}, verdict {} vs {})".format(
                    asset, lb, status, meds, pub_med, v, pub["verdict"]))
    print("  all 10 assets x 3 lookbacks (conservative) checked")

    print("\n=== Gate 3: zero-cost column cross-consistency vs momentum_lookback_sweep_results.json ===")
    # The prior sweep artifact stored medians rounded to 3 decimals; compare at
    # that stored precision (the cross-consistency contract), not at 6 decimals.
    sweep = load(SWEEP_ARTIFACT)
    sweep_lp = sweep["universe"]["per_asset"]
    cmis = False
    for asset in art["ticker_order"]:
        for lb in LOOKBACKS:
            pub_zero = [round(v, 3) for v in
                        art["per_asset"][asset]["per_lookback"][str(lb)]["zero"]["medians"]]
            prior = [round(v, 3) for v in sweep_lp[asset]["lookback_medians"][str(lb)]]
            if pub_zero != prior:
                cmis = True
                all_ok = False
                print("  {} lb={} zero medians {} vs prior {}".format(asset, lb, pub_zero, prior))
    if not cmis:
        print("  all 10 assets x 3 lookbacks zero-cost medians MATCH prior sweep artifact (3-dp)")

    print("\n=== Gate 4: classification reproducibility ===")
    counts = {"ROBUST": 0, "PARTIAL": 0, "FRAGILE": 0, "NO_EDGE": 0}
    zero_edge_n = 0
    for asset in art["ticker_order"]:
        pz, pc = {}, {}
        for lb in LOOKBACKS:
            z = art["per_asset"][asset]["per_lookback"][str(lb)]["zero"]
            c = art["per_asset"][asset]["per_lookback"][str(lb)]["conservative"]
            pz[lb] = (z["medians"], verdict_from(z["medians"], z["null_medians"]))
            pc[lb] = (c["medians"], verdict_from(c["medians"], c["null_medians"]))
        v = classify(pz, pc)
        counts[v] += 1
        if any(is_positive_edge(pz[lb][0], pz[lb][1]) for lb in LOOKBACKS):
            zero_edge_n += 1
        pub_v = art["per_asset"][asset]["cost_robustness"]
        status = "MATCH" if v == pub_v else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        print("  {}: recomputed {} vs published {} -> {}".format(asset, v, pub_v, status))
    if zero_edge_n == 0:
        overall = "NO_EDGE"
    elif counts["ROBUST"] > zero_edge_n / 2.0:
        overall = "COST_ROBUST"
    elif counts["FRAGILE"] > zero_edge_n / 2.0:
        overall = "COST_FRAGILE"
    else:
        overall = "MIXED"
    pub_overall = art["overall_cost_robustness"]
    print("  overall recomputed {} vs published {} -> {}".format(
        overall, pub_overall, "MATCH" if overall == pub_overall else "MISMATCH"))
    if overall != pub_overall:
        all_ok = False

    print("\n=== Gate 5: determinism (AMZN conservative lookback 5, twice) ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    m1 = [round(recomputed_segment("AMZN", s, e, 5, "conservative")[0], 6) for _, s, e in runs]
    m2 = [round(recomputed_segment("AMZN", s, e, 5, "conservative")[0], 6) for _, s, e in runs]
    det = "identical" if m1 == m2 else "DIFFERENT"
    print("  r1={} r2={} -> {}".format(m1, m2, det))
    if m1 != m2:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact; zero-cost "
              "column cross-checks against the prior sweep; classification "
              "reproduces; determinism holds.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()