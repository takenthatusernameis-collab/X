"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates, all using a fresh code path that does NOT call
`stress_segments`:

1. Fresh recomputation (AMZN, JPM; all lookbacks; all cost levels): reimplemented
   `volatility_blocks`, segment reconstruction, momentum signals, `walk_forward`
   with the cost `cfg`, and the framework `noise_benchmark` with the same cost
   `cfg`. Candidate medians, null medians, candidate/null dispersion, and the
   regime verdict are recomputed and compared to the artifact.

2. Zero-cost cross-consistency: the artifact's zero-cost column must reproduce
   `momentum_lookback_sweep_results.json` (an independent zero-cost artifact)
   for all 10 assets and all lookbacks (medians + nulls).

3. Reproducibility: the per-asset cost-robustness classification and n_edge
   counts are re-derived from the artifact via the documented a-priori rule and
   must match; the predeclared cost parameters must match; and a fresh
   recomputation of AMZN lookback=5 realistic medians twice must be identical.

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
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
OUTLIER_TOL = 0.05
ART = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

COST_CFGS = {
    "zero": bt.BacktestConfig(initial_capital=1e6),
    "realistic": bt.BacktestConfig(
        initial_capital=1e6, commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": bt.BacktestConfig(
        initial_capital=1e6, commission_per_trade=4.0, commission_per_share=0.006,
        slippage_cents=4.0, slippage_proportional=0.001),
}


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks (independent)."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0]) if e - s > window else 0.0)
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
        ret = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def verdict_rule(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        v = "CONSISTENT_WITH_NOISE"
    elif cand_disp > 2.0 * null_disp:
        v = "REGIME_DEPENDENT"
    elif all(m < 0 for m in medians):
        v = "REGIME_STABLE_LOSS"
    else:
        v = "REGIME_STABLE"
    return v, cand_disp, null_disp


def fresh_segment(ticker, s, e, lookback, cfg):
    """Fresh candidate medians + null median for one segment (no stress_segments)."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    res = bt.walk_forward(seg_bars, sig[s:e], train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(bars=seg_bars, param_grid=[{"lookback": lookback}],
                            train_window=TRAIN, test_window=TEST, warmup=0,
                            overlap_window=0, cfg=cfg, periods_per_year=252)
    return float(np.median(cand)), float(noise.baseline_median_log_return)


def load(path):
    return json.loads(Path(path).read_text())


def main():
    art = load(ART)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    all_ok = True

    # Gate 1: fresh recomputation for AMZN/JPM, all lookbacks, all cost levels.
    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks/costs) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for cname, cfg in COST_CFGS.items():
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n = fresh_segment(ticker, s, e, lb, cfg)
                    meds.append(m)
                    nulls.append(n)
                v, cd, nd = verdict_rule(meds, nulls)
                pub = art["per_asset"][ticker]["lookbacks"][str(lb)][cname]
                ok = (pub["segments"] == lbls
                      and [round(x, 3) for x in meds] == pub["medians"]
                      and [round(x, 3) for x in nulls] == pub["null_medians"]
                      and round(cd, 3) == pub["candidate_dispersion"]
                      and round(nd, 3) == pub["null_dispersion"]
                      and v == pub["verdict"])
                if not ok:
                    all_ok = False
                    print("  MISMATCH {} lb{} {}: recomputed med{} null{} v{} vs pub med{} null{} v{}".format(
                        ticker, lb, cname, [round(x, 3) for x in meds],
                        [round(x, 3) for x in nulls], v,
                        pub["medians"], pub["null_medians"], pub["verdict"]))
        print("  {}: all lookbacks/costs recomputed".format(ticker))
    print("  Gate 1 -> {}".format("PASS" if all_ok else "FAIL"))

    # Gate 2: zero-cost cross-consistency vs the lookback sweep artifact.
    print("\n=== Gate 2: zero-cost column vs momentum_lookback_sweep_results.json ===")
    gate2_ok = SWEEP.exists()
    if gate2_ok:
        sweep = load(SWEEP)["universe"]["per_asset"]
        for ticker in art["universe"]["ticker_order"]:
            for lb in LOOKBACKS:
                pub = sweep[ticker]["lookback_medians"][str(lb)]
                pubn = sweep[ticker]["lookback_null_medians"][str(lb)]
                mine = art["per_asset"][ticker]["lookbacks"][str(lb)]["zero"]
                if mine["medians"] != pub or mine["null_medians"] != pubn:
                    gate2_ok = False
                    all_ok = False
                    print("  MISMATCH {} lb{}: {} vs {}".format(ticker, lb, mine["medians"], pub))
        print("  Gate 2 -> {}".format("PASS" if gate2_ok else "FAIL"))
    else:
        all_ok = False
        print("  lookback sweep artifact missing -> FAIL")

    # Gate 3: classification reproducibility + cost params + determinism.
    print("\n=== Gate 3: classification reproducibility, cost params, determinism ===")
    gate3_ok = True
    expected_cost = {
        "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                     slippage_cents=0.0, slippage_proportional=0.0),
        "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                          slippage_cents=2.0, slippage_proportional=0.0005),
        "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.001),
    }
    if art["cost_levels"] != expected_cost:
        gate3_ok = False
        all_ok = False
        print("  cost params MISMATCH: {}".format(art["cost_levels"]))

    def lb_edge(sr):
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            return "EDGE"
        if sr["verdict"] == "REGIME_STABLE_LOSS":
            return "LOSS"
        return "NO_EDGE"

    def classify(n_edge):
        if n_edge["zero"] == 0:
            return "NO_BASELINE_EDGE"
        if n_edge["realistic"] >= 2:
            return "COST_ROBUST"
        if n_edge["realistic"] == 1:
            return "COST_FRAGILE"
        return "COST_ERASES_EDGE"

    recomp_counts = {k: 0 for k in ("COST_ROBUST", "COST_FRAGILE", "COST_ERASES_EDGE", "NO_BASELINE_EDGE")}
    for ticker in art["universe"]["ticker_order"]:
        n_edge = {c: 0 for c in COST_CFGS}
        for lb in LOOKBACKS:
            for c in COST_CFGS:
                if lb_edge(art["per_asset"][ticker]["lookbacks"][str(lb)][c]) == "EDGE":
                    n_edge[c] += 1
        cls = classify(n_edge)
        recomp_counts[cls] += 1
        if n_edge != art["per_asset"][ticker]["n_edge"] or cls != art["per_asset"][ticker]["classification"]:
            gate3_ok = False
            all_ok = False
            print("  classification MISMATCH {}: recomputed {} {} vs artifact {} {}".format(
                ticker, n_edge, cls, art["per_asset"][ticker]["n_edge"],
                art["per_asset"][ticker]["classification"]))
    if recomp_counts != art["universe"]["classification_counts"]:
        gate3_ok = False
        all_ok = False
        print("  counts MISMATCH: {} vs {}".format(recomp_counts, art["universe"]["classification_counts"]))

    # Determinism: AMZN lb5 realistic twice via the fresh path.
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    r1 = [round(fresh_segment("AMZN", s, e, 5, COST_CFGS["realistic"])[0], 3) for _, s, e in runs]
    r2 = [round(fresh_segment("AMZN", s, e, 5, COST_CFGS["realistic"])[0], 3) for _, s, e in runs]
    det_ok = r1 == r2
    if not det_ok:
        gate3_ok = False
        all_ok = False
    print("  AMZN lb5 realistic r1={} r2={} -> {}".format(r1, r2, "identical" if det_ok else "DIFFERENT"))
    print("  Gate 3 -> {}".format("PASS" if gate3_ok else "FAIL"))

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity artifact; the "
              "zero-cost column cross-checks against momentum_lookback_sweep_results.json; "
              "the classification reproduces from the artifact; cost params and determinism hold.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent path. "
              "Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution.")


if __name__ == "__main__":
    main()