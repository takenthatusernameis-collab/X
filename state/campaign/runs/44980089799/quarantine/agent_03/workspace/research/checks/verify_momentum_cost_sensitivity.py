"""Independent verification of ``research/checks/momentum_cost_sensitivity.py``.

Four verification gates:

1. Zero-cost cross-consistency: the ``zero`` cost column of the new artifact is
   a rerun of the momentum class at zero cost, so it must reproduce
   ``momentum_lookback_sweep_results.json`` exactly (medians, null medians,
   verdicts, segments) for all 10 tickers and all 3 lookbacks. A mismatch would
   flag a change in the momentum computation or a wiring defect.

2. Fresh cost recomputation: for AMZN/JPM at ``base`` and ``stress2x`` cost, a
   separate implementation path (fresh volatility_blocks, segment reconstruction,
   momentum signals, ``walk_forward`` with the cost cfg for the candidate, and
   the framework ``noise_benchmark`` with the same cost cfg for the null)
   recomputes every per-segment median/null and compares to the artifact.

3. Cost-application sanity: the candidate's full-sample ``run_bars`` total return
   must strictly decrease zero > base > stress2x (costs really are charged to
   the candidate), while the coin-flip null's base-cost commission must far
   exceed the candidate's (documents the turnover asymmetry that makes the
   framework's matched null cost-unfair).

4. Classification reproducibility: the candidate cost-robustness classification
   is re-derived from the artifact medians via the documented rule and compared
   to the artifact's own classification; determinism is checked by recomputing
   AMZN lookback-5 base medians twice.
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
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "base": dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.001),
}

ART = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def cfg_for(level):
    return bt.BacktestConfig(initial_capital=1e6, **COST_LEVELS[level])


def vol_blocks(closes, n_blocks, window):
    """Fresh re-implementation of research.backtest.volatility_blocks."""
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
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result_fresh(ticker, s, e, lookback, level):
    """Fresh per-segment candidate + null medians for one cost level. No call to
    stress_segments."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    res = bt.walk_forward(seg_bars, sig[s:e], train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg_for(level))
    cand = float(np.median(
        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds]))
    noise = noise_benchmark(seg_bars, param_grid=[{"lookback": lookback}],
                            train_window=TRAIN, test_window=TEST, warmup=0,
                            overlap_window=0, cfg=cfg_for(level), periods_per_year=252)
    return cand, noise.baseline_median_log_return


def is_edge(sr):
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def derive_classification(art, lb, tol):
    zero = art["per_cell"][str(lb)]["zero"]
    base = art["per_cell"][str(lb)]["base"]
    stress = art["per_cell"][str(lb)]["stress2x"]
    tickers = list(zero.keys())
    edge_zero = [t for t in tickers if is_edge(zero[t])]
    if not edge_zero:
        return "NO_EDGE_AT_ZERO"
    if not all(all(m > 0 for m in base[t]["medians"]) for t in edge_zero):
        return "COST_ELIMINATED"
    if not all(all(m > 0 for m in stress[t]["medians"]) for t in edge_zero):
        return "COST_STRESS_ELIMINATED"
    max_change = 0.0
    for t in edge_zero:
        for a, b in zip(zero[t]["medians"], stress[t]["medians"]):
            max_change = max(max_change, abs(a - b))
    return "COST_ROBUST" if max_change <= tol else "COST_DEGRADED"


def main():
    art = json.loads(ART.read_text())
    sweep = json.loads(SWEEP.read_text())
    assert art["dataset_id"] == DATASET_ID
    assert sweep["dataset_id"] == DATASET_ID
    all_ok = True

    print("=== Gate 1: zero-cost column cross-consistency vs momentum_lookback_sweep_results.json ===")
    for lb in LOOKBACKS:
        for t in art["per_cell"][str(lb)]["zero"].keys():
            z = art["per_cell"][str(lb)]["zero"][t]
            p = sweep["per_asset"][t][str(lb)]
            ok = (z["medians"] == p["medians"]
                  and z["null_medians"] == p["null_medians"]
                  and z["verdict"] == p["verdict"]
                  and z["segments"] == p["segments"])
            if not ok:
                all_ok = False
                print("  MISMATCH lookback {} {}: medians {} vs {} | nulls {} vs {} | verdict {} vs {}"
                      .format(lb, t, z["medians"], p["medians"], z["null_medians"],
                              p["null_medians"], z["verdict"], p["verdict"]))
    if all_ok:
        print("  all 10 tickers x 3 lookbacks -> MATCH")

    print("\n=== Gate 2: fresh walk_forward + noise_benchmark recomputation (AMZN, JPM; base, stress2x) ===")
    gate2_ok = True
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        for lb in LOOKBACKS:
            for level in ["base", "stress2x"]:
                pub = art["per_cell"][str(lb)][level][asset]
                meds, nulls = [], []
                for _, s, e in runs:
                    m, n = segment_result_fresh(asset, s, e, lb, level)
                    meds.append(round(m, 3))
                    nulls.append(round(n, 3))
                ok = (meds == pub["medians"] and nulls == pub["null_medians"])
                if not ok:
                    gate2_ok = False
                    all_ok = False
                print("  {} lookback {} {:9s} -> {} (med {} vs {}, null {} vs {})".format(
                    asset, lb, level, "MATCH" if ok else "MISMATCH",
                    meds, pub["medians"], nulls, pub["null_medians"]))
    if gate2_ok:
        print("  all fresh recomputations -> MATCH")

    print("\n=== Gate 3: cost-application sanity (candidate cost drag) + null turnover asymmetry ===")
    gate3_ok = True
    for asset in ["AMZN", "AAPL"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        rets = {}
        for level in ["zero", "base", "stress2x"]:
            r = bt.run_bars(list(bars), sig, cfg_for(level))
            rets[level] = (r.equity_curve[-1] - 1e6) / 1e6
        cand_comm = sum(f.commission for f in bt.run_bars(list(bars), sig, cfg_for("base")).trades)
        null = bt.run_bars(list(bars), bt.random_signals(len(closes), seed=42), cfg_for("base"))
        null_comm = sum(f.commission for f in null.trades)
        monotone = rets["zero"] > rets["base"] > rets["stress2x"]
        ratio = null_comm / cand_comm if cand_comm else float("inf")
        ok = monotone and ratio > 5.0
        gate3_ok = gate3_ok and ok
        all_ok = all_ok and ok
        print("  {}: total_return zero={:+.4f} base={:+.4f} stress2x={:+.4f} monotone={} | "
              "null_comm/cand_comm={:.1f}x -> {}".format(
                  asset, rets["zero"], rets["base"], rets["stress2x"], monotone,
                  ratio, "PASS" if ok else "FAIL"))
    if gate3_ok:
        print("  costs charged to candidate (monotone decrease) and null turnover >> candidate -> PASS")

    print("\n=== Gate 4: classification reproducibility + determinism ===")
    tol = art["cost_flat_tol"]
    gate4_ok = True
    for lb in LOOKBACKS:
        derived = derive_classification(art, lb, tol)
        pub = art["cost_robustness"][str(lb)]["classification"]
        ok = derived == pub
        gate4_ok = gate4_ok and ok
        all_ok = all_ok and ok
        print("  lookback {}: derived={} published={} -> {}".format(
            lb, derived, pub, "MATCH" if ok else "MISMATCH"))

    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result_fresh("AMZN", s, e, 5, "base")[0], 3) for _, s, e in runs]
    m2 = [round(segment_result_fresh("AMZN", s, e, 5, "base")[0], 3) for _, s, e in runs]
    det = m1 == m2
    all_ok = all_ok and det
    print("  AMZN lookback 5 base determinism r1==r2: {} ({} vs {})".format(det, m1, m2))

    print()
    if all_ok:
        print("All verification gates PASS: zero-cost column reproduces the prior sweep, "
              "fresh cost recomputation matches, costs are charged to the candidate with "
              "the null turnover far higher, and the classification reproduces deterministically.")
    else:
        print("VERIFICATION FAILED: one or more gates did not match. Investigate before "
              "recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution.")


if __name__ == "__main__":
    main()