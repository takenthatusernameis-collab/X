"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Four verification gates, all using a fresh code path that does NOT call
``stress_segments`` / ``stress_segments_across_tickers``:

1. Fresh recomputation of the candidate medians (direct ``walk_forward`` over
   each reconstructed segment with an explicit cost ``BacktestConfig``) and the
   matched coin-flip null medians (framework ``noise_benchmark`` with the same
   cost config) for a target subset of assets (AMZN, JPM, MSFT, NVDA) across
   every lookback {3,5,10} and every cost multiplier {0,1,2,5}x. Every
   recomputed median/null is compared to the artifact.

2. Cross-consistency gate: the zero-cost (0x) lookback-5 column is an
   independent rerun of the admitted momentum class, so it must match
   ``momentum_results.json`` exactly (medians, nulls, verdicts) on all 10
   tickers. A mismatch would flag a change in the baseline computation.

3. Classification reproducibility: the cost-robustness class and per-cost
   ``is_edge`` flags are re-derived from the artifact medians/verdicts via the
   documented rule, independently of the check's own classification code.

4. Determinism: one cell's candidate medians are recomputed twice and must be
   identical.

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
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 5.0)
REFERENCE_LOOKBACK = 5
TARGET = ["AMZN", "JPM", "MSFT", "NVDA"]

# Independent re-declaration of the repository realistic cost model.
BASE_REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"


def cost_config(multiplier):
    m = float(multiplier)
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=0,
        commission_per_trade=BASE_REALISTIC_COST["commission_per_trade"] * m,
        commission_per_share=BASE_REALISTIC_COST["commission_per_share"] * m,
        slippage_cents=BASE_REALISTIC_COST["slippage_cents"] * m,
        slippage_proportional=BASE_REALISTIC_COST["slippage_proportional"] * m,
    )


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


def candidate_median(seg_bars, seg_signals, lookback, multiplier):
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cost_config(multiplier))
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def null_median(seg_bars, lookback, multiplier):
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cost_config(multiplier))
    return float(noise.baseline_median_log_return)


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["cost_multipliers"] == list(COST_MULTIPLIERS)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (cost x lookback) ===")
    for asset in TARGET:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for m in COST_MULTIPLIERS:
                key = "{:.1f}".format(m)
                pub = art["grid"][str(lb)][key][asset]
                cand = []
                nulls = []
                for lab, s, e in runs:
                    seg_bars = list(bars)[s:e]
                    sig = momentum_signals(closes, lb)[s:e]
                    cm, _ = candidate_median(seg_bars, sig, lb, m)
                    nm = null_median(seg_bars, lb, m)
                    cand.append(round(cm, 3))
                    nulls.append(round(nm, 3))
                ok = (cand == pub["medians"] and nulls == pub["null_medians"]
                      and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                print("  {:5s} lb={:>2} cost={:>3}x -> {}".format(
                    asset, lb, key, "MATCH" if ok else "MISMATCH"))
                if not ok:
                    print("    cand {} vs {}".format(cand, pub["medians"]))
                    print("    null {} vs {}".format(nulls, pub["null_medians"]))

    print("\n=== Gate 2: 0x lookback-5 column vs momentum_results.json ===")
    mom = load(MOM_ARTIFACT)
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False
    for ticker in mom_lp:
        pub = mom_lp[ticker]
        z = art["grid"]["5"]["0.0"][ticker]
        ok = (z["medians"] == pub["medians"]
              and z["null_medians"] == pub["null_medians"]
              and z["verdict"] == pub["verdict"])
        if not ok:
            cmismatch = True
            all_ok = False
        print("  {:6s} -> {}".format(ticker, "MATCH" if ok else "MISMATCH"))
        if not ok:
            print("    medians {} vs {}".format(z["medians"], pub["medians"]))
            print("    nulls   {} vs {}".format(z["null_medians"], pub["null_medians"]))
            print("    verdict {} vs {}".format(z["verdict"], pub["verdict"]))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: classification reproducibility ===")

    def is_edge(medians, verdict):
        return verdict == "REGIME_STABLE" and all(v > 0 for v in medians)

    def classify(edge_by_cost):
        highest = None
        for m in COST_MULTIPLIERS:
            if edge_by_cost.get(m, False):
                highest = m
        if highest is None:
            return "NO_EDGE"
        if highest >= 5.0:
            return "COST_ROBUST"
        if highest >= 2.0:
            return "COST_MODERATE"
        if highest >= 1.0:
            return "COST_FRAGILE"
        return "ZERO_COST_ONLY"

    rmismatch = False
    for ticker in art["ticker_order"]:
        edge_by_cost = {}
        for m in COST_MULTIPLIERS:
            key = "{:.1f}".format(m)
            cell = art["grid"]["5"][key][ticker]
            edge_by_cost[m] = is_edge(cell["medians"], cell["verdict"])
        cls = classify(edge_by_cost)
        pub = art["classification"][ticker]
        ok = (cls == pub["cost_robustness"]
              and {("{:.1f}".format(m)): edge_by_cost[m] for m in COST_MULTIPLIERS}
              == pub["edge_by_cost"])
        if not ok:
            rmismatch = True
            all_ok = False
        print("  {:6s} recomputed={:<14} published={:<14} -> {}".format(
            ticker, cls, pub["cost_robustness"], "MATCH" if ok else "MISMATCH"))
    if not rmismatch:
        print("  all 10 classifications reproduce -> MATCH")

    print("\n=== Gate 4: determinism (AMZN lookback 5 cost 5x candidate medians) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    sig = momentum_signals(closes, 5)

    def one_run():
        return [round(candidate_median(list(bars)[s:e], sig[s:e], 5, 5.0)[0], 3)
                for lab, s, e in runs]

    r1 = one_run()
    r2 = one_run()
    det = r1 == r2
    if not det:
        all_ok = False
    print("  r1={} r2={} -> {}".format(r1, r2, "identical" if det else "DIFFERENT"))
    assert det, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the 0x "
              "lookback-5 column cross-checks against momentum_results.json, "
              "and the cost-robustness classification reproduces.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()