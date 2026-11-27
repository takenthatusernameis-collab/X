"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Separate implementation path -- does NOT call ``stress_segments`` or
``stress_segments_across_tickers``. Recomputes every reported number from a
fresh ``walk_forward`` (candidate) and ``noise_benchmark`` (matched coin-flip
null) using re-implemented ``volatility_blocks`` / segment reconstruction /
momentum signals, under the SAME cost cfg the artifact declares.

Gates:
  1. Fresh recomputation (AMZN, JPM; all lookbacks 3/5/10; all cost levels
     zero/realistic/stress2x): per-segment candidate medians and matched-null
     medians recomputed and compared to the artifact (6 dp).
  2. Classification reproducibility: the cost-robustness classification
     (E0/Er/E2 edge sets and COST_ROBUST / COST_MARGINAL / COST_FRAGILE /
     NO_ZERO_COST_EDGE) is re-derived from the artifact gaps via the documented
     rule, independently of the check's own classification code.
  3. Cross-consistency: the zero-cost lookback-5 column must reproduce
     ``momentum_results.json`` (the admitted zero-cost momentum artifact) at the
     published 3 dp (medians, null medians, verdicts) for all 10 tickers.
  4. Determinism of this verification path.

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
COST_NAMES = ("zero", "realistic", "stress2x")
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"


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


def cfg_from_kwargs(kwargs):
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM, **kwargs)


def segment_pair(ticker, s, e, lookback, cfg):
    """Fresh per-segment candidate median and matched-null median.

    Candidate: walk_forward with warmup=WARM, overlap=OVERLAP (matches the
    check's parameter_sweep). Null: noise_benchmark with warmup=0, overlap=0
    (matches stress_segments' internal null call). Both use the same cost cfg.
    """
    bars, dates = bt.load_ticker(ticker)
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
        warmup=0, overlap_window=0, periods_per_year=252, cfg=cfg)
    return float(np.median(cand_log)), float(noise.baseline_median_log_return)


def edge_set(cell):
    return {t for t, r in cell.items() if all(g > 0 for g in r["gaps"])}


def classify(cell):
    E0 = edge_set(cell["zero"])
    Er = edge_set(cell["realistic"])
    if not E0:
        return "NO_ZERO_COST_EDGE"
    if E0 <= Er:
        return "COST_ROBUST"
    if Er:
        return "COST_MARGINAL"
    return "COST_FRAGILE"


def main():
    np.random.seed(42)
    with open(ARTIFACT) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)

    # Build the cost cfg per level from the artifact's own declared grid.
    cfgs = {}
    for lvl in art["cost_levels"]:
        cfgs[lvl["name"]] = cfg_from_kwargs(lvl["kwargs"])

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lb x cost) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for cost in COST_NAMES:
                cfg = cfgs[cost]
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n = segment_pair(asset, s, e, lb, cfg)
                    meds.append(round(m, 6))
                    nulls.append(round(n, 6))
                pub = art["results"][str(lb)][cost][asset]
                pub_med = [round(v, 6) for v in pub["medians"]]
                pub_null = [round(v, 6) for v in pub["null_medians"]]
                ok = (meds == pub_med and nulls == pub_null and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                print("  {:5s} lb={:2d} {:9s} segments={} -> {}".format(
                    asset, lb, cost, lbls, "MATCH" if ok else "MISMATCH"))
                if not ok:
                    print("    med  recomp {} vs pub {}".format(meds, pub_med))
                    print("    null recomp {} vs pub {}".format(nulls, pub_null))

    print("\n=== Gate 2: classification reproducibility (from artifact gaps) ===")
    for lb in LOOKBACKS:
        cell = art["results"][str(lb)]
        cls = classify(cell)
        pub = art["per_lookback"][str(lb)]["classification"]
        ok = cls == pub
        if not ok:
            all_ok = False
        E0 = sorted(edge_set(cell["zero"]))
        Er = sorted(edge_set(cell["realistic"]))
        print("  lb={:2d}: recomputed {} vs published {} -> {}  E0={} Er={}".format(
            lb, cls, pub, "MATCH" if ok else "MISMATCH", E0, Er))
    overall = classify(art["results"][str(art["canonical_lookback"])])
    ok = overall == art["overall_classification"]
    if not ok:
        all_ok = False
    print("  overall: recomputed {} vs published {} -> {}".format(
        overall, art["overall_classification"], "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 3: zero-cost lookback-5 cross-consistency vs momentum_results.json ===")
    with open(MOM_ARTIFACT) as f:
        mom = json.load(f)
    cmismatch = False
    for ticker, pub in mom["universe"]["per_asset"].items():
        z = art["results"]["5"]["zero"][ticker]
        med3 = [round(v, 3) for v in z["medians"]]
        null3 = [round(v, 3) for v in z["null_medians"]]
        ok = (med3 == [round(v, 3) for v in pub["medians"]]
              and null3 == [round(v, 3) for v in pub["null_medians"]]
              and z["verdict"] == pub["verdict"]
              and z["segments"] == pub["segments"])
        if not ok:
            cmismatch = True
            all_ok = False
        print("  {:6s}: med {} vs {} | null {} vs {} | verdict {} vs {} -> {}".format(
            ticker, med3, [round(v, 3) for v in pub["medians"]],
            null3, [round(v, 3) for v in pub["null_medians"]],
            z["verdict"], pub["verdict"], "MATCH" if ok else "MISMATCH"))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 4: determinism of the verification path ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    cfg = cfgs["realistic"]

    def amzn_lb5():
        out = []
        for lab, s, e in runs:
            m, n = segment_pair("AMZN", s, e, 5, cfg)
            out.append((round(m, 6), round(n, 6)))
        return out

    r1 = amzn_lb5()
    r2 = amzn_lb5()
    det = r1 == r2
    if not det:
        all_ok = False
    print("  AMZN lb5 realistic r1 == r2: {}".format(det))
    assert det, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact; the "
              "classification reproduces; the zero-cost lookback-5 column "
              "cross-checks against momentum_results.json; determinism holds.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()