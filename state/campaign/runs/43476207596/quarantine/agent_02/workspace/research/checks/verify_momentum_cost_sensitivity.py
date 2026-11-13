"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates (all must MATCH):

1. Fresh recomputation via a separate implementation path: re-implemented
   ``volatility_blocks``, segment reconstruction, momentum signal, corrected
   return-based momentum signal, and cost config; fold log returns aggregated
   with ``walk_forward`` (no call to ``stress_segments``) and the framework's
   ``noise_benchmark`` under the same cost config. Every per-asset, per-lookback,
   per-cost-level median/null is recomputed for AMZN/JPM and compared to the
   artifact.

2. Cross-consistency: the existing_momentum zero-cost lookback-5 column is an
   independent rerun of the momentum class, so it must match
   ``momentum_lookback_sweep_results.json`` exactly (medians, nulls, verdicts).

3. Classification reproducibility: the cost-robustness verdicts are re-derived
   from the artifact medians/verdicts via the documented rule, independently of
   the check's own classification code.

4. Determinism: one representative cell is recomputed twice.

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
COST_LABELS = ("zero", "realistic", "double", "stress")
COST_MULT = {"zero": 0.0, "realistic": 1.0, "double": 2.0, "stress": 4.0}
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def cost_cfg(mult: float):
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=REALISTIC_COST["commission_per_trade"] * mult,
        commission_per_share=REALISTIC_COST["commission_per_share"] * mult,
        slippage_cents=REALISTIC_COST["slippage_cents"] * mult,
        slippage_proportional=REALISTIC_COST["slippage_proportional"] * mult,
    )


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


def existing_momentum(closes, lookback):
    """Fresh re-implementation of the framework's momentum_signals (as shipped).

    NOTE: this intentionally reproduces the shipped behaviour, which computes the
    mean log PRICE (always positive) rather than the mean log return, so it
    degenerates to a constant long. Matching the artifact requires reproducing
    the shipped computation exactly.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def corrected_momentum(closes, lookback):
    """Fresh re-implementation of the defect-repair diagnostic signal."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


VARIANT_FN = {
    "existing_momentum": existing_momentum,
    "corrected_momentum": corrected_momentum,
}


def segment_result_fresh(ticker, s, e, lookback, variant, mult):
    """Fresh walk_forward + noise_benchmark for one asset segment/lookback/cost."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = VARIANT_FN[variant](closes, lookback)
    seg_signals = sig[s:e]
    cfg = cost_cfg(mult)
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
    return float(np.median(cand_log)), float(noise.baseline_median_log_return), len(res.folds)


def edge_present(medians, verdict):
    if verdict not in ("REGIME_STABLE", "REGIME_DEPENDENT"):
        return False
    return len(medians) > 0 and all(m > 0 for m in medians)


def classify(per_level):
    ez = edge_present(per_level["zero"]["medians"], per_level["zero"]["verdict"])
    er = edge_present(per_level["realistic"]["medians"], per_level["realistic"]["verdict"])
    es = edge_present(per_level["stress"]["medians"], per_level["stress"]["verdict"])
    if not ez:
        return "NO_EDGE_AT_ZERO_COST"
    if er and es:
        return "COST_ROBUST"
    if er and not es:
        return "COST_SENSITIVE_AT_STRESS"
    return "COST_SENSITIVE_AT_REALISTIC"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all variants/lookbacks/costs) ===")
    for variant in ("existing_momentum", "corrected_momentum"):
        for asset in ["AMZN", "JPM"]:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]
            for lb in LOOKBACKS:
                for cl in COST_LABELS:
                    meds, nulls, nfold = [], [], None
                    for lab, s, e in runs:
                        m, n, nf = segment_result_fresh(
                            asset, s, e, lb, variant, COST_MULT[cl])
                        meds.append(round(m, 3))
                        nulls.append(round(n, 3))
                        nfold = nf
                    pub = art["per_variant"][variant][str(lb)][asset][cl]
                    pub_med = [round(v, 3) for v in pub["medians"]]
                    pub_null = [round(v, 3) for v in pub["null_medians"]]
                    status = ("MATCH"
                              if meds == pub_med and nulls == pub_null
                              and lbls == pub["segments"] else "MISMATCH")
                    if status != "MATCH":
                        all_ok = False
                        print("  MISMATCH {} {} lb={} {}: recomputed med={} null={} "
                              "vs pub med={} null={}".format(
                                  variant, asset, lb, cl, meds, nulls, pub_med, pub_null))
            print("  {} {}: all lookbacks x cost levels recomputed -> {}".format(
                variant, asset, "MATCH" if all_ok else "SEE MISMATCH ABOVE"))

    print("\n=== Gate 2: existing_momentum zero-cost lookback-5 cross-consistency vs momentum_lookback_sweep_results.json ===")
    sweep = load(SWEEP_ARTIFACT)
    sw_lp = sweep["universe"]["per_asset"]
    cmis = False
    for ticker in sw_lp:
        row = art["per_variant"]["existing_momentum"]["5"][ticker]["zero"]
        pub = sw_lp[ticker]
        ok = (row["medians"] == pub["lookback_medians"]["5"]
              and row["null_medians"] == pub["lookback_null_medians"]["5"]
              and row["verdict"] == pub["verdicts_5"])
        if not ok:
            cmis = True
            all_ok = False
        print("  {}: medians {} vs {} | nulls {} vs {} | verdict {} vs {} -> {}".format(
            ticker, row["medians"], pub["lookback_medians"]["5"],
            row["null_medians"], pub["lookback_null_medians"]["5"],
            row["verdict"], pub["verdicts_5"], "MATCH" if ok else "MISMATCH"))
    if not cmis:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: cost-robustness classification reproducibility ===")
    rmis = False
    for variant in ("existing_momentum", "corrected_momentum"):
        for lb in LOOKBACKS:
            for ticker in art["universe"][variant][str(lb)]["ticker_order"]:
                per_level = art["per_variant"][variant][str(lb)][ticker]
                derived = classify(per_level)
                published = per_level["_cost_robustness"]
                if derived != published:
                    rmis = True
                    all_ok = False
                    print("  MISMATCH {} lb={} {}: derived {} vs published {}".format(
                        variant, lb, ticker, derived, published))
    if not rmis:
        print("  all variant x lookback x asset classifications reproduce -> MATCH")

    print("\n=== Gate 4: determinism (existing_momentum AMZN lb5 realistic, twice) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result_fresh("AMZN", s, e, 5, "existing_momentum", 1.0)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result_fresh("AMZN", s, e, 5, "existing_momentum", 1.0)[0], 3)
          for lab, s, e in runs]
    det = (m1 == m2)
    print("  r1: {} | r2: {} -> {}".format(m1, m2, "identical" if det else "DIFFERENT"))
    if not det:
        all_ok = False

    print()
    if all_ok:
        print("ALL GATES MATCH: independent recomputation, cross-consistency vs "
              "momentum_lookback_sweep_results.json, classification reproducibility, "
              "and determinism.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent path.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()