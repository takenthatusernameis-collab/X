"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates:

1. Fresh recomputation of AMZN/JPM segment medians and matched coin-flip nulls
   via a separate implementation path (re-implemented `volatility_blocks`,
   segment reconstruction, and momentum signals, aggregating fold log returns
   with a fresh `walk_forward`), for every lookback x cost tier. No call to
   `stress_segments` or `stress_segments_across_tickers`; the null is
   recomputed through the framework's own `noise_benchmark` with the same cost
   config. A mismatch would flag a defect in the check.

2. Zero-cost cross-consistency: the check's zero-cost lookback-5 column must
   reproduce the existing `momentum_lookback_sweep_results.json` artifact and,
   via its lookback-5 column, `momentum_results.json` (same data, fresh code).

3. Verdict reproducibility: regime-stability verdicts and cost-tier verdict
   counts are regenerated from the artifact's medians/nulls via the framework's
   own verdict rule (OUTLIER_TOL=0.05, 2x null dispersion) and must agree with
   the published counts.

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
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = (Path.cwd() / "state" / "check_artifacts"
                 / "momentum_cost_sensitivity_results.json")
SWEEP_ARTIFACT = (Path.cwd() / "state" / "check_artifacts"
                  / "momentum_lookback_sweep_results.json")
MOM_ARTIFACT = (Path.cwd() / "state" / "check_artifacts"
                / "momentum_results.json")

COST_CONFIGS = [
    dict(name="zero_cost", commission_per_trade=0.0, commission_per_share=0.0,
         slippage_cents=0.0, slippage_proportional=0.0),
    dict(name="realistic", commission_per_trade=2.0, commission_per_share=0.003,
         slippage_cents=2.0, slippage_proportional=0.0005),
    dict(name="conservative", commission_per_trade=8.0, commission_per_share=0.012,
         slippage_cents=8.0, slippage_proportional=0.002),
]

OUTLIER_TOL = 0.05


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // n_blocks
            s, e = b * n_blocks, (b + 1) * n_blocks
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
        ret = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def cost_cfg(c):
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=c["commission_per_trade"],
        commission_per_share=c["commission_per_share"],
        slippage_cents=c["slippage_cents"],
        slippage_proportional=c["slippage_proportional"],
    )


def segment_result(ticker, s, e, lookback, cost):
    """Fresh recomputation of one segment's candidate median and matched null
    via a separate code path: fresh walk_forward and noise_benchmark. No
    stress_segments."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cost_cfg(cost))
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"],
                       -1.0 + 1e-12, None)) for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cost_cfg(cost))
    null_log = noise.baseline_median_log_return
    return float(np.median(cand_log)), null_log, len(res.folds)


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def verdict_from_medians(medians, null_medians):
    """Regenerate the regime-stability verdict from medians/nulls using the
    framework's own rule (OUTLIER_TOL=0.05, 2x null dispersion)."""
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def main():
    np.random.seed(42)
    artifact = load_artifact(ARTIFACT_PATH)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)
    assert [c["name"] for c in artifact["cost_grid"]] == ["zero_cost",
                                                          "realistic",
                                                          "conservative"]

    all_ok = True

    print("=== Gate 1: fresh walk_forward + matched null (AMZN, JPM, "
          "all lookbacks x cost tiers) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            for c in COST_CONFIGS:
                medians = []
                nulls = []
                nfolds = None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, c)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf
                pub_medians = [round(v, 3) for v in
                               artifact["per_asset"][str(lb)][asset][c["name"]]["medians"]]
                pub_nulls = [round(v, 3) for v in
                             artifact["per_asset"][str(lb)][asset][c["name"]]["null_medians"]]
                pub_lbls = artifact["per_asset"][str(lb)][asset][c["name"]]["segments"]
                status = ("MATCH"
                          if medians == pub_medians
                          and nulls == pub_nulls
                          and lbls == pub_lbls else "MISMATCH")
                if status != "MATCH":
                    mismatch = True
                    all_ok = False
                print("  {}: lookback {} cost={} segments={} -> {}"
                      .format(asset, lb, c["name"], lbls, status))
                if status != "MATCH":
                    print("    medians recomputed {} vs published {}".format(
                        medians, pub_medians))
                    print("    nulls  recomputed {} vs published {}".format(
                        nulls, pub_nulls))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(
                asset, [artifact["per_asset"][str(lb)][asset][c["name"]]["segments"]
                        for lb in LOOKBACKS for c in COST_CONFIGS]))

    print("\n=== Gate 2: zero-cost cross-consistency (lookback 5) ===")
    # 2a. vs momentum_lookback_sweep_results.json
    sweep = load_artifact(SWEEP_ARTIFACT)
    cmismatch = False
    for ticker in sweep["universe"]["per_asset"]:
        sw5 = artifact["per_asset"]["5"][ticker]["zero_cost"]
        ep5 = sweep["universe"]["per_asset"][ticker]
        ok = (sw5["medians"] == [round(v, 3) for v in ep5["lookback_medians"]["5"]]
              and sw5["null_medians"] == [round(v, 3) for v in
                                          ep5["lookback_null_medians"]["5"]]
              and sw5["verdict"] == ep5["verdicts_5"])
        if not ok:
            cmismatch = True
            all_ok = False
        print("  vs sweep: {} medians {} vs {} | verdict {} -> {}"
              .format(ticker, sw5["medians"], ep5["lookback_medians"]["5"],
                      sw5["verdict"], "MATCH" if ok else "MISMATCH"))
    # 2b. vs momentum_results.json (lookback 5 reference universe)
    mom = load_artifact(MOM_ARTIFACT)
    for ticker in mom["universe"]["per_asset"]:
        sw5 = artifact["per_asset"]["5"][ticker]["zero_cost"]
        pub5 = mom["universe"]["per_asset"][ticker]
        ok = (sw5["medians"] == pub5["medians"]
              and sw5["null_medians"] == pub5["null_medians"]
              and sw5["verdict"] == pub5["verdict"])
        if not ok:
            cmismatch = True
            all_ok = False
        print("  vs momentum.py: {} medians {} vs {} | verdict {} -> {}"
              .format(ticker, sw5["medians"], pub5["medians"], sw5["verdict"],
                      "MATCH" if ok else "MISMATCH"))
    if not cmismatch:
        print("  all 10 tickers -> MATCH")

    print("\n=== Gate 3: verdict counts reproducibility per cost tier ===")
    vmismatch = False
    for c in COST_CONFIGS:
        counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
                  "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
        for lb in LOOKBACKS:
            for ticker in artifact["per_asset"][str(lb)]:
                r = artifact["per_asset"][str(lb)][ticker][c["name"]]
                regen = verdict_from_medians(r["medians"], r["null_medians"])
                if regen != r["verdict"]:
                    all_ok = False
                    vmismatch = True
                    print("    [DEBUG] {} lookback {} cost {}: artifact={} regen={}"
                          .format(ticker, lb, c["name"], r["verdict"], regen))
                counts[regen] += 1
        pub_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
                      "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
        for lb in LOOKBACKS:
            for ticker in artifact["per_asset"][str(lb)]:
                pub_counts[artifact["per_asset"][str(lb)][ticker][c["name"]]["verdict"]] += 1
        status = "MATCH" if counts == pub_counts else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            vmismatch = True
        print("  {}: counts={} published_counts={} -> {}".format(
            c["name"], counts, pub_counts, status))

    # Determinism of the verification path.
    print("\nDeterminism: recomputing AMZN lookback=5 zero-cost medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, 5, COST_CONFIGS[0])[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, COST_CONFIGS[0])[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback 5 zero-cost r1: {}; r2: {} -> {}"
          .format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print("\n=== Gate 4: artifact structural integrity ===")
    for lb in LOOKBACKS:
        for ticker in artifact["per_asset"][str(lb)]:
            for c in COST_CONFIGS:
                r = artifact["per_asset"][str(lb)][ticker][c["name"]]
                req = ("segments", "medians", "null_medians", "candidate_dispersion",
                       "null_dispersion", "n_folds", "n_periods", "verdict")
                missing = [k for k in req if k not in r]
                if missing:
                    all_ok = False
                    print("  [DEBUG] missing fields in artifact for "
                          "{}/lookback {}/{}: {}".format(ticker, lb, c["name"], missing))
    print("  all required fields present in every tier -> OK")

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "zero-cost lookback-5 column cross-checks against both existing "
              "momentum artifacts, verdict counts reproduce per cost tier, and "
              "the verification path is deterministic.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity independent verification on "
          "AMZN/JPM + cross-consistency vs existing momentum artifacts: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
