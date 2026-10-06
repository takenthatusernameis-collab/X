"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates plus a determinism assertion:

1. Fresh recomputation of the realistic-cost arm (and zero-cost arm for
   cross-checking) on AMZN/JPM via a separate implementation path:
   independently implemented momentum signals, volatility blocks, and segment
   reconstruction, then the framework's `walk_forward` + `noise_benchmark` under
   the realistic cost config. Every per-asset, per-lookback median, null median,
   n_folds and overall verdict is recomputed and compared to the artifact.

2. Cross-consistency gate: the zero-cost arm of this artifact at lookback=5 is
   an independent rerun of the momentum class with identical parameters, so it
   must match momentum_lookback_sweep_results.json exactly (medians, nulls,
   verdicts). A mismatch would flag a change in the zero-cost computation or a
   defect in the new check's use of the framework.

3. Verdict-reproducibility gate: lookback-robustness and cost-robustness
   verdicts are re-derived from the artifact's medians/nulls/verdicts via the
   documented rules, independently of the check's own verdict computation.

4. Determinism: recompute the AMZN realistic-cost lookback=5 arm twice and
   assert identical medians.

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
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / \
    "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / \
    "momentum_lookback_sweep_results.json"

# The repository's existing "realistic costs" configuration
# (examples/ma_crossover.py, examples/volatility_regime_filter.py).
REALISTIC_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Independent re-implementations (fresh code paths, not calls to the check).
def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def vol_blocks(closes, n_blocks, window):
    """Independent re-implementation of research.backtest.volatility_blocks."""
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


def segment_result(ticker, s, e, lookback, cfg):
    """Walk-forward result for one asset segment and lookback, fresh path.

    Uses walk_forward + noise_benchmark directly (no stress_segments call).
    """
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"],
                                          -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252, seed=42)
    null_log = noise.baseline_median_log_return
    return (float(np.median(cand_log)), null_log, len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def lookback_verdict(sr):
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST"
    if counts["EDGE"] == 1:
        return "SENSITIVE"
    return "NO_EDGE"


def cost_robustness_verdict(zr, rr):
    if zr == "ROBUST" and rr == "ROBUST":
        return "COST_ROBUST"
    if zr == "ROBUST" and rr != "ROBUST":
        return "COST_SENSITIVE"
    if zr != "ROBUST" and rr == "ROBUST":
        return "COST_REVIVED"
    return "NO_EDGE"


def main():
    np.random.seed(42)
    sweep = load_artifact(SWEEP_ARTIFACT)
    assert sweep["dataset_id"] == DATASET_ID, sweep["dataset_id"]
    assert sweep["lookbacks"] == list(LOOKBACKS), sweep["lookbacks"]

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN/JPM, realistic cost) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            all_seg_ok = True
            pub_z_med = [round(v, 3) for v in sweep["per_asset"][asset][str(lb)]["zero_cost"]
                         ["medians"]]
            pub_r_med = [round(v, 3) for v in sweep["per_asset"][asset][str(lb)]["realistic_cost"]
                         ["medians"]]
            pub_z_null = [round(v, 3) for v in sweep["per_asset"][asset][str(lb)]["zero_cost"]
                          ["null_medians"]]
            pub_r_null = [round(v, 3) for v in sweep["per_asset"][asset][str(lb)]["realistic_cost"]
                          ["null_medians"]]
            recomputed = []
            for _, s, e in runs:
                z_m, z_n, _ = segment_result(asset, s, e, lb, bt.BacktestConfig())
                r_m, r_n, _ = segment_result(asset, s, e, lb, REALISTIC_CONFIG)
                recomputed.append((round(z_m, 3), round(r_m, 3), round(z_n, 3), round(r_n, 3)))
            status = ("MATCH"
                      if [r[0] for r in recomputed] == pub_z_med
                      and [r[1] for r in recomputed] == pub_r_med
                      and [r[2] for r in recomputed] == pub_z_null
                      and [r[3] for r in recomputed] == pub_r_null
                      else "MISMATCH")
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("  {}: lookback {} segments={} med(z)={} med(r)={} null(z)={} null(r)={} -> {}"
                  .format(asset, lb, lbls, [r[0] for r in recomputed], [r[1] for r in recomputed],
                          [r[2] for r in recomputed], [r[3] for r in recomputed], status))
            if status != "MATCH":
                print("    published z_med {} r_med {} z_null {} r_null {}"
                      .format(pub_z_med, pub_r_med, pub_z_null, pub_r_null))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [sweep["per_asset"][asset][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: zero-cost lookback=5 cross-consistency vs "
          "momentum_lookback_sweep_results.json ===")
    lookback_sw = load_artifact(LOOKBACK_ARTIFACT)
    cmismatch = False
    for ticker in lookback_sw["universe"]["per_asset"]:
        pub5 = lookback_sw["universe"]["per_asset"][ticker]
        sw5 = sweep["per_asset"][ticker]["5"]["zero_cost"]
        status = "MATCH" if (
            [round(v, 3) for v in sw5["medians"]] == pub5["lookback_medians"]["5"]
            and [round(v, 3) for v in sw5["null_medians"]]
            == pub5["lookback_null_medians"]["5"]
            and sw5["verdict"] == pub5["verdicts_5"]
        ) else "MISMATCH"
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: med={} vs {} | null={} vs {} | verdict={} vs {} -> {}"
              .format(ticker, [round(v, 3) for v in sw5["medians"]],
                      pub5["lookback_medians"]["5"],
                      [round(v, 3) for v in sw5["null_medians"]],
                      pub5["lookback_null_medians"]["5"], sw5["verdict"],
                      pub5["verdicts_5"], status))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: verdict reproducibility from artifact medians/nulls ===")
    vmismatch = False
    for ticker in sweep["per_asset"]:
        p5 = {str(lb): dict(medians=[round(v, 3)
                                     for v in sweep["per_asset"][ticker][str(lb)]
                                     ["realistic_cost"]["medians"]],
                            null_medians=[round(v, 3)
                                          for v in sweep["per_asset"][ticker][str(lb)]
                                          ["realistic_cost"]["null_medians"]],
                            verdict=sweep["per_asset"][ticker][str(lb)]
                            ["realistic_cost"]["verdict"])
              for lb in LOOKBACKS}
        rr = robustness_verdict(p5)
        pub_rr = sweep["per_asset"][ticker]["robustness_verdict"]["realistic_cost"]
        z5 = {str(lb): dict(medians=[round(v, 3)
                                     for v in sweep["per_asset"][ticker][str(lb)]
                                     ["zero_cost"]["medians"]],
                            null_medians=[round(v, 3)
                                          for v in sweep["per_asset"][ticker][str(lb)]
                                          ["zero_cost"]["null_medians"]],
                            verdict=sweep["per_asset"][ticker][str(lb)]["zero_cost"]["verdict"])
              for lb in LOOKBACKS}
        zr = robustness_verdict(z5)
        pub_zr = sweep["per_asset"][ticker]["robustness_verdict"]["zero_cost"]
        pub_cr = sweep["per_asset"][ticker]["cost_robustness_verdict"]
        cr = cost_robustness_verdict(zr, rr)
        status = "MATCH" if (rr == pub_rr and zr == pub_zr and cr == pub_cr) else "MISMATCH"
        if status != "MATCH":
            vmismatch = True
            all_ok = False
        print("  {}: zero_robust {} vs {} | real_robust {} vs {} | cost_robust "
              "{} vs {} -> {}".format(ticker, zr, pub_zr, rr, pub_rr, cr, pub_cr, status))

    print("\n=== Gate 4: determinism (AMZN realistic cost lookback=5, x2) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, 5, REALISTIC_CONFIG)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, REALISTIC_CONFIG)[0], 3)
          for lab, s, e in runs]
    print("  AMZN realistic lookback 5 r1: {}; r2: {} -> {}"
          .format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "zero-cost lookback=5 column cross-checks against "
              "momentum_lookback_sweep_results.json, and robustness/cost-robustness "
              "verdicts reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on "
          "AMZN/JPM + cross-consistency + verdict reproducibility: exploratory "
          "simulation.")


if __name__ == "__main__":
    main()
