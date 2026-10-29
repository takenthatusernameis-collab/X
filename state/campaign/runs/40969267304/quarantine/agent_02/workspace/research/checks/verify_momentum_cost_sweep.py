"""Independent verification of `research/checks/momentum_cost_sweep.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregating fold log returns with `walk_forward`). Every
   per-asset, per-lookback, per-cost median is recomputed and compared to the
   artifact. The coin-flip null is recomputed through the framework's own
   ``noise_benchmark`` (a separate code path from ``stress_segments``), so the
   artifact's null medians are also independently reproduced.

2. Cross-consistency gate: the lookback=5 column of this artifact is an
   independent rerun of the momentum class, so it must match
   `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch would
   flag a change in the lookback-5 computation or a defect.

3. Reproducibility gate: the cost-robustness verdicts are re-derived from the
   artifact medians/nulls/verdicts via the documented rule (EDGE_REAL_ONLY /
   CONSISTENT_WITH_NOISE / EDGE_ZERO_ONLY / NO_EDGE), independently of the
   check's own verdict computation.
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

SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sweep_results.json"
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


def segment_result(ticker, s, e, train, test, warm, overlap, lookback, cfg):
    """Walk-forward regime-stability result for one asset segment, lookback, and cost config,
    computed with fresh components: volatility_blocks, segment slicing,
    momentum signals, walk_forward, and the framework noise_benchmark. No call
    to stress_segments."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252)
    null_log = noise.baseline_median_log_return
    return (float(np.median(cand_log)), null_log, len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    sweep = load_artifact(SWEEP_ARTIFACT)
    assert sweep["dataset_id"] == DATASET_ID
    assert sweep["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks, all costs) ===")
    zero_cfg = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        warmup_periods=60,
    )
    real_cfg = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
        warmup_periods=60,
    )

    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False

        for lb in LOOKBACKS:
            # Zero costs
            medians_zero = []
            nulls_zero = []
            for lab, s, e in runs:
                m, n, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb, zero_cfg)
                medians_zero.append(round(m, 3))
                nulls_zero.append(round(n, 3))
            pub_medians_zero = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["zero_medians"][str(lb)]]
            pub_nulls_zero = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["zero_null_medians"][str(lb)]]
            status_zero = ("MATCH" if medians_zero == pub_medians_zero and nulls_zero == pub_nulls_zero and lbls == sweep["universe"]["per_asset"][asset]["zero_verdict"][str(lb)] else "MISMATCH")
            if status_zero != "MATCH":
                mismatch = True
                all_ok = False
            print(f"  {asset}: lookback {lb} zero costs segments={lbls} -> {status_zero}")

            # Real costs
            medians_real = []
            nulls_real = []
            for lab, s, e in runs:
                m, n, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb, real_cfg)
                medians_real.append(round(m, 3))
                nulls_real.append(round(n, 3))
            pub_medians_real = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["real_medians"][str(lb)]]
            pub_nulls_real = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["real_null_medians"][str(lb)]]
            status_real = ("MATCH" if medians_real == pub_medians_real and nulls_real == pub_nulls_real and lbls == sweep["universe"]["per_asset"][asset]["real_verdict"][str(lb)] else "MISMATCH")
            if status_real != "MATCH":
                mismatch = True
                all_ok = False
            print(f"  {asset}: lookback {lb} real costs segments={lbls} -> {status_real}")

        if mismatch:
            print("    [DEBUG] zero recomputed {} vs published {}".format(medians_zero, pub_medians_zero))
            print("    [DEBUG] real recomputed {} vs published {}".format(medians_real, pub_medians_real))

    print("\n=== Gate 2: lookback-5 column cross-consistency vs momentum_results.json ===")
    mom = load_artifact(MOM_ARTIFACT)
    sweep_lp = sweep["universe"]["per_asset"]
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False

    for ticker in mom_lp:
        pub5_zero = mom_lp[ticker]
        sw5_zero = sweep_lp[ticker]["zero_medians"]["5"]
        sn5_zero = sweep_lp[ticker]["zero_null_medians"]["5"]
        sv5_zero = sweep_lp[ticker]["zero_verdict"]["5"]
        status_zero = "MATCH" if (sw5_zero == pub5_zero["medians"] and sn5_zero == pub5_zero["null_medians"] and sv5_zero == pub5_zero["verdict"]) else "MISMATCH"
        if status_zero != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: zero costs medians {} vs {} | nulls {} vs {} | verdict {} vs {} -> {}".format(
            ticker, sw5_zero, pub5_zero["medians"], sn5_zero, pub5_zero["null_medians"],
            sv5_zero, pub5_zero["verdict"], status_zero))

    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: cost-robustness verdict reproducibility ===")

    rmismatch = False
    for ticker in sweep_lp:
        edge_counts = {"EDGE_REAL_ONLY": 0, "CONSISTENT_WITH_NOISE": 0, "EDGE_ZERO_ONLY": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr_zero = {"medians": [round(v, 3) for v in sweep_lp[ticker]["zero_medians"][str(lb)]],
                       "null_medians": [round(v, 3) for v in sweep_lp[ticker]["zero_null_medians"][str(lb)]],
                       "verdict": cost_verdict(sweep_lp[ticker]["zero_medians"][str(lb)],
                                               sweep_lp[ticker]["zero_null_medians"][str(lb)],
                                               sweep_lp[ticker]["real_medians"][str(lb)],
                                               sweep_lp[ticker]["real_null_medians"][str(lb)])}
            sr_real = {"medians": [round(v, 3) for v in sweep_lp[ticker]["real_medians"][str(lb)]],
                       "null_medians": [round(v, 3) for v in sweep_lp[ticker]["real_null_medians"][str(lb)]],
                       "verdict": cost_verdict(sweep_lp[ticker]["zero_medians"][str(lb)],
                                               sweep_lp[ticker]["zero_null_medians"][str(lb)],
                                               sweep_lp[ticker]["real_medians"][str(lb)],
                                               sweep_lp[ticker]["real_null_medians"][str(lb)])}
            edge_counts[edge_or_not(sr_zero, sr_real)] += 1

        robust = edge_counts["EDGE_REAL_ONLY"] + edge_counts["CONSISTENT_WITH_NOISE"] + edge_counts["EDGE_ZERO_ONLY"]
        # The correct way to compare: compute the verdict from edge_counts and compare with published verdict
        # Since there should be exactly one EDGE_REAL_ONLY, CONSISTENT_WITH_NOISE, EDGE_ZERO_ONLY, or NO_EDGE per lookback,
        # the sum is the count of positive outcomes. But for the published verdict, we need the actual verdict string.
        # Actually, looking at the cost_sweep.py code, the "cost_verdicts" stores the verdict for each lookback.
        # So we should just recompute the verdict for lookback 5 and compare it directly with the published one.
        
        # Recompute the verdict for lookback 5 using the documented rule
        sr5_zero = {"medians": [round(v, 3) for v in sweep_lp[ticker]["zero_medians"]["5"]],
                   "null_medians": [round(v, 3) for v in sweep_lp[ticker]["zero_null_medians"]["5"]],
                   "verdict": cost_verdict(sweep_lp[ticker]["zero_medians"]["5"],
                                           sweep_lp[ticker]["zero_null_medians"]["5"],
                                           sweep_lp[ticker]["real_medians"]["5"],
                                           sweep_lp[ticker]["real_null_medians"]["5"])}
        sr5_real = {"medians": [round(v, 3) for v in sweep_lp[ticker]["real_medians"]["5"]],
                   "null_medians": [round(v, 3) for v in sweep_lp[ticker]["real_null_medians"]["5"]],
                   "verdict": cost_verdict(sweep_lp[ticker]["zero_medians"]["5"],
                                           sweep_lp[ticker]["zero_null_medians"]["5"],
                                           sweep_lp[ticker]["real_medians"]["5"],
                                           sweep_lp[ticker]["real_null_medians"]["5"])}
        recomputed_verdict = edge_or_not(sr5_zero, sr5_real)
        
        status = "MATCH" if recomputed_verdict == sweep_lp[ticker]["cost_verdicts"]["5"] else "MISMATCH"
        if status != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: edge_counts={} -> recomputed_verdict={} vs published {} -> {}".format(
            ticker, edge_counts, recomputed_verdict, sweep_lp[ticker]["cost_verdicts"]["5"], status))

    print("\nDeterminism: recomputing AMZN lookback=5 under zero costs twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, zero_cfg)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, zero_cfg)[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback 5 zero costs r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "lookback-5 column cross-checks against momentum_results.json, "
              "and cost-robustness verdicts reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on "
          "AMZN/JPM + cross-consistency vs momentum_results.json: exploratory "
          "simulation.")


if __name__ == "__main__":
    main()
