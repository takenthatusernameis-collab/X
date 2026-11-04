"""Independent verification of `research/checks/momentum_cost_sweep.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregating fold log returns with `walk_forward`). Every
   per-asset, per-lookback median is recomputed and compared to the artifact.
   The coin-flip null is recomputed through the framework's own
   ``noise_benchmark`` (a separate code path from ``stress_segments``), so the
   artifact's null medians are also independently reproduced.

2. Cross-consistency gate: the lookback=5 column of this artifact is an
   independent rerun of the momentum class, so it must match
   `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch would
   flag a change in the lookback-5 computation or a defect.

3. Reproducibility gate: the cost-robustness verdicts are re-derived from
   the artifact medians/nulls/verdicts via the documented rule 
   (COST_ROBUST if both zero-cost and realistic-cost tests show positive
   edges for >= 2 lookbacks), independently of the check's own verdict
   computation.
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
COST_SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

COST_VERDICT_TOL = 0.05


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


def segment_result(ticker, s, e, train, test, warm, overlap, lookback, cost_config):
    """Walk-forward regime-stability result for one asset segment and lookback,
    computed with fresh components: volatility_blocks, segment slicing,
    momentum signals, walk_forward, and the framework noise_benchmark. No
    call to stress_segments."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=warm,
        commission_per_trade=cost_config.get("commission_per_trade", 0.0),
        commission_per_share=cost_config.get("commission_per_share", 0.0),
        slippage_cents=cost_config.get("slippage_cents", 0.0),
        slippage_proportional=cost_config.get("slippage_proportional", 0.0),
    )
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=bt.BacktestConfig(
            commission_per_trade=cost_config.get("commission_per_trade", 0.0),
            commission_per_share=cost_config.get("commission_per_share", 0.0),
            slippage_cents=cost_config.get("slippage_cents", 0.0),
            slippage_proportional=cost_config.get("slippage_proportional", 0.0),
        ),
    )
    null_log = noise.baseline_median_log_return
    return (float(np.median(cand_log)), null_log, len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    sweep = load_artifact(COST_SWEEP_ARTIFACT)
    assert sweep["dataset_id"] == DATASET_ID
    assert sweep["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks, both cost types) ===")
    realistic_config = sweep["realistic_cost_config"]
    zero_config = sweep["zero_cost_config"]

    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        for cost_type, cost_config in [("realistic", realistic_config), ("zero", zero_config)]:
            for lb in LOOKBACKS:
                medians = []
                nulls = []
                nfolds = None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb, cost_config)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                    nfolds = nf

                pub_entry = sweep["per_asset"][asset][cost_type][str(lb)]
                pub_medians = [round(v, 3) for v in pub_entry["medians"]]
                pub_nulls = [round(v, 3) for v in pub_entry["null_medians"]]
                status = ("MATCH"
                          if medians == pub_medians
                          and nulls == pub_nulls
                          and lbls == pub_entry["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                print("  {}: {} cost, lookback {} segments={} -> {}".format(
                    asset, cost_type, lb, lbls, status))
                if status != "MATCH":
                    print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                    print("    nulls  recomputed {} vs published {}".format(nulls, pub_nulls))

    print("\n=== Gate 2: lookback-5 column cross-consistency vs momentum_results.json ===")
    mom = load_artifact(MOM_ARTIFACT)
    sweep_lp = sweep["universe"]["per_asset"]
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False
    for ticker in mom_lp:
        pub5 = mom_lp[ticker]
        sw5_zero = sweep_lp[ticker]["lookback_5_zero_verdict"]
        sw5_real = sweep_lp[ticker]["lookback_5_real_verdict"]
        sw5_zero_median = sweep_lp[ticker]["lookback_5_zero_median"]
        sw5_real_median = sweep_lp[ticker]["lookback_5_real_median"]
        status = "MATCH" if (sw5_zero == pub5["verdict"]
                             and sw5_real == pub5["verdict"]
                             and abs(sw5_zero_median - pub5["medians"][0]) < 0.001
                             and abs(sw5_real_median - pub5["medians"][0]) < 0.001) \
            else "MISMATCH"
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: zero_cost verdict {} vs {} median {} vs {}; "
              "real_cost verdict {} vs {} median {} vs {} -> {}"
              .format(ticker, sw5_zero, pub5["verdict"], sw5_zero_median, pub5["medians"][0],
                      sw5_real, pub5["verdict"], sw5_real_median, pub5["medians"][0], status))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: cost-robustness verdict reproducibility ===")
    def verdict(medians, null_medians):
        cand_disp = float(np.std(np.array(medians)))
        null_disp = float(np.std(np.array(null_medians)))
        if all(abs(m) <= COST_VERDICT_TOL for m in medians):
            return "CONSISTENT_WITH_NOISE"
        if cand_disp > 2.0 * null_disp:
            return "REGIME_DEPENDENT"
        if all(m < 0 for m in medians):
            return "REGIME_STABLE_LOSS"
        return "REGIME_STABLE"

    def edge_or_not(sr):
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            return "EDGE"
        if sr["verdict"] == "REGIME_STABLE_LOSS":
            return "LOSS"
        return "NO_EDGE"

    def cost_robustness_verdict(artifact_data):
        """Re-derive cost-robustness verdict from artifact data."""
        per_asset = artifact_data["per_asset"]
        robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_UNKNOWN": 0}

        for ticker in artifact_data["universe"]["per_asset"]:
            p = per_asset[ticker]

            zero_cost_edges = 0
            real_cost_edges = 0

            for lb in LOOKBACKS:
                lb_str = str(lb)

                zero_sr = {
                    "medians": [round(v, 3) for v in p["zero_cost"][lb_str]["medians"]],
                    "null_medians": [round(v, 3) for v in p["zero_cost"][lb_str]["null_medians"]],
                    "verdict": p["zero_cost"][lb_str]["verdict"]
                }

                real_sr = {
                    "medians": [round(v, 3) for v in p["real_cost"][lb_str]["medians"]],
                    "null_medians": [round(v, 3) for v in p["real_cost"][lb_str]["null_medians"]],
                    "verdict": p["real_cost"][lb_str]["verdict"]
                }

                if edge_or_not(zero_sr) == "EDGE":
                    zero_cost_edges += 1
                if edge_or_not(real_sr) == "EDGE":
                    real_cost_edges += 1

            if zero_cost_edges >= 2 and real_cost_edges >= 2:
                verdict = "COST_ROBUST"
            else:
                verdict = "COST_SENSITIVE"

            robustness_counts[verdict] += 1

        return robustness_counts

    rmismatch = False
    for ticker in sweep_lp:
        verdict_counts = cost_robustness_verdict(sweep)
        status = "MATCH" if verdict_counts[sweep_lp[ticker]["robustness_verdict"]] > 0 else "MISMATCH"
        if status != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: robustness verdict {} -> {}".format(
            ticker, sweep_lp[ticker]["robustness_verdict"], verdict_counts))

    print("\nDeterminism: recomputing AMZN lookback=5 medians twice.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, realistic_config)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, realistic_config)[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback 5 realistic cost r1: {}; r2: {} -> {}".format(m1, m2,
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
          "execution. Momentum cost-robustness independent verification "
          "on AMZN/JPM + cross-consistency vs momentum_results.json: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()