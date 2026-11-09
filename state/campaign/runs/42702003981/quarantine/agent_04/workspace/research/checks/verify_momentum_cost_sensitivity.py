"""Independent verification of `research/checks/cost_sensitive_momentum.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregating fold log returns with `walk_forward`). Every
   per-asset, per-lookback median is recomputed and compared to the artifact
   for BOTH realistic and zero cost configurations. The matched null is recomputed
   through the framework's own ``noise_benchmark`` (a separate code path from
   ``stress_segments``), so the artifact's null medians are also independently
   reproduced for both cost configurations.

2. Cross-consistency gate: the lookback=5 column of this artifact is an
   independent rerun of the momentum class, so it must match
   `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch would
   flag a change in the lookback-5 computation or a defect.

3. Reproducibility gate: the lookback-robustness verdicts are re-derived from
   the artifact medians/nulls/verdicts via the documented rule (ROBUST if a
   positive REGIME_STABLE edge appears at >= 2 lookbacks), independently of the
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
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

SWEEP_VERDICT_TOL = 0.05


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
    momentum signals, walk_forward, and the framework noise_benchmark. No call
    to stress_segments."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback=lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cost_config)
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

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks, both cost types) ===")
    
    # Load manifest for universe
    manifest = load_manifest()
    
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    
    # Realistic costs verification
    print("\n  Realistic transaction costs verification:")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            medians = []
            nulls = []
            nfolds = None
            for lab, s, e in runs:
                m, n, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb, REALISTIC_COSTS)
                medians.append(round(m, 3))
                nulls.append(round(n, 3))
                nfolds = nf
            pub_medians = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["realistic"]["lookback_medians"][str(lb)]]
            pub_nulls = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["realistic"]["lookback_null_medians"][str(lb)]]
            status = ("MATCH"
                      if medians == pub_medians
                      and nulls == pub_nulls
                      and lbls == sweep["universe"]["per_asset"][asset]["realistic"]["segments"]
                      else "MISMATCH")
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("    {}: lookback {} segments={} -> {}".format(
                asset, lb, lbls, status))
            if status != "MATCH":
                print("      medians recomputed {} vs published {}".format(medians, pub_medians))
                print("      nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [sweep["universe"]["per_asset"][asset]["realistic"][str(lb)]["segments"] for lb in LOOKBACKS]))

    # Zero costs verification
    print("\n  Zero transaction costs verification:")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        mismatch = False
        for lb in LOOKBACKS:
            medians = []
            nulls = []
            nfolds = None
            for lab, s, e in runs:
                m, n, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb, ZERO_COSTS)
                medians.append(round(m, 3))
                nulls.append(round(n, 3))
                nfolds = nf
            pub_medians = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["zero"]["lookback_medians"][str(lb)]]
            pub_nulls = [round(v, 3) for v in sweep["universe"]["per_asset"][asset]["zero"]["lookback_null_medians"][str(lb)]]
            status = ("MATCH"
                      if medians == pub_medians
                      and nulls == pub_nulls
                      and lbls == sweep["universe"]["per_asset"][asset]["zero"]["segments"]
                      else "MISMATCH")
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("    {}: lookback {} segments={} -> {}".format(
                asset, lb, lbls, status))
            if status != "MATCH":
                print("      medians recomputed {} vs published {}".format(medians, pub_medians))
                print("      nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [sweep["universe"]["per_asset"][asset]["zero"][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: lookback-5 column cross-consistency vs momentum_results.json ===")
    mom = load_artifact(MOM_ARTIFACT)
    sweep_lp = sweep["universe"]["per_asset"]
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False
    for ticker in mom_lp:
        pub5 = mom_lp[ticker]
        sw5_real = sweep_lp[ticker]["realistic"]["lookback_medians"]["5"]
        sw5_zero = sweep_lp[ticker]["zero"]["lookback_medians"]["5"]
        sv5_real = sweep_lp[ticker]["realistic"]["verdicts_5_realistic"]
        sv5_zero = sweep_lp[ticker]["zero"]["verdicts_5_zero"]
        status_real = "MATCH" if (sw5_real == pub5["medians"] and sv5_real == pub5["verdict"]) else "MISMATCH"
        status_zero = "MATCH" if (sw5_zero == pub5["medians"] and sv5_zero == pub5["verdict"]) else "MISMATCH"
        if status_real != "MATCH" or status_zero != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: lookback-5 real medians {} vs {} | verdict {} vs {} -> {} ; "
              "lookback-5 zero medians {} vs {} | verdict {} vs {} -> {}"
              .format(ticker, sw5_real, pub5["medians"], sv5_real, pub5["verdict"], status_real,
                      sw5_zero, pub5["medians"], sv5_zero, pub5["verdict"], status_zero))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: lookback-robustness verdict reproducibility ===")
    def verdict(medians, null_medians):
        cand_disp = float(np.std(np.array(medians)))
        null_disp = float(np.std(np.array(null_medians)))
        if all(abs(m) <= SWEEP_VERDICT_TOL for m in medians):
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

    rmismatch = False
    for ticker in sweep_lp:
        edge_counts_real = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        edge_counts_zero = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr_real = {"medians": [round(v, 3) for v in sweep_lp[ticker]["realistic"]["lookback_medians"][str(lb)]],
                      "null_medians": [round(v, 3) for v in sweep_lp[ticker]["realistic"]["lookback_null_medians"][str(lb)]],
                      "verdict": verdict([round(v, 3) for v in sweep_lp[ticker]["realistic"]["lookback_medians"][str(lb)]],
                                         [round(v, 3) for v in sweep_lp[ticker]["realistic"]["lookback_null_medians"][str(lb)]])}
            sr_zero = {"medians": [round(v, 3) for v in sweep_lp[ticker]["zero"]["lookback_medians"][str(lb)]],
                      "null_medians": [round(v, 3) for v in sweep_lp[ticker]["zero"]["lookback_null_medians"][str(lb)]],
                      "verdict": verdict([round(v, 3) for v in sweep_lp[ticker]["zero"]["lookback_medians"][str(lb)]],
                                         [round(v, 3) for v in sweep_lp[ticker]["zero"]["lookback_null_medians"][str(lb)]])}
            edge_counts_real[edge_or_not(sr_real)] += 1
            edge_counts_zero[edge_or_not(sr_zero)] += 1
        robust_real = "ROBUST" if edge_counts_real["EDGE"] >= 2 else ("SENSITIVE" if edge_counts_real["EDGE"] == 1 else "NO_EDGE")
        robust_zero = "ROBUST" if edge_counts_zero["EDGE"] >= 2 else ("SENSITIVE" if edge_counts_zero["EDGE"] == 1 else "NO_EDGE")
        status_real = "MATCH" if robust_real == sweep_lp[ticker]["realistic"]["robustness_verdict"] else "MISMATCH"
        status_zero = "MATCH" if robust_zero == sweep_lp[ticker]["zero"]["robustness_verdict"] else "MISMATCH"
        if status_real != "MATCH" or status_zero != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: real edge_counts={} -> {} vs published {} -> {}; "
              "zero edge_counts={} -> {} vs published {} -> {}".format(
            ticker, edge_counts_real, robust_real, sweep_lp[ticker]["realistic"]["robustness_verdict"], status_real,
            edge_counts_zero, robust_zero, sweep_lp[ticker]["zero"]["robustness_verdict"], status_zero))

    print("\nDeterminism: recomputing AMZN lookback=5 medians twice (realistic costs).")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, REALISTIC_COSTS)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, REALISTIC_COSTS)[0], 3)
          for lab, s, e in runs]
    print("  AMZN r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print("\nDeterminism: recomputing AMZN lookback=5 medians twice (zero costs).")
    m1 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, ZERO_COSTS)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, ZERO_COSTS)[0], 3)
          for lab, s, e in runs]
    print("  AMZN r1: {}; r2: {} -> {}".format(m1, m2,
          "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print("\n")
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "lookback-5 column cross-checks against momentum_results.json, "
              "and robustness verdicts reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on "
          "AMZN/JPM + cross-consistency vs momentum_results.json: exploratory "
          "simulation.")


if __name__ == "__main__":
    main()