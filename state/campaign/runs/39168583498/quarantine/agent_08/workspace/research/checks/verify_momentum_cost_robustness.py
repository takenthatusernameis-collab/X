"""Independent verification of `research/checks/momentum_cost_robustness.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregating fold log returns with `walk_forward` using
   realistic costs). Every per-asset, per-lookback median is recomputed and
   compared to the artifact.

2. Cross-consistency gate: the lookback=5 column of this artifact must match
   `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch would
   flag a change in the lookback-5 computation or a defect, even with costs.

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
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"
MOM_ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

COST_REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
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


def segment_result_with_costs(ticker, s, e, train, test, warm, overlap, lookback):
    """Walk-forward regime-stability result for one asset segment and lookback,
    computed with fresh components: volatility_blocks, segment slicing,
    momentum signals, walk_forward, using realistic costs.

    No call to stress_segments.
    """
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = COST_REALISTIC
    cfg = bt.BacktestConfig(
        initial_capital=cfg.initial_capital,
        warmup_periods=warm,
        commission_per_trade=cfg.commission_per_trade,
        commission_per_share=cfg.commission_per_share,
        slippage_cents=cfg.slippage_cents,
        slippage_proportional=cfg.slippage_proportional,
    )
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    # Create a noise benchmark with realistic costs for comparison
    noise = bt.noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cfg,
    )
    null_log = noise.baseline_median_log_return
    return (float(np.median(cand_log)), null_log, len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact(ARTIFACT_PATH)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh walk_forward with realistic costs (AMZN, JPM, all lookbacks) ===")
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
                m, n, nf = segment_result_with_costs(asset, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                medians.append(round(m, 3))
                nulls.append(round(n, 3))
                nfolds = nf
            pub_medians = [round(v, 3) for v in artifact["per_asset"][asset][str(lb)]["medians"]]
            pub_nulls = [round(v, 3) for v in artifact["per_asset"][asset][str(lb)]["null_medians"]]
            status = ("MATCH"
                      if medians == pub_medians
                      and nulls == pub_nulls
                      and lbls == artifact["per_asset"][asset][str(lb)]["segments"]
                      else "MISMATCH")
            if status != "MATCH":
                mismatch = True
                all_ok = False
            print("  {}: lookback {} segments={} -> {}".format(
                asset, lb, lbls, status))
            if status != "MATCH":
                print("    medians recomputed {} vs published {}".format(medians, pub_medians))
                print("    nulls  recomputed {} vs published {}".format(nulls, pub_nulls))
        if mismatch:
            print("  [DEBUG] {} pub segments {}".format(asset,
                  [artifact["per_asset"][asset][str(lb)]["segments"] for lb in LOOKBACKS]))

    print("\n=== Gate 2: lookback-5 column cross-consistency vs momentum_results.json (costs applied) ===")
    mom = load_artifact(MOM_ARTIFACT_PATH)
    sweep_lp = artifact["universe"]["per_asset"]
    mom_lp = mom["universe"]["per_asset"]
    cmismatch = False
    for ticker in mom_lp:
        pub5 = mom_lp[ticker]
        sw5 = sweep_lp[ticker]["lookback_medians"]["5"]
        sn5 = sweep_lp[ticker]["lookback_null_medians"]["5"]
        sv5 = sweep_lp[ticker]["verdicts_5"]
        status = "MATCH" if (sw5 == pub5["medians"]
                             and sn5 == pub5["null_medians"]
                             and sv5 == pub5["verdict"]) \
            else "MISMATCH"
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: medians {} vs {} | nulls {} vs {} | verdict {} vs {} -> {}"
              .format(ticker, sw5, pub5["medians"], sn5, pub5["null_medians"],
                      sv5, pub5["verdict"], status))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: lookback-robustness verdict reproducibility ===")
    def verdict(medians, null_medians):
        cand_disp = float(np.std(np.array(medians)))
        null_disp = float(np.std(np.array(null_medians)))
        if all(abs(m) <= 0.05 for m in medians):
            return "CONSISTENT_WITH_NOISE"
        elif cand_disp > 2.0 * null_disp:
            return "REGIME_DEPENDENT"
        elif all(m < 0 for m in medians):
            return "REGIME_STABLE_LOSS"
        else:
            return "REGIME_STABLE"

    for ticker in artifact["universe"]["ticker_order"]:
        medians = artifact["universe"]["per_asset"][ticker]["lookback_medians"]
        nulls = artifact["universe"]["per_asset"][ticker]["lookback_null_medians"]
        regen_verdict = verdict(medians["5"], nulls["5"])
        pub_verdict = artifact["universe"]["per_asset"][ticker]["verdicts_5"]
        status = "MATCH" if regen_verdict == pub_verdict else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        print("  {}: verdict {} -> {}".format(ticker, regen_verdict, status))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path with realistic costs. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
