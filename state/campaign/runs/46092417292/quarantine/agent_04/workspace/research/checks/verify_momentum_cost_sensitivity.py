"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Four verification gates, none of which call `stress_segments` (the check's own
path) for the recomputation:

1. Fresh recomputation (AMZN, JPM; all lookbacks; all cost levels) via a
   separate implementation path: re-implemented `volatility_blocks`, segment
   reconstruction, and momentum signals, aggregating fold log returns with
   `walk_forward` directly, and recomputing the coin-flip null with the
   framework's `noise_benchmark` (a separate code path from `stress_segments`).
   The per-segment candidate medians, null medians, and the verdict (re-derived
   from the recomputed medians/nulls via the documented rule) are compared to
   the artifact.

2. Cross-consistency gate: the zero-cost lookback-5 column of this artifact is
   an independent rerun of the admitted momentum lookback sweep, so it must
   match `momentum_lookback_sweep_results.json` exactly (medians, nulls,
   verdicts) for all 10 tickers. A mismatch would flag a change in the
   zero-cost computation.

3. Classification reproducibility: the cost-robustness verdict, `survives`,
   `survives_pos`, and `max_candidate_median_shift` are re-derived from the
   artifact cells via the documented rules, independently of the check's own
   `classify()`.

4. Determinism: the AMZN lookback-5 medians at all three cost levels are
   recomputed twice and asserted identical.
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
COST_LEVELS = ("zero", "realistic", "conservative")
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
VERDICT_TOL = 0.05


def cost_cfgs():
    return {
        "zero": bt.BacktestConfig(initial_capital=1e6),
        "realistic": bt.BacktestConfig(
            initial_capital=1e6, commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005),
        "conservative": bt.BacktestConfig(
            initial_capital=1e6, commission_per_trade=4.0, commission_per_share=0.006,
            slippage_cents=4.0, slippage_proportional=0.001),
    }


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


def segment_result(bars, s, e, lookback, cfg):
    """Candidate median + null median for one segment via fresh components.

    Uses walk_forward directly (not stress_segments) for the candidate and the
    framework noise_benchmark for the null, both with the same cfg.
    """
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
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST, warmup=0, overlap_window=0,
        cfg=cfg, periods_per_year=252)
    return float(np.median(cand_log)), float(noise.baseline_median_log_return), len(res.folds)


def derive_verdict(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def is_edge(cell):
    return cell["verdict"] == "REGIME_STABLE" and all(m > 0 for m in cell["medians"])


def positive_candidate(cell):
    return all(m > 0 for m in cell["medians"])


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["cost_levels"] == list(COST_LEVELS)
    cfgs = cost_cfgs()
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all "
          "lookbacks; all cost levels) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for cost in COST_LEVELS:
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = segment_result(bars, s, e, lb, cfgs[cost])
                    meds.append(round(m, 3))
                    nulls.append(round(n, 3))
                verdict = derive_verdict(meds, nulls)
                pub = art["per_asset"][asset]["cells"][str(lb)][cost]
                status = ("MATCH"
                          if meds == pub["medians"]
                          and nulls == pub["null_medians"]
                          and verdict == pub["verdict"]
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {} lb{} {}: {} | meds {} vs {} | nulls {} vs {} | "
                          "verdict {} vs {}".format(
                              asset, lb, cost, status, meds, pub["medians"],
                              nulls, pub["null_medians"], verdict, pub["verdict"]))
        print("  {}: all lookback x cost cells recomputed -> {}".format(
            asset, "MATCH" if all_ok else "SEE MISMATCH ABOVE"))

    print("\n=== Gate 2: zero-cost lookback-5 column cross-consistency vs "
          "momentum_lookback_sweep_results.json ===")
    sweep = load(SWEEP_ARTIFACT)
    sw_lp = sweep["universe"]["per_asset"]
    cmismatch = False
    for ticker in sw_lp:
        pub5 = sw_lp[ticker]
        z5 = art["per_asset"][ticker]["cells"]["5"]["zero"]
        status = ("MATCH" if z5["medians"] == pub5["lookback_medians"]["5"]
                  and z5["null_medians"] == pub5["lookback_null_medians"]["5"]
                  and z5["verdict"] == pub5["verdicts_5"]
                  else "MISMATCH")
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {}: medians {} vs {} | nulls {} vs {} | verdict {} vs {} -> {}"
              .format(ticker, z5["medians"], pub5["lookback_medians"]["5"],
                      z5["null_medians"], pub5["lookback_null_medians"]["5"],
                      z5["verdict"], pub5["verdicts_5"], status))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")

    print("\n=== Gate 3: cost-robustness classification reproducibility ===")
    class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    lb5_cons = 0
    for ticker in art["universe"]["ticker_order"]:
        cells = art["per_asset"][ticker]["cells"]
        survives = {c: sum(1 for lb in LOOKBACKS if is_edge(cells[str(lb)][c]))
                    for c in COST_LEVELS}
        survives_pos = {c: sum(1 for lb in LOOKBACKS if positive_candidate(cells[str(lb)][c]))
                        for c in COST_LEVELS}
        if survives["conservative"] >= 1:
            verdict = "COST_ROBUST"
        elif survives["zero"] >= 1:
            verdict = "COST_FRAGILE"
        else:
            verdict = "NO_EDGE"
        max_shift = 0.0
        for lb in LOOKBACKS:
            for a, b in zip(cells[str(lb)]["zero"]["medians"],
                            cells[str(lb)]["conservative"]["medians"]):
                max_shift = max(max_shift, abs(b - a))
        max_shift = round(max_shift, 4)
        lb5 = is_edge(cells["5"]["conservative"])
        pub = art["per_asset"][ticker]
        ok = (survives == pub["survives"]
              and survives_pos == pub["survives_pos"]
              and verdict == pub["cost_robustness"]
              and max_shift == pub["max_candidate_median_shift"]
              and lb5 == pub["lb5_survives_conservative"])
        if not ok:
            all_ok = False
        class_counts[verdict] += 1
        if lb5:
            lb5_cons += 1
        print("  {:6s}: survives={} pos={} shift={:.4f} verdict={} lb5cons={} -> {}"
              .format(ticker, survives, survives_pos, max_shift, verdict, lb5,
                      "MATCH" if ok else "MISMATCH"))
    u = art["universe"]
    uok = (class_counts == u["class_counts"]
           and lb5_cons == u["lb5_survives_conservative"])
    if not uok:
        all_ok = False
    print("  universe counts recomputed {} vs published {} | lb5_cons {} vs {} -> {}"
          .format(class_counts, u["class_counts"], lb5_cons,
                  u["lb5_survives_conservative"], "MATCH" if uok else "MISMATCH"))

    print("\n=== Gate 4: determinism (AMZN lookback 5, all cost levels, twice) ===")
    bars, dates = bt.load_ticker("AMZN")
    labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    for cost in COST_LEVELS:
        r1 = [round(segment_result(bars, s, e, 5, cfgs[cost])[0], 3) for _, s, e in runs]
        r2 = [round(segment_result(bars, s, e, 5, cfgs[cost])[0], 3) for _, s, e in runs]
        same = r1 == r2
        if not same:
            all_ok = False
        print("  {}: r1={} r2={} -> {}".format(cost, r1, r2,
              "identical" if same else "DIFFERENT"))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "zero-cost lookback-5 column cross-checks against "
              "momentum_lookback_sweep_results.json, the cost-robustness "
              "classification reproduces, and the path is deterministic.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()