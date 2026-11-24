"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

This is a separate implementation path: it recomputes each segment's
walk-forward median and the coin-flip null from a fresh `walk_forward` call
(not ``stress_segments``), for every asset / lookback / cost level, then
compares against the artifact written by the cost-sensitivity check
(`state/check_artifacts/momentum_cost_sensitivity_results.json`).

Independent path design (mirrors verify_momentum_longer_horizon.py):
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``vol_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Candidate medians are computed via ``walk_forward`` directly on each segment
  slice with the cost config; the coin-flip null is recomputed through the
  framework's own ``noise_benchmark`` (a separate code path from
  ``stress_segments``), with the SAME cost config.
- Verdicts are regenerated from the FRESH UNROUNDED medians and nulls via the
  documented rules, so verdicts decided on unrounded numbers are not confused
  by a 3-dp rounding round-trip.
- The full cost-robustness classification is regenerated from scratch using
  the same predeclared thresholds and compared to the artifact.

Gates:
- Gate 1: AMZN / JPM, all lookbacks, all cost levels -> fresh medians/nulls
  MATCH the artifact.
- Gate 2: all 10 assets, all lookbacks, all cost levels -> fresh medians/nulls
  MATCH; regenerated verdicts / edge counts / robustness verdicts MATCH; and
  the regenerated universe-level cost-robustness classification MATCHes.
- Gate 3: determinism of the verification path.
- Gate 4: zero-cost lookback-5 medians equal the prior independently verified
  lookback-sweep artifact (external consistency of the C0 baseline).

This keeps the same inputs (dataset, seed, windows, segments, cost grid) so a
match confirms the check computed from those inputs; a mismatch would flag an
error in the check's pipeline.
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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
PRIOR_LOOKBACK_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
SWEEP_VERDICT_TOL = 0.05

COST_GRID = {
    "zero": dict(),
    "realistic": dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "conservative2x": dict(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    ),
}
COST_ORDER = ["zero", "realistic", "conservative2x"]
COST_ROBUST_FRACTION_THRESHOLD = 0.5


def make_cfg(cost_name):
    return bt.BacktestConfig(**COST_GRID[cost_name])


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


def segment_result(ticker, s, e, lookback, cfg):
    """Walk-forward candidate median + matched coin-flip null for one segment.

    Uses fresh components: volatility blocks, segment slicing, momentum
    signals, walk_forward, and the framework noise_benchmark. No call to
    stress_segments.
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
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def regime_verdict(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= SWEEP_VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def edge_of(medians, verdict):
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness(edge_by_lb):
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[edge_by_lb[str(lb)]] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


def fresh_cost_cell(ticker, lookback, cost_name, runs):
    """Fresh per-segment medians/nulls + regenerated verdict for one cell."""
    cfg = make_cfg(cost_name)
    medians, nulls = [], []
    for lab, s, e in runs:
        m, n, _ = segment_result(ticker, s, e, lookback, cfg)
        medians.append(m)
        nulls.append(n)
    verdict = regime_verdict(medians, nulls)
    return medians, nulls, verdict


def main():
    np.random.seed(42)
    artifact = json.load(open(ARTIFACT_PATH))
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS), artifact["lookbacks"]

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
            for cname in COST_ORDER:
                meds, nls, _ = fresh_cost_cell(asset, lb, cname, runs)
                pub = artifact["per_asset"][asset][str(lb)][cname]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                status = ("MATCH"
                          if [round(v, 3) for v in meds] == pub_medians
                          and [round(v, 3) for v in nls] == pub_nulls
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {}: lb {} cost {} -> {}".format(asset, lb, cname, status))
                    print("    medians recomputed {} vs published {}".format(
                        [round(v, 3) for v in meds], pub_medians))
                    print("    nulls   recomputed {} vs published {}".format(
                        [round(v, 3) for v in nls], pub_nulls))
    print("  Gate 1 complete (only mismatches printed above).")

    print("\n=== Gate 2: all 10 assets; fresh medians/nulls + verdict "
          "regeneration + full cost-robustness classification ===")
    regen = {}  # regen[cname][ticker] = {lb: (medians, nulls, verdict)}
    for ticker in artifact["universe"]["per_cost"]["zero"]["ticker_order"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        ticker_ok = True
        for cname in COST_ORDER:
            regen.setdefault(cname, {})
            regen[cname][ticker] = {}
            for lb in LOOKBACKS:
                meds, nls, verdict = fresh_cost_cell(ticker, lb, cname, runs)
                pub = artifact["per_asset"][ticker][str(lb)][cname]
                pub_medians = [round(v, 3) for v in pub["medians"]]
                pub_nulls = [round(v, 3) for v in pub["null_medians"]]
                if ([round(v, 3) for v in meds] != pub_medians
                        or [round(v, 3) for v in nls] != pub_nulls):
                    ticker_ok = False
                    all_ok = False
                    print("  [DEBUG] {} lb {} cost {}: fresh medians={} nulls={} vs published medians={} nulls={}"
                          .format(ticker, lb, cname, [round(v, 3) for v in meds],
                                  [round(v, 3) for v in nls], pub_medians, pub_nulls))
                if verdict != pub["verdict"]:
                    ticker_ok = False
                    all_ok = False
                    print("  [DEBUG] {} lb {} cost {}: regenerated verdict {} vs published {}"
                          .format(ticker, lb, cname, verdict, pub["verdict"]))
                regen[cname][ticker][str(lb)] = (meds, nls, verdict)
        # Regenerate robustness verdict and compare to the artifact's universe view.
        for cname in COST_ORDER:
            edge_by_lb = {str(lb): edge_of(regen[cname][ticker][str(lb)][0],
                                          regen[cname][ticker][str(lb)][2])
                          for lb in LOOKBACKS}
            rob, counts = robustness(edge_by_lb)
            pub_rob = artifact["universe"]["per_cost"][cname]["per_asset"][ticker]["robustness_verdict"]
            if rob != pub_rob:
                ticker_ok = False
                all_ok = False
                print("  [DEBUG] {} cost {}: regenerated robustness {} vs published {}"
                      .format(ticker, cname, rob, pub_rob))
        if not ticker_ok:
            print("  {}: MISMATCH".format(ticker))
    print("  Gate 2 asset-level recomputation complete (only mismatches printed above).")

    # Regenerate the full universe-level cost-robustness classification.
    ticker_order = artifact["universe"]["per_cost"]["zero"]["ticker_order"]
    zero_rob = {t: robustness({str(lb): edge_of(regen["zero"][t][str(lb)][0],
                                                 regen["zero"][t][str(lb)][2])
                               for lb in LOOKBACKS})[0] for t in ticker_order}
    real_rob = {t: robustness({str(lb): edge_of(regen["realistic"][t][str(lb)][0],
                                                 regen["realistic"][t][str(lb)][2])
                               for lb in LOOKBACKS})[0] for t in ticker_order}
    cons_rob = {t: robustness({str(lb): edge_of(regen["conservative2x"][t][str(lb)][0],
                                                 regen["conservative2x"][t][str(lb)][2])
                               for lb in LOOKBACKS})[0] for t in ticker_order}
    E0 = [t for t in ticker_order if zero_rob[t] == "ROBUST"]
    real_robust = [t for t in E0 if real_rob[t] == "ROBUST"]
    cons_robust = [t for t in E0 if cons_rob[t] == "ROBUST"]
    n_e0 = len(E0)
    frac = (len(real_robust) / n_e0) if n_e0 else 0.0
    if n_e0 == 0:
        classification = "COST_FRAGILE"
    elif frac >= COST_ROBUST_FRACTION_THRESHOLD:
        classification = "COST_ROBUST"
    elif len(real_robust) == 0:
        classification = "COST_FRAGILE"
    else:
        classification = "COST_SENSITIVE"
    pub_cr = artifact["cost_robustness"]
    cls_ok = (classification == pub_cr["classification"]
              and E0 == pub_cr["e0_zero_robust"]
              and real_robust == pub_cr["realistic_robust"]
              and cons_robust == pub_cr["conservative_robust"])
    if not cls_ok:
        all_ok = False
        print("  [DEBUG] classification mismatch: regenerated class={} E0={} real={} cons={}"
              .format(classification, E0, real_robust, cons_robust))
        print("          published: class={} E0={} real={} cons={}"
              .format(pub_cr["classification"], pub_cr["e0_zero_robust"],
                      pub_cr["realistic_robust"], pub_cr["conservative_robust"]))
    print("  Regenerated classification: {} | E0={} | realistic_robust={} | "
          "conservative_robust={} | MATCH={}"
          .format(classification, E0, real_robust, cons_robust, cls_ok))

    print("\n=== Gate 3: determinism of the verification path ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    r1 = [round(segment_result("AMZN", s, e, 5, make_cfg("realistic"))[0], 3)
          for lab, s, e in runs]
    r2 = [round(segment_result("AMZN", s, e, 5, make_cfg("realistic"))[0], 3)
          for lab, s, e in runs]
    print("  AMZN lb5 realistic r1: {}; r2: {} -> {}".format(
        r1, r2, "identical" if r1 == r2 else "DIFFERENT"))
    if r1 != r2:
        all_ok = False
    assert r1 == r2, "verification path not deterministic"

    print("\n=== Gate 4: zero-cost lookback-5 medians equal prior verified "
          "lookback-sweep artifact ===")
    if PRIOR_LOOKBACK_ARTIFACT.exists():
        prior = json.load(open(PRIOR_LOOKBACK_ARTIFACT))
        xcheck_ok = True
        for ticker in ticker_order:
            pub = prior["per_asset"][ticker]["5"]["medians"]
            mine = [round(v, 3) for v in regen["zero"][ticker]["5"][0]]
            if pub != mine:
                xcheck_ok = False
                all_ok = False
                print("  [MISMATCH] {}: fresh zero lb5 {} vs prior {}".format(ticker, mine, pub))
        print("  zero-cost lookback-5 medians vs prior verified artifact: {}".format(
            "ALL MATCH" if xcheck_ok else "MISMATCH FOUND"))
    else:
        print("  prior lookback artifact not found; Gate 4 SKIPPED")

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity check "
              "artifact; regenerated verdicts, robustness verdicts, and the "
              "universe-level cost-robustness classification all MATCH; and the "
              "verification path is deterministic.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on the "
          "collected universe: exploratory simulation.")


if __name__ == "__main__":
    main()