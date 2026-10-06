"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Verification gates:

1. Fresh recomputation of AMZN/JPM candidate medians at every cost level
   via a separate implementation path (fresh `volatility_blocks`, segment
   reconstruction, fresh `momentum_signals`, `walk_forward`), plus the
   framework's own ``noise_benchmark`` for the null medians. Compared
   against the artifact medians/nulls per segment.

2. Cross-consistency: the zero-cost lookback-3/5/10 column of this artifact
   is an independent rerun of the momentum class and must match
   `momentum_lookback_sweep_results.json` exactly.

3. Cost-robustness verdict reproducibility: the artifact's
   cost-robustness verdicts are re-derived from the artifact's own
   medians/nulls/verdicts via the documented rule, independently of the
   check's verdict computation.

4. Data-integrity gate: each ticker's adjusted-close series must be
   internally consistent -- ``close / adjclose`` must be >= 1 at every bar
   and must not decrease over time (a split-adjusted series has a
   non-decreasing adjustment factor). A decrease indicates the series
   mixes adjustment bases or corrupted prices; in that case the artifact
   is flagged as computed on corrupted data and the verification fails
   regardless of recomputation MATCHes.

5. Determinism: recomputing AMZN lookback-5 medians twice produces
   identical output at every cost level.
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
REFERENCE_PRICE = 100.0
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = ARTIFACT / "momentum_lookback_sweep_results.json" if False else (
    Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json")

SWEEP_VERDICT_TOL = 0.05

COST_LEVELS = {
    "zero": bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0),
    "realistic": bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
                                   slippage_proportional=0.001, commission_per_share=0.005),
    "conservative": bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
                                      slippage_proportional=0.002, commission_per_share=0.010),
    "heavy": bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
                               slippage_proportional=0.005, commission_per_share=0.020),
}


def vol_blocks(closes, n_blocks, window):
    """Fresh implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    block_size = n // n_blocks
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
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test, warmup=0, overlap_window=0,
        periods_per_year=252, cfg=cfg)
    return (float(np.median(cand_log)), float(noise.baseline_median_log_return),
            len(res.folds))


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def check_data_integrity(manifest):
    """Verify each collected ticker's adjusted-close series is consistent.

    For any split-adjusted series, close/adjclose equals the cumulative
    split factor relative to the last date; this factor can only grow, so
    the ratio must be >= 1 and non-decreasing across the series. A decrease
    proves the series mixes adjustment bases or holds corrupted prices,
    and a value < 1 is impossible.
    """
    errors = []
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        loc = Path(entry["location"])
        lines = [ln for ln in loc.read_text().splitlines() if ln.strip()]
        vals = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) < 7:
                continue
            try:
                vals.append((parts[0], float(parts[4]), float(parts[5])))
            except ValueError:
                continue
        if not vals:
            errors.append(f"{ticker}: no price rows")
            continue
        ratios = []
        for d, c, a in vals:
            if a <= 0:
                errors.append(f"{ticker}: nonpositive adjclose on {d}")
                continue
            ratios.append(c / a)
        if ratios:
            if any(r < 1.0 for r in ratios):
                errors.append(f"{ticker}: close/adjclose < 1 at some bar "
                              f"(min {min(ratios):.4f})")
            if any(ratios[i + 1] < ratios[i] - 1e-9 for i in range(len(ratios) - 1)):
                first = min((i, r) for i, r in enumerate(ratios)
                            if i + 1 < len(ratios) and ratios[i + 1] < ratios[i])
                errors.append(
                    f"{ticker}: close/adjclose DECREASES over time "
                    f"(min {min(ratios):.4f} .. max {max(ratios):.4f}); "
                    f"first drop at index {first[0]}; "
                    f"series mixes adjustment bases or is corrupted")
        # manifest entry window must be compatible with the file length
        claimed_window = entry.get("dataset_id", "")
        if "to-2026-10-03" in claimed_window and len(vals) > 2000:
            # entries collected as yf-ohlcv-<TICKER>-2024-01-01-to-2026-10-03
            # should contain ~750 bars; 4000+ bars indicates spliced history
            if "2024-01-01" in claimed_window and len(vals) > 1500:
                errors.append(
                    f"{ticker}: manifest claims 2024-01-01-to-2026-10-03 "
                    f"(~750 bars) but file has {len(vals)} bars (2009-2026); "
                    "manifest window does not describe the data file")
    return errors


def main():
    artifact = load_artifact(ARTIFACT)
    assert artifact["dataset_id"] == DATASET_ID, artifact["dataset_id"]
    assert artifact["lookbacks"] == list(LOOKBACKS)

    manifest = json.load(open(Path.cwd() / "research" / "data" / "manifest.json"))
    integrity_errors = check_data_integrity(manifest)

    all_ok = True
    data_corrupted = bool(integrity_errors)

    # =========================================================================
    # Gate 1: fresh walk_forward + noise_benchmark recomputation (AMZN, JPM)
    # =========================================================================
    print("=== Gate 1: fresh recomputation per cost level (AMZN, JPM) ===")
    all_gates = {}
    if data_corrupted:
        print("  DATA-INTEGRITY-FAILURE detected (see Gate 4); recomputation still")
        print("  reported below but the artifact is computed on corrupted data.")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for name in COST_LEVELS:
            cfg = COST_LEVELS[name]
            mismatch = False
            for lab, s, e in runs:
                m, n, nf = segment_result(asset, s, e, TRAIN, TEST, WARM,
                                          OVERLAP, 5, cfg)
                idx = runs.index((lab, s, e))
                pub_m = [round(v, 3) for v in
                         artifact["per_asset"][asset]["5"][name]["medians"]]
                pub_n = [round(v, 3) for v in
                         artifact["per_asset"][asset]["5"][name]["null_medians"]]
                if round(m, 3) != pub_m[idx] or round(n, 3) != pub_n[idx]:
                    mismatch = True
                    all_ok = False
                    print("    MISMATCH {}: segment {} -> medians "
                          "{},{} vs published {},{}".format(
                              asset, lab, round(m, 3), round(n, 3),
                              pub_m[idx], pub_n[idx]))
            status = "MATCH" if not mismatch else "MISMATCH"
            all_gates[f"g1_{asset}_{name}"] = status
            print("  {} ({}): segments={} -> {}".format(
                asset, name, lbls, status))

    # =========================================================================
    # Gate 2: zero-cost column cross-consistency vs lookback sweep
    # =========================================================================
    print("\n=== Gate 2: zero-cost column vs momentum_lookback_sweep_results.json ===")
    sweep = load_artifact(SWEEP_ARTIFACT)
    cmismatch = False
    for ticker in sweep["per_asset"]:
        for lb in [str(x) for x in sweep["lookbacks"]]:
            pub5 = sweep["per_asset"][ticker][lb]
            sw5 = artifact["per_asset"][ticker][lb]["zero"]
            for k in ["medians", "null_medians", "candidate_dispersion",
                      "null_dispersion", "n_folds", "verdict", "segments"]:
                if sw5[k] != pub5[k]:
                    cmismatch = True
                    all_ok = False
                    all_gates[f"g2_{ticker}_{lb}_{k}"] = "MISMATCH"
    if not cmismatch:
        print("  all 10 tickers x 3 lookbacks -> MATCH")
    else:
        print("  MISMATCH FOUND")
    all_gates["g2"] = "MATCH" if not cmismatch else "MISMATCH"
    if cmismatch:
        all_ok = False

    # =========================================================================
    # Gate 3: cost-robustness verdict reproducibility
    # =========================================================================
    print("\n=== Gate 3: cost-robustness verdict reproducibility ===")
    def edge_counts(artifact_data, ticker):
        counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for name in ["zero", "realistic", "conservative", "heavy"]:
            for lb in LOOKBACKS:
                lb_data = artifact_data["per_asset"][ticker][str(lb)][name]
                if lb_data["verdict"] == "REGIME_STABLE" and all(
                        m > 0 for m in lb_data["medians"]):
                    counts["EDGE"] += 1
                elif lb_data["verdict"] == "REGIME_STABLE_LOSS":
                    counts["LOSS"] += 1
                else:
                    counts["NO_EDGE"] += 1
        return counts

    def cost_verdict(survives):
        if "conservative" in survives:
            return "cost-ROBUST"
        if "realistic" in survives:
            return "cost-SENSITIVE"
        return "cost-NO_EDGE"

    rmismatch = False
    for ticker in artifact["per_asset"]:
        edge_counts_check = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        survives = []
        for name in ["zero", "realistic", "conservative", "heavy"]:
            ec = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            for lb in LOOKBACKS:
                lb_data = artifact["per_asset"][ticker][str(lb)][name]
                if lb_data["verdict"] == "REGIME_STABLE" and all(
                        m > 0 for m in lb_data["medians"]):
                    ec["EDGE"] += 1
                elif lb_data["verdict"] == "REGIME_STABLE_LOSS":
                    ec["LOSS"] += 1
                else:
                    ec["NO_EDGE"] += 1
            if ec["EDGE"] >= 2:
                survives.append(name)
            if name == "zero":
                edge_counts_check = ec
        if cost_verdict(survives) != artifact["cost_robustness"][ticker]["verdict"]:
            rmismatch = True
            all_ok = False
            all_gates[f"g3_{ticker}"] = "MISMATCH"
            print("  {}: {} vs {} -> MISMATCH".format(
                ticker, cost_verdict(survives),
                artifact["cost_robustness"][ticker]["verdict"]))
        elif edge_counts_check != artifact["per_asset"][ticker]["edge_counts"]:
            rmismatch = True
            all_ok = False
            all_gates[f"g3_{ticker}"] = "MISMATCH"
    if not rmismatch:
        print("  all 10 tickers cost-robustness verdicts -> MATCH")
    all_gates["g3"] = "MATCH" if not rmismatch else "MISMATCH"
    if rmismatch:
        all_ok = False

    # =========================================================================
    # Gate 4: data integrity
    # =========================================================================
    print("\n=== Gate 4: data integrity of the collected universe ===")
    if integrity_errors:
        for err in integrity_errors[:10]:
            print("  [DATA-PROBLEM] " + err)
        print("  Artifact computed on corrupted/mismatched adjusted-close data; "
              "recomputed medians are internally reproducible but not economically "
              "trustworthy. Gate 4 -> FAILED.")
        all_gates["g4"] = "DATA-PROBLEM"
        all_ok = False
    else:
        print("  all 10 tickers: close/adjclose >= 1 and non-decreasing; "
              "manifest windows compatible -> MATCH")
        all_gates["g4"] = "MATCH"

    # =========================================================================
    # Gate 5: determinism
    # =========================================================================
    print("\n=== Gate 5: determinism (AMZN lookback-5 at each cost level) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    det_ok = True
    for name in COST_LEVELS:
        m1 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5,
                                   COST_LEVELS[name])[0], 3)
              for lab, s, e in runs]
        m2 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5,
                                   COST_LEVELS[name])[0], 3)
              for lab, s, e in runs]
        ok = m1 == m2
        det_ok = det_ok and ok
        all_gates[f"g5_{name}"] = "MATCH" if ok else "MISMATCH"
        if not ok:
            all_ok = False
            print("  {}: r1={} r2={} -> DIFFERENT".format(name, m1, m2))
    print("  determinism across all 4 cost levels -> {}".format(
        "identical" if det_ok else "DIFFERENT"))
    if not det_ok:
        all_ok = False

    print("\n=== Verification summary ===")
    for k, v in sorted(all_gates.items()):
        print("  {}: {}".format(k, v))

    print()
    if all_ok and not data_corrupted:
        print("All gates PASS: recomputations MATCH, zero-cost column cross-checks "
              "against momentum_lookback_sweep_results.json, cost-robustness "
              "verdicts reproduce, data integrity holds, and output is deterministic.")
    elif data_corrupted:
        print("DATA-INTEGRITY-FAILURE: the collected universe's adjusted-close "
              "series are corrupted/mismatched (close/adjclose decreases over time "
              "in some tickers, manifest windows contradict file contents). "
              "The cost-sensitivity artifact is internally reproducible and the "
              "recomputation gates match, but it was computed on corrupted data "
              "and its quantitative edge/cost-robustness claims are not trustworthy.")
        sys.exit(2)
    else:
        print("MISMATCH FOUND: one or more verification gates failed. "
              "Investigate before recording results.")
        sys.exit(1)

    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity verification on AMZN/JPM + "
          "cross-consistency + data-integrity: exploratory simulation.")


if __name__ == "__main__":
    main()
