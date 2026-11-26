"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Four verification gates, all independent of the check's own code path:

1. Fresh recomputation (AMZN, JPM; all lookbacks; all cost levels) via a
   separate implementation path: re-implemented volatility blocks and segment
   reconstruction, fresh momentum signals, direct ``walk_forward`` (no call to
   ``stress_segments``), and the framework ``noise_benchmark`` called directly
   for the matched null. Every per-segment candidate median and null median is
   compared to the artifact.

2. Cross-consistency against the prior zero-cost momentum artifacts: the
   zero-cost column of this artifact must reproduce
   ``momentum_results.json`` (lookback 5) and
   ``momentum_lookback_sweep_results.json`` (lookbacks 3/5/10) exactly. A
   mismatch would flag a change in the zero-cost computation.

3. Classification reproducibility: the EDGE/LOSS/NO_EDGE status, the per-cell
   cost_class, and the per-asset overall class are re-derived from the artifact
   medians/verdicts via the documented rule, independently of the check's own
   classification code, for all 10 tickers.

4. Determinism: the AMZN lookback-5 realistic cell is recomputed twice.
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
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

# Independent re-declaration of the predeclared cost grid (do not import from
# the check; a typo in the check's config must be caught here).
COST_CFG = {
    "zero": bt.BacktestConfig(),
    "realistic": bt.BacktestConfig(
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005),
    "double": bt.BacktestConfig(
        commission_per_trade=4.0, commission_per_share=0.006,
        slippage_cents=4.0, slippage_proportional=0.0010),
}


def vol_blocks(closes, n_blocks, window):
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
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_cell(ticker, s, e, lookback, cfg):
    """Fresh candidate median + matched null median for one segment/cell."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    res = bt.walk_forward(
        seg_bars, sig[s:e], train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST, warmup=0, overlap_window=0,
        cfg=cfg, periods_per_year=252)
    return float(np.median(cand)), float(noise.baseline_median_log_return)


def edge_status(medians, verdict):
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cost_class(ez, er):
    if ez == "EDGE" and er == "EDGE":
        return "COST_ROBUST"
    if ez == "EDGE" and er != "EDGE":
        return "COST_FRAGILE"
    return "NO_EDGE"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    all_ok = True

    print("=== Gate 0: signal characterization (independent weight recount) ===")
    n_degenerate = 0
    for ticker, pub in art["signal_profile"].items():
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = momentum_signals(closes, 5)
        n_pos = sum(1 for s in sig if s.weight > 0)
        n_neg = sum(1 for s in sig if s.weight < 0)
        nonneutral = n_pos + n_neg
        frac_long = round(n_pos / nonneutral, 4) if nonneutral else 0.0
        ok = (n_pos == pub["n_long"] and n_neg == pub["n_short"]
              and frac_long == pub["frac_long"])
        if not ok:
            all_ok = False
        if frac_long > 0.99:
            n_degenerate += 1
        print("  {}: long={} short={} frac_long={} vs published long={} short={} "
              "frac_long={} -> {}".format(
                  ticker, n_pos, n_neg, frac_long, pub["n_long"], pub["n_short"],
                  pub["frac_long"], "MATCH" if ok else "MISMATCH"))
    print("  assets effectively always-long (frac_long>0.99): {}/10".format(n_degenerate))
    if n_degenerate >= 8:
        print("  FINDING: the existing momentum signal is degenerate (near-constant")
        print("  long) for most assets; the cost classification therefore describes a")
        print("  constant-long position, not a short-horizon momentum direction bet.")

    print("\n=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks/costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for level in ("zero", "realistic", "double"):
                cfg = COST_CFG[level]
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n = segment_cell(asset, s, e, lb, cfg)
                    meds.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["per_cell"][f"{lb}|{asset}"][level]
                status = ("MATCH"
                          if meds == pub["medians"]
                          and nulls == pub["null_medians"]
                          and lbls == pub["segments"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {} lb={} {}: {} recomputed medians {} vs {} | nulls {} vs {}".format(
                        asset, lb, level, status, meds, pub["medians"], nulls, pub["null_medians"]))
                else:
                    print("  {} lb={} {:9s} segments={} -> {}".format(asset, lb, level, lbls, status))

    print("\n=== Gate 2: zero-cost cross-consistency vs prior momentum artifacts ===")
    mom = load(MOM_ARTIFACT)
    sweep = load(SWEEP_ARTIFACT)
    for ticker in mom["universe"]["per_asset"]:
        pub5 = art["per_cell"][f"5|{ticker}"]["zero"]
        ref = mom["universe"]["per_asset"][ticker]
        ok = (pub5["medians"] == ref["medians"]
              and pub5["null_medians"] == ref["null_medians"]
              and pub5["verdict"] == ref["verdict"])
        all_ok = all_ok and ok
        print("  {} lookback5 zero vs momentum_results.json -> {}".format(
            ticker, "MATCH" if ok else "MISMATCH"))
    for ticker in sweep["per_asset"]:
        for lb in LOOKBACKS:
            pub = art["per_cell"][f"{lb}|{ticker}"]["zero"]
            ref = sweep["per_asset"][ticker][str(lb)]
            ok = (pub["medians"] == ref["medians"]
                  and pub["null_medians"] == ref["null_medians"]
                  and pub["verdict"] == ref["verdict"])
            if not ok:
                all_ok = False
                print("  {} lb={} zero vs lookback_sweep -> MISMATCH".format(ticker, lb))
    print("  all zero-cost cells cross-checked against momentum_results.json and "
          "momentum_lookback_sweep_results.json")

    print("\n=== Gate 3: classification reproducibility (all 10 tickers) ===")
    for ticker in art["summary"]["per_ticker"]:
        classes = []
        for lb in LOOKBACKS:
            c = art["per_cell"][f"{lb}|{ticker}"]
            ez = edge_status(c["zero"]["medians"], c["zero"]["verdict"])
            er = edge_status(c["realistic"]["medians"], c["realistic"]["verdict"])
            ed = edge_status(c["double"]["medians"], c["double"]["verdict"])
            cc = cost_class(ez, er)
            ok = (ez == c["edge_zero"] and er == c["edge_realistic"]
                  and ed == c["edge_double"] and cc == c["cost_class"])
            if not ok:
                all_ok = False
                print("  {} lb={}: recomputed e0={} er={} ed={} cc={} vs published "
                      "e0={} er={} ed={} cc={} -> MISMATCH".format(
                          ticker, lb, ez, er, ed, cc, c["edge_zero"],
                          c["edge_realistic"], c["edge_double"], c["cost_class"]))
            classes.append(cc)
        n_robust = classes.count("COST_ROBUST")
        overall = ("COST_ROBUST" if n_robust >= 2
                   else ("COST_FRAGILE" if "COST_FRAGILE" in classes else "NO_EDGE"))
        pub_overall = art["summary"]["per_ticker"][ticker]["overall"]
        ok = overall == pub_overall
        if not ok:
            all_ok = False
        print("  {}: classes={} -> overall={} vs published {} -> {}".format(
            ticker, classes, overall, pub_overall, "MATCH" if ok else "MISMATCH"))

    print("\nDeterminism: recomputing AMZN lookback5 realistic medians twice.")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    m1 = [round(segment_cell("AMZN", s, e, 5, COST_CFG["realistic"])[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_cell("AMZN", s, e, 5, COST_CFG["realistic"])[0], 3)
          for lab, s, e in runs]
    det = m1 == m2
    print("  AMZN lb5 realistic r1: {}; r2: {} -> {}".format(
        m1, m2, "identical" if det else "DIFFERENT"))
    if not det:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "zero-cost column cross-checks against the prior momentum "
              "artifacts, and the cost-robustness classification reproduces.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()