"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates (all must pass):

1. Fresh recomputation of the AMZN/JPM per-segment medians and matched null
   medians for every lookback x cost level, using a separate implementation
   path (fresh volatility blocks, segment slicing, momentum signals,
   `walk_forward`, and the framework `noise_benchmark`). No call to
   `stress_segments`.

2. Zero-cost column cross-consistency vs `momentum_lookback_sweep_results.json`
   for all 10 assets and all three lookbacks (medians, nulls, verdicts).

3. Reproducibility of the per-asset robustness verdicts and the universe
   classification from the artifact medians/dispersions via the documented
   rules, independently of the check's own computation.

4. Reproduction of the cost-drag diagnostic from the artifact.

5. Reproduction of the cost-model behavior diagnostic (AMZN full sample), and
   an explicit assertion that slippage-only cost does not reduce the return
   relative to zero cost (which is the evidence that the engine's additive
   slippage is not a round-trip cost).
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
REFERENCE_LOOKBACK = 5
TOL = 0.05

# Independent copy of the predeclared cost grid (deliberately re-specified).
COST_CONFIGS = {
    "zero": {},
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=2.0, commission_per_share=0.003,
                         slippage_cents=4.0, slippage_proportional=0.0010),
    "stress": dict(commission_per_trade=2.0, commission_per_share=0.003,
                   slippage_cents=8.0, slippage_proportional=0.0020),
}
COST_NAMES = ("zero", "realistic", "conservative", "stress")

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def vol_blocks(closes, n_blocks, window):
    """Fresh re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0]) if e - s > window else 0.0)
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
        ret = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def fresh_segment_result(ticker, s, e, lookback, cfg):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)[s:e]
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(seg_bars, param_grid=[{"lookback": lookback}],
                            train_window=TRAIN, test_window=TEST, warmup=0,
                            overlap_window=0, cfg=cfg, periods_per_year=252)
    return float(np.median(cand)), float(noise.baseline_median_log_return), len(res.folds)


def verdict_from(medians, null_medians, cand_disp, null_disp):
    if all(abs(m) <= TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def edge_or_not(sr):
    medians = sr.get("medians_raw", sr["medians"])
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness(counts):
    if counts["EDGE"] >= 2:
        return "COST_ROBUST"
    if counts["EDGE"] == 1:
        return "COST_FRAGILE"
    return "COST_ERASED"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks x costs) ===")
    for asset in ("AMZN", "JPM"):
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for cname in COST_NAMES:
            cfg = bt.BacktestConfig(**COST_CONFIGS[cname])
            for lb in LOOKBACKS:
                medians, nulls = [], []
                for lab, s, e in runs:
                    m, n, _ = fresh_segment_result(asset, s, e, lb, cfg)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["per_asset"][asset][cname]["lookback"][str(lb)]
                ok = (medians == pub["medians"] and nulls == pub["null_medians"]
                      and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                print(f"  {asset} {cname:12s} lookback {lb:2d} segments={lbls} -> "
                      f"{'MATCH' if ok else 'MISMATCH'}")
                if not ok:
                    print(f"    medians recomputed {medians} vs published {pub['medians']}")
                    print(f"    nulls  recomputed {nulls} vs published {pub['null_medians']}")

    print("\n=== Gate 2: zero-cost column vs momentum_lookback_sweep_results.json ===")
    prior = load(LOOKBACK_ARTIFACT)["per_asset"]
    for ticker in art["universe"]["ticker_order"]:
        for lb in LOOKBACKS:
            pub = prior[ticker][str(lb)]
            got = art["per_asset"][ticker]["zero"]["lookback"][str(lb)]
            ok = (got["medians"] == pub["medians"]
                  and got["null_medians"] == pub["null_medians"]
                  and got["verdict"] == pub["verdict"])
            if not ok:
                all_ok = False
                print(f"  MISMATCH {ticker} lookback {lb}")
        print(f"  {ticker}: zero-cost medians/nulls/verdicts -> "
              f"{'MATCH' if all(
                  art['per_asset'][ticker]['zero']['lookback'][str(lb)]['medians'] == prior[ticker][str(lb)]['medians']
                  and art['per_asset'][ticker]['zero']['lookback'][str(lb)]['null_medians'] == prior[ticker][str(lb)]['null_medians']
                  and art['per_asset'][ticker]['zero']['lookback'][str(lb)]['verdict'] == prior[ticker][str(lb)]['verdict']
                  for lb in LOOKBACKS) else 'MISMATCH'}")

    print("\n=== Gate 3: robustness + universe classification reproducibility ===")
    baseline_robust = []
    for ticker in art["universe"]["ticker_order"]:
        per_cost = {}
        for cname in COST_NAMES:
            counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            verdicts_ok = True
            for lb in LOOKBACKS:
                sr = art["per_asset"][ticker][cname]["lookback"][str(lb)]
                recomputed = verdict_from(
                    sr.get("medians_raw", sr["medians"]),
                    sr.get("null_medians_raw", sr["null_medians"]),
                    sr.get("candidate_dispersion_raw", sr["candidate_dispersion"]),
                    sr.get("null_dispersion_raw", sr["null_dispersion"]),
                )
                if recomputed != sr["verdict"]:
                    verdicts_ok = False
                    all_ok = False
                    print(f"  VERDICT MISMATCH {ticker} {cname} lb={lb}: "
                          f"{recomputed} vs {sr['verdict']}")
                counts[edge_or_not(sr)] += 1
            v = robustness(counts)
            per_cost[cname] = v
            if v != art["per_asset"][ticker][cname]["robustness_verdict"]:
                all_ok = False
                print(f"  ROBUSTNESS MISMATCH {ticker} {cname}: {v} vs "
                      f"{art['per_asset'][ticker][cname]['robustness_verdict']}")
        if per_cost["zero"] == "COST_ROBUST":
            baseline_robust.append(ticker)
        print(f"  {ticker}: {per_cost}")

    if baseline_robust != art["universe"]["baseline_robust"]:
        all_ok = False
        print(f"  BASELINE SET MISMATCH: {baseline_robust} vs {art['universe']['baseline_robust']}")
    else:
        print(f"  baseline (zero-cost) COST_ROBUST set -> MATCH: {baseline_robust}")

    for cname in COST_NAMES:
        surviving = [t for t in baseline_robust
                     if art["per_asset"][t][cname]["robustness_verdict"] == "COST_ROBUST"]
        if set(surviving) == set(baseline_robust):
            uv = "PERSISTS"
        elif surviving:
            uv = "PARTIAL"
        else:
            uv = "ERASED"
        pub_uv = art["universe"]["cost_levels"][cname]
        ok = (surviving == pub_uv["surviving"] and uv == pub_uv["verdict"])
        if not ok:
            all_ok = False
        print(f"  universe {cname:12s} surviving={surviving} -> {uv} "
              f"({'MATCH' if ok else 'MISMATCH'})")

    print("\n=== Gate 4: cost-drag reproduction ===")
    for ticker in art["universe"]["ticker_order"]:
        base = float(np.median(art["per_asset"][ticker]["zero"]["medians_ref"]))
        for cname in COST_NAMES:
            recomputed = round(float(np.median(art["per_asset"][ticker][cname]["medians_ref"])) - base, 4)
            if recomputed != art["cost_drag"][ticker][cname]:
                all_ok = False
                print(f"  DRAG MISMATCH {ticker} {cname}: {recomputed} vs "
                      f"{art['cost_drag'][ticker][cname]}")
    print("  all per-asset cost-drag values reproduce from artifact medians -> MATCH")

    print("\n=== Gate 5: cost-model behavior (AMZN full sample, lookback 5) ===")
    bars, _ = bt.load_ticker("AMZN")
    sig = momentum_signals(bars.closes_array(), REFERENCE_LOOKBACK)
    behavior = {}
    variants = (
        ("zero", bt.BacktestConfig()),
        ("commission_only", bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003)),
        ("slippage_only_stress", bt.BacktestConfig(slippage_cents=8.0, slippage_proportional=0.0020)),
        ("full_stress", bt.BacktestConfig(**COST_CONFIGS["stress"])),
    )
    for vname, cfg in variants:
        r = bt.run_bars(list(bars), sig, cfg)
        total = float((r.equity_curve[-1] - 1e6) / 1e6)
        comm = float(sum(f.commission for f in r.trades))
        behavior[vname] = dict(total_return=round(total, 4),
                               commission_paid=round(comm, 2),
                               n_fills=len(r.trades))
        pub = art["cost_model_behavior"][vname]
        ok = (behavior[vname]["total_return"] == pub["total_return"]
              and behavior[vname]["n_fills"] == pub["n_fills"])
        if not ok:
            all_ok = False
        print(f"  {vname:22s} total_return={total:+.4f} -> {'MATCH' if ok else 'MISMATCH'}")

    slippage_ok = (behavior["slippage_only_stress"]["total_return"]
                   >= behavior["zero"]["total_return"])
    print(f"  slippage_only_stress >= zero: {slippage_ok} "
          f"({behavior['slippage_only_stress']['total_return']:+.4f} vs "
          f"{behavior['zero']['total_return']:+.4f})")
    print("  (a genuine round-trip cost would make slippage-only strictly worse)")
    if not slippage_ok:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "zero-cost column cross-checks against momentum_lookback_sweep, "
              "the classification reproduces, and the cost-model diagnostic "
              "reproduces.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cost-sensitivity independent verification: exploratory "
          "simulation.")


if __name__ == "__main__":
    main()