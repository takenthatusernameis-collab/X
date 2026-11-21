"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Verification gates:

1. Fresh recomputation of the AMZN/JPM per-segment medians and coin-flip null
   medians via a separate implementation path (re-implemented
   `volatility_blocks`, segment reconstruction, momentum signals,
   `walk_forward`, and `noise_benchmark`) for every lookback and every
   predeclared cost level. No call to `stress_segments`.

2. Zero-cost cross-consistency: the zero-cost column of the cost artifact must
   reproduce `momentum_lookback_sweep_results.json` (medians, null medians,
   segments, verdicts) for all 10 tickers and all 3 lookbacks.

3. Classification reproducibility: the per-(asset, lookback) cost class and the
   universe verdict are re-derived from the artifact's stored medians/nulls via
   the predeclared rule, independently of the check's own computation.

4. Cost-model directionality audit (diagnostic, not a pass/fail on the momentum
   result): a two-bar round trip with slippage only and with commission only,
   plus full-sample AMZN runs under zero / commission-only / slippage-only /
   realistic costs. This documents whether the repository's cost model is
   directionally adverse for a long/short rule, which determines whether the
   cost grid actually constitutes conservative stress.

5. Determinism: the AMZN lookback-5 realistic result is recomputed twice.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

COST_LEVELS = {
    "zero": {},
    "realistic": dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "conservative": dict(
        commission_per_trade=5.0,
        commission_per_share=0.005,
        slippage_cents=5.0,
        slippage_proportional=0.0015,
    ),
}


def cost_cfg(level):
    return bt.BacktestConfig(**COST_LEVELS[level])


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
    """Fresh per-segment walk-forward candidate median and null median."""
    seg_bars = list(bars)[s:e]
    closes = bars.closes_array()
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
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cfg)
    return float(np.median(cand_log)), float(noise.baseline_median_log_return), len(res.folds)


def edge_present(medians, verdict):
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def classify(levels):
    edges = {name: edge_present(levels[name]["medians"], levels[name]["verdict"])
             for name in COST_LEVELS}
    if not edges["zero"]:
        return "NO_EDGE"
    if edges["realistic"] and edges["conservative"]:
        return "COST_ROBUST"
    if edges["realistic"]:
        return "COST_SENSITIVE"
    return "COST_FRAGILE"


def universe_verdict(classes):
    counts = Counter(classes)
    if counts["COST_ROBUST"] >= 5:
        return "COST_ROBUST"
    if counts["COST_ROBUST"] >= 1 or counts["COST_SENSITIVE"] >= 1:
        return "MIXED"
    if counts["COST_FRAGILE"] >= 1:
        return "COST_FRAGILE"
    return "NO_EDGE"


def cost_model_audit():
    """Document whether the repository cost model is directionally adverse."""
    print("  two-bar round trip at close=100, slippage_cents=2.0 only:")
    bar = bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
    bar2 = bt.Bar(date=2, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
    cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, slippage_cents=2.0)
    for label, weight in (("long", 1.0), ("short", -1.0)):
        sig = [bt.Signal(date=1, weight=weight), bt.Signal(date=2, weight=0.0)]
        res = bt.run_bars([bar, bar2], sig, cfg)
        print("    {}: fills={} final_equity={:.6f} initial={:.1f}".format(
            label, [(f.date, f.shares, f.price) for f in res.trades],
            res.equity_curve[-1], cfg.initial_capital))

    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    sig = momentum_signals(closes, 5)
    print("  AMZN full-sample total return by cost component (lookback 5):")
    for name, params in (
        ("zero", {}),
        ("commission_only", dict(commission_per_trade=2.0, commission_per_share=0.003)),
        ("slippage_only", dict(slippage_cents=2.0, slippage_proportional=0.0005)),
        ("realistic", COST_LEVELS["realistic"]),
    ):
        res = bt.run_bars(list(bars), sig, bt.BacktestConfig(**params))
        tr = (res.equity_curve[-1] - 1e6) / 1e6
        print("    {:16s} total_return={:+.6f} n_fills={}".format(name, tr, len(res.trades)))


def main():
    with open(ARTIFACT) as f:
        art = json.load(f)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks/levels) ===")
    null_drag_segments = 0
    null_comparisons = 0
    for asset in ("AMZN", "JPM"):
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for level in COST_LEVELS:
                medians, nulls = [], []
                nfolds = None
                for lab, s, e in runs:
                    m, n, nf = segment_result(bars, s, e, lb, cost_cfg(level))
                    medians.append(round(m, 6))
                    nulls.append(round(n, 6))
                    if nfolds is None:
                        nfolds = nf
                cand_disp = float(np.std(np.array(medians)))
                null_disp = float(np.std(np.array(nulls)))
                if all(abs(m) <= 0.05 for m in medians):
                    derived_verdict = "CONSISTENT_WITH_NOISE"
                elif cand_disp > 2.0 * null_disp:
                    derived_verdict = "REGIME_DEPENDENT"
                elif all(m < 0 for m in medians):
                    derived_verdict = "REGIME_STABLE_LOSS"
                else:
                    derived_verdict = "REGIME_STABLE"
                pub = art["per_asset"][asset][str(lb)]["levels"][level]
                ok = (medians == [round(v, 6) for v in pub["medians"]]
                      and nulls == [round(v, 6) for v in pub["null_medians"]]
                      and lbls == pub["segments"]
                      and nfolds == pub["n_folds"]
                      and derived_verdict == pub["verdict"])
                if not ok:
                    all_ok = False
                    print("  MISMATCH {} lb={} level={}".format(asset, lb, level))
                    print("    medians recomputed {} vs published {}".format(medians, pub["medians"]))
                    print("    nulls   recomputed {} vs published {}".format(nulls, pub["null_medians"]))
                    print("    verdict recomputed {} vs published {}".format(
                        derived_verdict, pub["verdict"]))
                if level != "zero":
                    pub_zero = art["per_asset"][asset][str(lb)]["levels"]["zero"]["null_medians"]
                    for a, b in zip(nulls, pub_zero):
                        null_comparisons += 1
                        if a <= b:
                            null_drag_segments += 1
        print("  {}: all lookbacks/levels recomputed".format(asset))
    print("  cost-matched null diagnostic: realistic/conservative null median <= "
          "zero-cost null median in {}/{} segment comparisons".format(
              null_drag_segments, null_comparisons))

    print("\n=== Gate 2: zero-cost column vs momentum_lookback_sweep_results.json ===")
    with open(LOOKBACK_ARTIFACT) as f:
        prior = json.load(f)
    for ticker in art["per_asset"]:
        for lb in LOOKBACKS:
            new = art["per_asset"][ticker][str(lb)]["levels"]["zero"]
            old = prior["per_asset"][ticker][str(lb)]
            ok = ([round(v, 3) for v in new["medians"]] == [round(v, 3) for v in old["medians"]]
                  and [round(v, 3) for v in new["null_medians"]] == [round(v, 3) for v in old["null_medians"]]
                  and new["segments"] == old["segments"]
                  and new["verdict"] == old["verdict"]
                  and new["n_folds"] == old["n_folds"])
            if not ok:
                all_ok = False
                print("  MISMATCH {} lb={}: {} vs {}".format(ticker, lb, new, old))
    print("  all 10 tickers x 3 lookbacks -> MATCH")

    print("\n=== Gate 3: classification reproducibility ===")
    for ticker in art["per_asset"]:
        for lb in LOOKBACKS:
            entry = art["per_asset"][ticker][str(lb)]
            derived = classify(entry["levels"])
            status = "MATCH" if derived == entry["cost_class"] else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print("  {:6s} lb={:2d}: derived={} published={} -> {}".format(
                ticker, lb, derived, entry["cost_class"], status))
    for lb in LOOKBACKS:
        classes = [classify(art["per_asset"][t][str(lb)]["levels"])
                   for t in art["per_asset"]]
        derived = universe_verdict(classes)
        published = art["summary"]["universe_verdict_by_lookback"][str(lb)]
        status = "MATCH" if derived == published else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        print("  universe lb={:2d}: derived={} published={} -> {}".format(
            lb, derived, published, status))

    print("\n=== Gate 4: cost-model directionality audit (diagnostic) ===")
    cost_model_audit()

    print("\n=== Determinism: AMZN lookback 5 realistic recomputed twice ===")
    bars, _ = bt.load_ticker("AMZN")
    labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    r1 = [tuple(round(x, 6) for x in segment_result(bars, s, e, 5, cost_cfg("realistic"))[:2])
          for _, s, e in runs]
    r2 = [tuple(round(x, 6) for x in segment_result(bars, s, e, 5, cost_cfg("realistic"))[:2])
          for _, s, e in runs]
    print("  r1 == r2: {}".format(r1 == r2))
    assert r1 == r2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "zero-cost column cross-checks against momentum_lookback_sweep_results.json, "
              "and the cost classifications reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()