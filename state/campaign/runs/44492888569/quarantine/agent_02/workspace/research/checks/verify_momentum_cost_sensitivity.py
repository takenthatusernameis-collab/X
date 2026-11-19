"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates:

1. Fresh recomputation of AMZN/JPM at every lookback and cost multiplier via a
   separate implementation path (re-implemented volatility blocks, segment
   reconstruction, and momentum signals, aggregating fold log returns with
   `walk_forward`; the null is recomputed through `noise_benchmark`). Every
   per-asset/per-lookback/per-cost median and null median is compared to the
   artifact.
2. Zero-cost cross-consistency: the 0.0x column must match
   `momentum_lookback_sweep_results.json` exactly for all 10 tickers and all
   three lookbacks.
3. Classification reproducibility: the predeclared rule is re-derived from the
   artifact's verdicts and segment medians and compared to the published
   classification and break-even multiplier.
4. Cost-model diagnostic: candidate medians are compared across 0.0x and 2.0x,
   and null medians are checked for bankruptcy-scale instability. This is a
   diagnostic, not a tuning step; it records why the classification must be
   read as conditional on the existing cost model.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
BASE_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
REFERENCE_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def cost_cfg(multiplier):
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=BASE_COSTS["commission_per_trade"] * float(multiplier),
        commission_per_share=BASE_COSTS["commission_per_share"] * float(multiplier),
        slippage_cents=BASE_COSTS["slippage_cents"] * float(multiplier),
        slippage_proportional=BASE_COSTS["slippage_proportional"] * float(multiplier),
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
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
            )
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
                if e - s > window
                else 0.0
            )
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


def fresh_segment_result(bars_list, closes, s, e, lookback, multiplier):
    """Walk-forward + noise benchmark for one segment via fresh components."""
    seg_bars = bars_list[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cost_cfg(multiplier)
    )
    cand_log = np.array(
        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
         for f in res.folds],
        dtype=np.float64,
    )
    noise = bt.noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0,
        periods_per_year=252,
        cfg=cost_cfg(multiplier),
    )
    return float(np.median(cand_log)), float(noise.baseline_median_log_return)


def edge_flag(rec):
    return rec["verdict"] == "REGIME_STABLE" and all(m > 0 for m in rec["medians"])


def classify(by_cost):
    e0 = edge_flag(by_cost["0.0"])
    e1 = edge_flag(by_cost["1.0"])
    e2 = edge_flag(by_cost["2.0"])
    if not e0:
        return "NO_PRE_COST_EDGE"
    if e1 and e2:
        return "COST_ROBUST"
    if e1:
        return "COST_RESILIENT_BASE_ONLY"
    return "COST_FRAGILE"


def break_even(by_cost):
    if edge_flag(by_cost["2.0"]):
        return "GE_2.0"
    for multiplier in reversed(COST_MULTIPLIERS[:-1]):
        if edge_flag(by_cost[str(multiplier)]):
            return str(multiplier)
    return "NONE"


def main():
    with open(ARTIFACT_PATH) as fh:
        art = json.load(fh)
    assert art["dataset_id"] == DATASET_ID

    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks/costs) ===")
    for asset in ("AMZN", "JPM"):
        bars, _ = bt.load_ticker(asset)
        bars_list = list(bars)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lookback in LOOKBACKS:
            published = art["per_asset"][asset][f"lookback_{lookback}"]["by_cost"]
            for multiplier in COST_MULTIPLIERS:
                medians = []
                nulls = []
                for _, s, e in runs:
                    m, n = fresh_segment_result(
                        bars_list, closes, s, e, lookback, multiplier
                    )
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub_medians = published[str(multiplier)]["medians"]
                pub_nulls = published[str(multiplier)]["null_medians"]
                status = (
                    "MATCH"
                    if medians == pub_medians
                    and nulls == pub_nulls
                    and lbls == published[str(multiplier)]["segments"]
                    else "MISMATCH"
                )
                if status != "MATCH":
                    all_ok = False
                print(
                    "  {} lb={} cost={} segments={} -> {} (cand {} vs {}, null {} vs {})".format(
                        asset, lookback, multiplier, lbls, status,
                        medians, pub_medians, nulls, pub_nulls,
                    )
                )

    print("\n=== Gate 2: zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    with open(REFERENCE_PATH) as fh:
        reference = json.load(fh)
    ref_universe = reference["per_asset"]
    for ticker in ref_universe:
        for lookback in LOOKBACKS:
            published = ref_universe[ticker][str(lookback)]
            current = art["per_asset"][ticker][f"lookback_{lookback}"]["by_cost"]["0.0"]
            fields = ("segments", "medians", "null_medians", "verdict")
            status = "MATCH" if all(current[f] == published[f] for f in fields) else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print(f"  {ticker} lb={lookback} -> {status}")

    print("\n=== Gate 3: classification reproducibility ===")
    for ticker, per_lb in art["per_asset"].items():
        for lookback in LOOKBACKS:
            rec = per_lb[f"lookback_{lookback}"]
            cls = classify(rec["by_cost"])
            be = break_even(rec["by_cost"])
            status = (
                "MATCH"
                if cls == rec["classification"] and be == rec["break_even_multiplier"]
                else "MISMATCH"
            )
            if status != "MATCH":
                all_ok = False
            print(
                f"  {ticker} lb={lookback}: recomputed {cls}/{be} vs "
                f"published {rec['classification']}/{rec['break_even_multiplier']} -> {status}"
            )

    print("\n=== Gate 4: cost-model diagnostic (not a tuning step) ===")
    for ticker in ("AAPL", "MSFT", "NVDA"):
        rec5 = art["per_asset"][ticker]["lookback_5"]["by_cost"]
        c0 = rec5["0.0"]
        c2 = rec5["2.0"]
        delta = round(c2["median_candidate"] - c0["median_candidate"], 3)
        print(
            f"  {ticker} lb=5: candidate median 0.0x={c0['median_candidate']:+.3f} "
            f"2.0x={c2['median_candidate']:+.3f} delta={delta:+.3f}"
        )
    for ticker in ("NVDA",):
        for lookback in LOOKBACKS:
            rec = art["per_asset"][ticker][f"lookback_{lookback}"]["by_cost"]
            for multiplier in ("0.5", "1.0", "2.0"):
                nulls = rec[multiplier]["null_medians"]
                unstable = any((n <= -1.0 + 1e-9) or (n >= 0.25) for n in nulls)
                print(
                    f"  {ticker} lb={lookback} cost={multiplier}: null_medians={nulls} "
                    f"bankruptcy_scale_or_large={unstable}"
                )
    print("  Interpretation: candidate medians are essentially invariant to the")
    print("  cost multiplier, and the existing engine's slippage is added to the")
    print("  fill price on sells as well as buys, so slippage is a gain on sells.")
    print("  The matched null can also reach bankruptcy-scale fold returns for")
    print("  low-priced/high-turnover cases. The cost-robustness classification is")
    print("  therefore conditional on the existing cost model, not a clean")
    print("  conservative stress.")

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact; the")
        print("classification reproduces from the artifact; and the zero-cost column")
        print("matches the prior momentum lookback sweep.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the")
        print("independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()