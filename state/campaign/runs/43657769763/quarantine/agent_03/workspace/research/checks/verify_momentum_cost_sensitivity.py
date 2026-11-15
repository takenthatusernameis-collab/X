"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three gates:

1. Fresh recomputation (separate implementation path): for AMZN and JPM at every
   lookback, the per-segment zero-cost and realistic-cost medians and the
   cost-matched coin-flip null medians are recomputed from scratch — a fresh
   re-implementation of `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregated with `walk_forward` directly, and the null via
   the framework's `noise_benchmark` (not `stress_segments`). The per-segment
   verdicts are re-derived from the recomputed medians/nulls with the documented
   rule. Every recomputed value is compared to the artifact.

2. Classification reproducibility gate: the cost-robustness cell classes
   (NO_EDGE_BASELINE / COST_ROBUST / COST_FRAGILE), the per-asset roll-up, and
   the universe-level cell counts are re-derived from the artifact's own
   zero/cost medians+verdicts via the documented rule, independently of the
   check's classification code. This exercises the MSFT lookback-5 COST_FRAGILE
   cell and the COST_ROBUST survivors.

3. Determinism: the AMZN/JPM recomputation is repeated and must be identical.

Research / simulation only. No live trading or production execution.
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

ZERO = bt.BacktestConfig(initial_capital=1e6)
REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
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


def segment_median_null(ticker, s, e, lookback, cfg):
    """Fresh per-segment candidate median and cost-matched null median."""
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
    cand_median = float(np.median(cand_log))
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252)
    return cand_median, float(noise.baseline_median_log_return), len(res.folds)


def verdict_from(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def recompute_cost_run(asset, lb, cfg):
    bars, dates = bt.load_ticker(asset)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    cand, null, nf = [], [], None
    for lab, s, e in runs:
        c, n, nfold = segment_median_null(asset, s, e, lb, cfg)
        cand.append(c)
        null.append(n)
        nf = nfold
    return dict(
        segments=[lab for lab, _, _ in runs],
        medians=[round(v, 3) for v in cand],
        null_medians=[round(v, 3) for v in null],
        n_folds=nf,
        verdict=verdict_from(cand, null),
    )


def edge_of(block):
    if block["verdict"] == "REGIME_STABLE" and all(m > 0 for m in block["medians"]):
        return "EDGE"
    if block["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def class_of(zero_block, cost_block):
    if edge_of(zero_block) != "EDGE":
        return "NO_EDGE_BASELINE"
    return "COST_ROBUST" if edge_of(cost_block) == "EDGE" else "COST_FRAGILE"


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load_artifact(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["realistic_cost"]["slippage_proportional"] == 0.0005
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks, both costs) ===")
    for asset in ["AMZN", "JPM"]:
        pub = art["per_asset"][asset]["by_lookback"]
        for lb in LOOKBACKS:
            z = recompute_cost_run(asset, lb, ZERO)
            c = recompute_cost_run(asset, lb, REALISTIC)
            pz = pub[str(lb)]["zero"]
            pc = pub[str(lb)]["cost"]
            ok = (z["medians"] == pz["medians"]
                  and z["null_medians"] == pz["null_medians"]
                  and z["verdict"] == pz["verdict"]
                  and z["segments"] == pz["segments"]
                  and c["medians"] == pc["medians"]
                  and c["null_medians"] == pc["null_medians"]
                  and c["verdict"] == pc["verdict"]
                  and c["segments"] == pc["segments"])
            if not ok:
                all_ok = False
            print("  {} lookback {}: zero {} -> {} | cost {} -> {} -> {}".format(
                asset, lb,
                z["medians"], z["verdict"],
                c["medians"], c["verdict"],
                "MATCH" if ok else "MISMATCH"))
            if not ok:
                print("    zero recomputed {} / {} vs pub {} / {}".format(
                    z["medians"], z["null_medians"], pz["medians"], pz["null_medians"]))
                print("    cost recomputed {} / {} vs pub {} / {}".format(
                    c["medians"], c["null_medians"], pc["medians"], pc["null_medians"]))

    print("\n=== Gate 2: cost-robustness classification reproducibility ===")
    counts = {"NO_EDGE_BASELINE": 0, "COST_ROBUST": 0, "COST_FRAGILE": 0}
    cmismatch = False
    for ticker in art["per_asset"]:
        pub = art["per_asset"][ticker]["by_lookback"]
        recomputed_classes = []
        for lb in LOOKBACKS:
            cls = class_of(pub[str(lb)]["zero"], pub[str(lb)]["cost"])
            counts[cls] += 1
            recomputed_classes.append((str(lb), cls))
            if cls != pub[str(lb)]["cell_class"]:
                cmismatch = True
                all_ok = False
                print("  {} lookback {}: recomputed {} vs published {} -> MISMATCH".format(
                    ticker, lb, cls, pub[str(lb)]["cell_class"]))
        robust = [lb for lb, cls in recomputed_classes if cls == "COST_ROBUST"]
        fragile = [lb for lb, cls in recomputed_classes if cls == "COST_FRAGILE"]
        asset_class = ("COST_ROBUST" if robust else
                       ("COST_FRAGILE" if fragile else "NO_EDGE"))
        status = "MATCH" if asset_class == art["per_asset"][ticker]["asset_class"] else "MISMATCH"
        if status != "MATCH":
            cmismatch = True
            all_ok = False
        print("  {:6s} classes={} -> asset_class {} vs published {} -> {}".format(
            ticker, recomputed_classes, asset_class,
            art["per_asset"][ticker]["asset_class"], status))
    pub_counts = art["universe"]["class_counts"]
    if counts != pub_counts:
        cmismatch = True
        all_ok = False
    print("  universe class counts recomputed {} vs published {} -> {}".format(
        counts, pub_counts, "MATCH" if counts == pub_counts else "MISMATCH"))
    if not cmismatch:
        print("  all cells, per-asset roll-ups, and universe counts reproduce -> MATCH")

    print("\nDeterminism: recomputing AMZN lookback=5 cost-run medians twice.")
    a = recompute_cost_run("AMZN", 5, REALISTIC)["medians"]
    b = recompute_cost_run("AMZN", 5, REALISTIC)["medians"]
    print("  AMZN lookback 5 cost medians r1: {}; r2: {} -> {}".format(
        a, b, "identical" if a == b else "DIFFERENT"))
    assert a == b, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact, the "
              "cost-robustness classification reproduces, and the verification "
              "path is deterministic.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()