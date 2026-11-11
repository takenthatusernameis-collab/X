"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

This is a separate implementation path. It recomputes the candidate and the
matched null from a fresh ``walk_forward`` (not ``stress_segments``), using a
re-implemented ``volatility_blocks`` / segment reconstruction and a fresh
momentum-signal implementation, then compares against the artifact
``state/check_artifacts/momentum_cost_sensitivity_results.json``.

Gates:

1. Fresh recomputation of AMZN/JPM candidate and null medians for all
   lookbacks and all three cost levels, per segment, compared to the artifact.
2. Fresh recomputation of the full 10-ticker universe at lookback 5 under the
   realistic cost model (candidate and null medians), compared to the artifact.
3. Zero-cost cross-consistency: the artifact's ``zero`` columns must reproduce
   ``momentum_lookback_sweep_results.json`` exactly (medians, null medians,
   verdict) for every ticker and lookback. This ties the new artifact to the
   prior independently-verified momentum sweep.
4. Classification reproducibility: the predeclared cost classes are re-derived
   from the artifact medians/nulls and compared to the stored summary.
5. Cost-model control: the constant-long control is recomputed and must show
   the same return increase from zero to realistic cost, which is the evidence
   that the engine's symmetric-slippage convention is not a conservative cost.
6. Determinism of the verification path.

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
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

COST_LEVELS = {
    "zero": {},
    "realistic": {
        "commission_per_trade": 2.0,
        "commission_per_share": 0.003,
        "slippage_cents": 2.0,
        "slippage_proportional": 0.0005,
    },
    "conservative": {
        "commission_per_trade": 4.0,
        "commission_per_share": 0.006,
        "slippage_cents": 4.0,
        "slippage_proportional": 0.001,
    },
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


def _fold_median(res):
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log))


def candidate_segment_median(ticker, s, e, lookback, cost_kwargs):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = momentum_signals(closes, lookback)[s:e]
    cfg = bt.BacktestConfig(**cost_kwargs)
    res = bt.walk_forward(
        list(bars)[s:e], sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    return _fold_median(res)


def null_segment_median(seg_bars, lookback, cost_kwargs):
    """Fresh null path: coin-flip signal + walk_forward directly (no stress_segments)."""
    n = len(seg_bars)
    seed = SEED + int(lookback) * 1000
    sig = bt.random_signals(n, seed=seed)
    cfg = bt.BacktestConfig(**cost_kwargs)
    res = bt.walk_forward(
        list(seg_bars), sig, train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg)
    return _fold_median(res)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def stats(medians, nulls):
    cand = [float(m) for m in medians]
    null = [float(m) for m in nulls]
    gaps = [c - n for c, n in zip(cand, null)]
    return dict(
        mean_median=round(float(np.mean(cand)), 3),
        mean_gap=round(float(np.mean(gaps)), 3),
        pos_segments=int(sum(1 for m in cand if m > 0)),
        n_segments=len(cand),
    )


def classify(z, r, c):
    if z["mean_median"] <= 0:
        return "NO_ZERO_COST_EDGE"
    if r["mean_median"] > 0 and c["mean_median"] > 0:
        return "COST_ROBUST"
    if r["mean_median"] > 0:
        return "COST_DEGRADED"
    return "COST_ERASED"


def main():
    np.random.seed(42)
    artifact = load_json(ARTIFACT_PATH)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)
    assert artifact["cost_model_confound"] is True, "artifact does not flag the cost-model confound"

    all_ok = True

    print("=== Gate 1: fresh walk_forward + fresh null (AMZN, JPM; all lookbacks/costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        seg_bars = list(bars)
        for lb in LOOKBACKS:
            for level in COST_LEVELS:
                cand = []
                null = []
                for lab, s, e in runs:
                    cand.append(round(candidate_segment_median(asset, s, e, lb, COST_LEVELS[level]), 3))
                    null.append(round(null_segment_median(seg_bars[s:e], lb, COST_LEVELS[level]), 3))
                pub = artifact["per_cost"][level][str(lb)][asset]
                ok = (cand == pub["medians"] and null == pub["null_medians"]
                      and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                    print("  MISMATCH {} lb={} {}: cand {} vs {} | null {} vs {}".format(
                        asset, lb, level, cand, pub["medians"], null, pub["null_medians"]))
            print("  {}: lookbacks {} all cost levels recomputed".format(asset, list(LOOKBACKS)))
    print("  Gate 1 -> {}".format("MATCH" if all_ok else "MISMATCH"))

    print("\n=== Gate 2: full universe, lookback 5, realistic cost ===")
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    g2_ok = True
    for entry in manifest["entries"]:
        t = entry["ticker"]
        bars, _ = bt.load_ticker(t)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        cand = [round(candidate_segment_median(t, s, e, 5, COST_LEVELS["realistic"]), 3)
                for lab, s, e in runs]
        null = [round(null_segment_median(list(bars)[s:e], 5, COST_LEVELS["realistic"]), 3)
                for lab, s, e in runs]
        pub = artifact["per_cost"]["realistic"]["5"][t]
        ok = cand == pub["medians"] and null == pub["null_medians"]
        if not ok:
            g2_ok = False
            all_ok = False
            print("  MISMATCH {}: cand {} vs {} | null {} vs {}".format(
                t, cand, pub["medians"], null, pub["null_medians"]))
    print("  Gate 2 -> {}".format("MATCH" if g2_ok else "MISMATCH"))

    print("\n=== Gate 3: zero-cost columns cross-check vs momentum_lookback_sweep_results.json ===")
    sweep = load_json(SWEEP_ARTIFACT_PATH)
    g3_ok = True
    for ticker in sweep["universe"]["per_asset"]:
        for lb in LOOKBACKS:
            pub_new = artifact["per_cost"]["zero"][str(lb)][ticker]
            pub_old = sweep["per_asset"][ticker][str(lb)]
            ok = (pub_new["medians"] == pub_old["medians"]
                  and pub_new["null_medians"] == pub_old["null_medians"]
                  and pub_new["verdict"] == pub_old["verdict"])
            if not ok:
                g3_ok = False
                all_ok = False
                print("  MISMATCH {} lb={}: new {} vs old {}".format(
                    ticker, lb, pub_new, pub_old))
    print("  Gate 3 -> {}".format("MATCH" if g3_ok else "MISMATCH"))

    print("\n=== Gate 4: classification reproducibility ===")
    g4_ok = True
    for lb in LOOKBACKS:
        key = str(lb)
        for ticker in artifact["summary"][key]:
            z = stats(artifact["per_cost"]["zero"][key][ticker]["medians"],
                      artifact["per_cost"]["zero"][key][ticker]["null_medians"])
            r = stats(artifact["per_cost"]["realistic"][key][ticker]["medians"],
                      artifact["per_cost"]["realistic"][key][ticker]["null_medians"])
            c = stats(artifact["per_cost"]["conservative"][key][ticker]["medians"],
                      artifact["per_cost"]["conservative"][key][ticker]["null_medians"])
            cls = classify(z, r, c)
            pub = artifact["summary"][key][ticker]
            if cls != pub["cost_class"] or z != pub["zero"] or r != pub["realistic"] or c != pub["conservative"]:
                g4_ok = False
                all_ok = False
                print("  MISMATCH {} lb={}: recomputed {} vs stored {}".format(
                    ticker, lb, cls, pub["cost_class"]))
    print("  Gate 4 -> {}".format("MATCH" if g4_ok else "MISMATCH"))

    print("\n=== Gate 5: cost-model control (constant-long return rises with cost) ===")
    g5_ok = True
    for t in ["AAPL", "NVDA"]:
        bars, _ = bt.load_ticker(t)
        n = bars.n_bars
        const_long = [bt.Signal(date=i + 1, weight=1.0) for i in range(n)]
        rets = {}
        for level in ["zero", "realistic", "conservative"]:
            res = bt.run_bars(list(bars), const_long, bt.BacktestConfig(**COST_LEVELS[level]))
            rets[level] = float(res.equity_curve[-1] / res.equity_curve[0] - 1.0)
        pub = artifact["cost_model_control"]["constant_long"]
        rises = rets["realistic"] > rets["zero"]
        pub_rises = (pub["realistic"][t]["const_long_total_return"]
                     > pub["zero"][t]["const_long_total_return"])
        ok = (round(rets["zero"], 6) == pub["zero"][t]["const_long_total_return"]
              and round(rets["realistic"], 6) == pub["realistic"][t]["const_long_total_return"]
              and rises == pub_rises)
        if not ok:
            g5_ok = False
            all_ok = False
        print("  {}: zero={:+.4f} real={:+.4f} cons={:+.4f} rises={} (pub rises={}) -> {}".format(
            t, rets["zero"], rets["realistic"], rets["conservative"], rises, pub_rises,
            "MATCH" if ok else "MISMATCH"))
    print("  Gate 5 -> {}".format("MATCH" if g5_ok else "MISMATCH"))

    print("\nDeterminism: recomputing AMZN lookback 5 realistic medians twice.")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(candidate_segment_median("AMZN", s, e, 5, COST_LEVELS["realistic"]), 3)
          for lab, s, e in runs]
    m2 = [round(candidate_segment_median("AMZN", s, e, 5, COST_LEVELS["realistic"]), 3)
          for lab, s, e in runs]
    print("  AMZN r1: {}; r2: {} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact: candidate "
              "and null medians (fresh walk_forward), zero-cost cross-consistency vs "
              "momentum_lookback_sweep_results.json, classification, and the "
              "cost-model control.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()