"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

The check writes `state/check_artifacts/momentum_cost_sensitivity_results.json`.
This verifier recomputes the whole grid through a separate implementation path
and compares against the artifact:

Gate 1 — fresh recomputation of every asset / lookback / cost cell using fresh
components (`vol_blocks`, segment reconstruction, momentum signals,
`walk_forward` directly, and `noise_benchmark` directly). No call to
`stress_segments` or `stress_segments_across_tickers`, so the candidate path and
the null path are independent of the check's aggregation code.

Gate 2 — zero-cost column cross-consistency: the zero-cost cell must reproduce
`momentum_lookback_sweep_results.json` (medians, nulls, verdicts) at 3 decimals.

Gate 3 — classification reproducibility: the per-lookback and overall
cost-robustness classifications are re-derived from the artifact's own metrics
using the documented rule, independently of the check's classification code.

Gate 4 — determinism: a repeated fresh recomputation of AMZN lookback 5 at the
realistic cost is identical.

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
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
OUTLIER_TOL = 0.05
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LOOKBACK_SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_cfgs():
    return {
        "zero": bt.BacktestConfig(initial_capital=1e6),
        "realistic": bt.BacktestConfig(
            initial_capital=1e6,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
        "conservative": bt.BacktestConfig(
            initial_capital=1e6,
            commission_per_trade=4.0,
            commission_per_share=0.006,
            slippage_cents=4.0,
            slippage_proportional=0.0010,
        ),
    }


def vol_blocks(closes, n_blocks, window):
    """Independent re-implementation of research.backtest.volatility_blocks."""
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
            bmed = (
                float(np.median([v for v in bar_vols[s:e] if v > 0]))
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


def candidate_median(seg_bars, seg_signals, cfg):
    wf = bt.walk_forward(
        seg_bars,
        seg_signals,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
    )
    logs = np.array(
        [
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            for f in wf.folds
        ],
        dtype=np.float64,
    )
    return float(np.median(logs)), len(wf.folds)


def null_median(seg_bars, lookback, cfg):
    nb = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=0,
        overlap_window=0,
        cfg=cfg,
        periods_per_year=252,
    )
    return float(nb.baseline_median_log_return)


def verdict(medians, nulls):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(nulls)))
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def cell_metrics(per_asset):
    n = len(per_asset)
    edge = []
    beat = []
    cand_means = {}
    null_means = {}
    for t, r in per_asset.items():
        cm = float(np.mean(r["medians"]))
        nm = float(np.mean(r["null_medians"]))
        cand_means[t] = cm
        null_means[t] = nm
        if r["verdict"] == "REGIME_STABLE" and all(m > 0 for m in r["medians"]):
            edge.append(t)
        if cm > nm:
            beat.append(t)
    return {
        "n_assets": n,
        "edge_asset_count": len(edge),
        "beat_null_asset_count": len(beat),
        "mean_candidate_median": float(np.mean(list(cand_means.values()))),
        "mean_null_median": float(np.mean(list(null_means.values()))),
    }


def classify(m):
    n = m["zero"]["n_assets"]
    er = m["realistic"]["edge_asset_count"]
    ec = m["conservative"]["edge_asset_count"]
    br = m["realistic"]["beat_null_asset_count"]
    bc = m["conservative"]["beat_null_asset_count"]
    if er == 0 or br <= n / 2:
        return "COST_ELIMINATED"
    if er >= 1 and ec >= 1 and bc > n / 2:
        return "COST_ROBUST"
    return "COST_FRAGILE"


def load_artifact():
    with open(ARTIFACT) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load_artifact()
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    cfgs = cost_cfgs()
    all_ok = True

    tickers = {}
    for entry in load_manifest()["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(tickers.keys())

    print("=== Gate 1: fresh walk_forward + noise_benchmark for all assets/lookbacks/costs ===")
    fresh = {}
    for lb in LOOKBACKS:
        fresh[str(lb)] = {}
        for cname, cfg in cfgs.items():
            per_asset = {}
            for ticker in ticker_order:
                bars = tickers[ticker]
                closes = bars.closes_array()
                labels = vol_blocks(closes, N_BLOCKS, WINDOW)
                runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
                full_sig = momentum_signals(closes, lb)
                meds = []
                nulls = []
                seg_names = []
                for lab, s, e in runs:
                    seg_bars = list(bars)[s:e]
                    seg_sig = full_sig[s:e]
                    m, _ = candidate_median(seg_bars, seg_sig, cfg)
                    nm = null_median(seg_bars, lb, cfg)
                    meds.append(round(m, 6))
                    nulls.append(round(nm, 6))
                    seg_names.append(lab)
                per_asset[ticker] = {
                    "segments": seg_names,
                    "medians": meds,
                    "null_medians": nulls,
                    "verdict": verdict(meds, nulls),
                }
            fresh[str(lb)][cname] = per_asset

            published = art["cells"][str(lb)]["per_cost_level"][cname]["per_asset"]
            mismatch = False
            for ticker in ticker_order:
                f = per_asset[ticker]
                p = published[ticker]
                ok = (
                    f["medians"] == p["medians"]
                    and f["null_medians"] == p["null_medians"]
                    and f["verdict"] == p["verdict"]
                    and f["segments"] == p["segments"]
                )
                if not ok:
                    mismatch = True
                    all_ok = False
                    print(
                        f"  MISMATCH lb={lb} cost={cname} {ticker}: "
                        f"fresh med={f['medians']} null={f['null_medians']} "
                        f"verdict={f['verdict']} vs published med={p['medians']} "
                        f"null={p['null_medians']} verdict={p['verdict']}"
                    )
            print(f"  lb={lb} cost={cname}: {'MATCH' if not mismatch else 'MISMATCH'} (10 assets)")

    print("\n=== Gate 2: zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    with open(LOOKBACK_SWEEP_ARTIFACT) as f:
        prev = json.load(f)
    prev_lp = prev["per_asset"]
    cmismatch = False
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            zero = fresh[str(lb)]["zero"][ticker]
            pub = prev_lp[ticker][str(lb)]
            ok = (
                [round(v, 3) for v in zero["medians"]] == pub["medians"]
                and [round(v, 3) for v in zero["null_medians"]] == pub["null_medians"]
                and zero["verdict"] == pub["verdict"]
            )
            if not ok:
                cmismatch = True
                all_ok = False
                print(f"  MISMATCH {ticker} lb={lb}: {zero} vs {pub}")
    print("  zero-cost cross-check: " + ("MISMATCH" if cmismatch else "MATCH (10 tickers x 3 lookbacks)"))

    print("\n=== Gate 3: classification reproducibility from artifact metrics ===")
    labels = {}
    for lb in LOOKBACKS:
        m = {
            c: cell_metrics(art["cells"][str(lb)]["per_cost_level"][c]["per_asset"])
            for c in cfgs
        }
        re_class = classify(m)
        pub_class = art["cells"][str(lb)]["classification"]
        status = "MATCH" if re_class == pub_class else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        labels[str(lb)] = re_class
        print(
            f"  lb={lb}: recomputed={re_class} published={pub_class} -> {status} "
            f"(edge {m['zero']['edge_asset_count']}/"
            f"{m['realistic']['edge_asset_count']}/"
            f"{m['conservative']['edge_asset_count']}, "
            f"beat_null {m['zero']['beat_null_asset_count']}/"
            f"{m['realistic']['beat_null_asset_count']}/"
            f"{m['conservative']['beat_null_asset_count']})"
        )
    vals = list(labels.values())
    if all(v == "COST_ROBUST" for v in vals):
        overall = "COST_ROBUST"
    elif vals.count("COST_ELIMINATED") >= 2:
        overall = "COST_ELIMINATED"
    else:
        overall = "COST_FRAGILE"
    pub_overall = art["overall"]["classification"]
    status = "MATCH" if overall == pub_overall else "MISMATCH"
    if status != "MATCH":
        all_ok = False
    print(f"  overall: recomputed={overall} published={pub_overall} -> {status}")

    print("\n=== Gate 4: determinism (AMZN lookback 5 realistic, r1 == r2) ===")
    bars = tickers["AMZN"]
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    full_sig = momentum_signals(closes, 5)

    def one_pass():
        meds = []
        nulls = []
        for lab, s, e in runs:
            seg_bars = list(bars)[s:e]
            m, _ = candidate_median(seg_bars, full_sig[s:e], cfgs["realistic"])
            nm = null_median(seg_bars, 5, cfgs["realistic"])
            meds.append(round(m, 6))
            nulls.append(round(nm, 6))
        return meds, nulls

    m1, n1 = one_pass()
    m2, n2 = one_pass()
    det = (m1 == m2) and (n1 == n2)
    print(f"  r1 == r2: {det} (medians {m1} vs {m2})")
    if not det:
        all_ok = False

    print()
    if all_ok:
        print(
            "All independent recomputations MATCH the check artifact, the "
            "zero-cost column cross-checks against momentum_lookback_sweep_results.json, "
            "and the cost-robustness classification reproduces."
        )
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        sys.exit(1)
    print(
        "NOTE: research/simulation only. No live trading or production "
        "execution. Momentum cost-sensitivity independent verification: "
        "exploratory simulation."
    )


if __name__ == "__main__":
    main()