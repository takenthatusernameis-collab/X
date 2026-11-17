"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Verification gates (each recomputes from the artifact's own declared inputs and
compares against the artifact written by the check):

1. Fresh recomputation of AMZN/JPM medians and null medians for every lookback
   x cost level via a separate implementation path: re-implemented
   ``volatility_blocks`` / segment reconstruction / momentum signal, aggregating
   fold log returns with ``walk_forward`` directly (NOT ``stress_segments``), and
   the framework's ``noise_benchmark`` for the matched null (a separate code path
   from ``stress_segments``). The cost config is rebuilt from the artifact's
   ``cost_levels`` declaration, so the gate also checks that the artifact's
   declared cost grid is the one actually applied.

2. Classification reproducibility: ``edge_status``, per-cell
   ``cost_robustness``, per-level edge/verdict counts, and the headline
   classification are re-derived from the artifact's medians/verdicts using the
   documented rules, independently of the check's own computation.

3. Zero-cost cross-consistency: the ``zero`` cost column is an independent rerun
   of the prior momentum lookback sweep, so its medians / null medians / verdicts
   must match `momentum_lookback_sweep_results.json` for all 10 tickers and all
   lookbacks. A mismatch would flag a change in the zero-cost computation.

4. Determinism: the AMZN lookback=5 realistic medians are recomputed twice.

5. Turnover/degeneracy reproducibility: the signal weight counts and sign flips
   are recomputed from a fresh momentum-signal implementation and compared to the
   artifact's ``turnover`` block, so the degeneracy finding is independently
   reproduced rather than trusted.

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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def load(path):
    with open(path) as f:
        return json.load(f)


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
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        runs.append((cur, start, len(labels)))
    return runs


def fresh_momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal's sign statistic.

    Mirrors the specified definition (mean log price over the lookback window is
    the sign statistic) so this gate verifies the artifact's turnover; it does
    not silently substitute a different signal definition.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        stat = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if stat > 0 else -1.0)
    return out


def cfg_from_decl(decl):
    return bt.BacktestConfig(**decl)


def segment_result(ticker, s, e, lookback, cfg):
    """Fresh per-segment candidate median and matched null median.

    Candidate: walk_forward on the segment slice with the artifact's cost config.
    Null: framework noise_benchmark on the same segment with the same config and
    the framework's null window convention (warmup=0, overlap=0).
    """
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = fresh_momentum_signals(closes, lookback)[s:e]
    res = bt.walk_forward(
        seg_bars, sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, cfg=cfg)
    return float(np.median(cand)), float(noise.baseline_median_log_return), len(res.folds)


def edge_status(medians, verdict):
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def classify_cell(e0, er, ec):
    if not e0:
        return "NO_EDGE_AT_ZERO"
    if er and ec:
        return "SURVIVES"
    if er:
        return "SURVIVES_REALISTIC"
    return "COST_FRAGILE"


def main():
    np.random.seed(42)
    art = load(ARTIFACT_PATH)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    cost_order = art["cost_order"]
    cost_levels = art["cost_levels"]
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for level in cost_order:
                cfg = cfg_from_decl(cost_levels[level])
                meds, nulls, folds = [], [], []
                for lab, s, e in runs:
                    m, n, nf = segment_result(ticker, s, e, lb, cfg)
                    meds.append(round(m, 3))
                    nulls.append(round(n, 3))
                    folds.append(nf)
                pub = art["cells"][ticker][str(lb)][level]
                pub_m = [round(v, 3) for v in pub["medians"]]
                pub_n = [round(v, 3) for v in pub["null_medians"]]
                # The artifact stores the first segment's candidate fold count
                # (RegimeStressResult.n_folds = first scenario's n_folds).
                first_folds = folds[0] if folds else None
                ok = (meds == pub_m and nulls == pub_n
                      and lbls == pub["segments"] and first_folds == pub["n_folds"])
                status = "MATCH" if ok else "MISMATCH"
                if not ok:
                    all_ok = False
                print("  {:5s} lb={:2d} {:16s} -> {} (folds {})".format(
                    ticker, lb, level, status, folds))
                if status == "MISMATCH":
                    print("    medians recomputed {} vs published {}".format(meds, pub_m))
                    print("    nulls   recomputed {} vs published {}".format(nulls, pub_n))
                    print("    labels  recomputed {} vs published {}".format(lbls, pub["segments"]))

    print("\n=== Gate 2: classification reproducibility ===")
    gate2_ok = True
    edge_counts = {level: {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0} for level in cost_order}
    class_counts = {"SURVIVES": 0, "SURVIVES_REALISTIC": 0,
                    "COST_FRAGILE": 0, "NO_EDGE_AT_ZERO": 0}
    for ticker in art["cells"]:
        for lb in LOOKBACKS:
            per = art["cells"][ticker][str(lb)]
            for level in cost_order:
                es = edge_status(per[level]["medians"], per[level]["verdict"])
                edge_counts[level][es] += 1
                if es != per[level]["edge_status"]:
                    gate2_ok = False
                    print("  MISMATCH edge_status {} lb{} {}: {} vs {}".format(
                        ticker, lb, level, es, per[level]["edge_status"]))
            e0 = per["zero"]["edge_status"] == "EDGE"
            er = per["realistic"]["edge_status"] == "EDGE"
            ec = per["conservative_2x"]["edge_status"] == "EDGE"
            cls = classify_cell(e0, er, ec)
            class_counts[cls] += 1
            if cls != art["robustness"][ticker][str(lb)]["cost_robustness"]:
                gate2_ok = False
                print("  MISMATCH cost_robustness {} lb{}: {} vs {}".format(
                    ticker, lb, cls, art["robustness"][ticker][str(lb)]["cost_robustness"]))
    for level in cost_order:
        pub = art["summary"]["edge_counts"][level]
        if edge_counts[level] != pub:
            gate2_ok = False
            print("  MISMATCH edge_counts {}: {} vs {}".format(level, edge_counts[level], pub))
    if class_counts != art["summary"]["cost_robustness_counts"]:
        gate2_ok = False
        print("  MISMATCH class counts: {} vs {}".format(
            class_counts, art["summary"]["cost_robustness_counts"]))
    zero_edge = edge_counts["zero"]["EDGE"]
    survived = sum(
        1 for t in art["cells"] for lb in LOOKBACKS
        if art["robustness"][t][str(lb)]["edge_zero"]
        and art["robustness"][t][str(lb)]["edge_realistic"])
    if zero_edge == 0:
        headline = "NO_EDGE_AT_ZERO"
    else:
        headline = "COST_ROBUST" if survived / zero_edge >= 0.5 else "COST_FRAGILE"
    if headline != art["summary"]["headline"]:
        gate2_ok = False
        print("  MISMATCH headline: {} vs {}".format(headline, art["summary"]["headline"]))
    if not gate2_ok:
        all_ok = False
    print("  edge_counts={} class_counts={} headline={} -> {}".format(
        edge_counts, class_counts, headline, "MATCH" if gate2_ok else "MISMATCH"))

    print("\n=== Gate 3: zero-cost column cross-consistency vs momentum_lookback_sweep ===")
    sweep = load(SWEEP_ARTIFACT)
    cmismatch = False
    for ticker in sweep["universe"]["per_asset"]:
        for lb in LOOKBACKS:
            pub = sweep["universe"]["per_asset"][ticker]
            zm = art["cells"][ticker][str(lb)]["zero"]["medians"]
            zn = art["cells"][ticker][str(lb)]["zero"]["null_medians"]
            zv = art["cells"][ticker][str(lb)]["zero"]["verdict"]
            sm = pub["lookback_medians"][str(lb)]
            sn = pub["lookback_null_medians"][str(lb)]
            sv = pub["verdicts_5"] if lb == 5 else None
            ok = (zm == sm and zn == sn)
            if lb == 5:
                ok = ok and zv == sv
            if not ok:
                cmismatch = True
                all_ok = False
            print("  {:6s} lb={:2d} -> {} (medians {}, nulls {})".format(
                ticker, lb, "MATCH" if ok else "MISMATCH", zm, zn))
    if not cmismatch:
        print("  all zero-cost cells reproduce the prior lookback sweep -> MATCH")

    print("\n=== Gate 4: determinism (AMZN lb=5 realistic, twice) ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    cfg = cfg_from_decl(cost_levels["realistic"])
    m1 = [round(segment_result("AMZN", s, e, 5, cfg)[0], 3) for _, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, 5, cfg)[0], 3) for _, s, e in runs]
    det_ok = m1 == m2
    print("  r1={} r2={} -> {}".format(m1, m2, "identical" if det_ok else "DIFFERENT"))
    if not det_ok:
        all_ok = False

    print("\n=== Gate 5: turnover / degeneracy reproducibility ===")
    tmic = False
    for ticker in art["turnover"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            w = np.array([s.weight for s in fresh_momentum_signals(closes, lb)], dtype=np.float64)
            flips = int(np.sum(np.diff(w) != 0.0))
            n_long = int(np.sum(w > 0))
            pub = art["turnover"][ticker][str(lb)]
            ok = (flips == pub["n_sign_flips"] and n_long == pub["n_long"]
                  and len(w) == pub["n_bars"])
            if not ok:
                tmic = True
                all_ok = False
            if ticker in ("AMZN", "NVDA"):
                print("  {:5s} lb={:2d} flips={} n_long={} -> {}".format(
                    ticker, lb, flips, n_long, "MATCH" if ok else "MISMATCH"))
    if not tmic:
        print("  all turnover entries reproduce -> MATCH")

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity artifact: "
              "per-cell medians/nulls, classification, zero-cost cross-consistency, "
              "determinism, and the turnover/degeneracy diagnostic.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()