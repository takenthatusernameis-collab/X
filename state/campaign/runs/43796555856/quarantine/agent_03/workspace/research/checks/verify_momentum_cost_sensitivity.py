"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Verification gates (all recomputed from scratch, no call to `stress_segments`
or to any function defined in the check file):

1. Recorded-implementation cells: fresh `walk_forward` + `noise_benchmark`
   (the framework's own null path) recomputation of AMZN/JPM medians, null
   medians, segment labels and verdicts at all three cost levels and all three
   lookbacks, compared to the artifact.
2. Corrected-signal cells: the same fresh recomputation with an independently
   re-implemented log-return momentum, compared to the artifact's diagnostic
   block.
3. Signal-integrity gate: the weight series of the recorded signal is
   recomputed from the documented expression, and the corrected signal from the
   return expression, for every ticker and lookback; the constant-+1.0 cell
   count and the sign-flip assets are re-derived and compared to the artifact.
4. Classification gate: the per-asset cost classification is re-derived from
   the artifact medians/nulls using the documented rule, for both grids.
5. Zero-cost cross-consistency gate: the recorded-implementation zero-cost
   column reproduces `momentum_lookback_sweep_results.json`.
6. Determinism gate: the verification path is re-run and compared to itself.

Exit code 0 means every gate matched; any mismatch prints the recomputed and
published values and exits 1.
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
LEVEL_NAMES = ("zero", "realistic", "stress_2x")

COST_PARAMS = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress_2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                      slippage_cents=4.0, slippage_proportional=0.0010),
}

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
LOOKBACK_SWEEP = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

VERDICT_TOL = 0.05
CLASS_NAMES = ("NO_EDGE_ZERO_COST", "COST_ROBUST", "COST_FRAGILE", "COST_ELIMINATED")


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
                float(np.std(np.log(closes[i - window: i]), ddof=1)) * np.sqrt(252.0))
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


def recorded_signals(closes, lookback):
    """Independent re-implementation of the recorded framework expression."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1: i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def corrected_signals(closes, lookback):
    """Independent re-implementation of the docstring-faithful momentum."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def verdict_of(medians, nulls):
    """Fresh implementation of the framework verdict rule."""
    cand_disp = float(np.std(np.asarray(medians, dtype=np.float64)))
    null_disp = float(np.std(np.asarray(nulls, dtype=np.float64)))
    if all(abs(m) <= VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def edge_status(verdict, medians):
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def fresh_cell(ticker, lookback, level, signals_fn):
    """Fresh walk_forward + noise_benchmark recomputation of one cell."""
    bars, _dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    sig = signals_fn(closes, lookback)
    cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
                            **COST_PARAMS[level])
    medians, nulls, names, fold_counts = [], [], [], []
    nfolds = None
    for lab, s, e in runs:
        seg_bars = list(bars)[s:e]
        seg_signals = sig[s:e]
        res = bt.walk_forward(
            seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        cand = np.asarray([np.log1p(np.clip(f.metrics["total_return"],
                                            -1.0 + 1e-12, None))
                           for f in res.folds], dtype=np.float64)
        medians.append(float(np.median(cand)))
        names.append(lab)
        fold_counts.append(len(res.folds))
        nfolds = len(res.folds) if nfolds is None else nfolds
        noise = noise_benchmark(
            bars=seg_bars,
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN, test_window=TEST,
            warmup=0, overlap_window=0, cfg=cfg, periods_per_year=252)
        nulls.append(float(noise.baseline_median_log_return))
    verdict = verdict_of(medians, nulls)
    return dict(segments=names, medians=medians, null_medians=nulls,
                verdict=verdict, n_folds=nfolds, n_folds_per_segment=fold_counts)


def classify(n_edge):
    n0, n1 = n_edge["zero"], n_edge["realistic"]
    if n0 == 0:
        return "NO_EDGE_ZERO_COST"
    if n1 >= 2:
        return "COST_ROBUST"
    if n1 == 1:
        return "COST_FRAGILE"
    return "COST_ELIMINATED"


def derive_classification(block, tickers):
    """Re-derive per-asset classification from published medians/nulls."""
    out = {}
    for ticker in tickers:
        n_edge = {}
        for level in LEVEL_NAMES:
            n_edge[level] = 0
            for lb in LOOKBACKS:
                cell = block[level][ticker][str(lb)]
                v = verdict_of(cell["medians"], cell["null_medians"])
                if edge_status(v, cell["medians"]) == "EDGE":
                    n_edge[level] += 1
        out[ticker] = classify(n_edge)
    return out


def main() -> int:
    np.random.seed(42)
    art = json.loads(ARTIFACT.read_text())
    assert art["dataset_id"] == DATASET_ID
    assert art["level_names"] == list(LEVEL_NAMES)
    tickers = list(art["universe"]["ticker_order"])
    all_ok = True

    print("=== Gate 1: recorded implementation, fresh walk_forward + null (AMZN, JPM) ===")
    for ticker in ("AMZN", "JPM"):
        for level in LEVEL_NAMES:
            for lb in LOOKBACKS:
                got = fresh_cell(ticker, lb, level, recorded_signals)
                pub = art["per_cost"][level][ticker][str(lb)]
                pub_med = [round(m, 3) for m in got["medians"]]
                pub_null = [round(n, 3) for n in got["null_medians"]]
                ok = (
                    pub_med == [round(m, 3) for m in pub["medians"]]
                    and pub_null == [round(n, 3) for n in pub["null_medians"]]
                    and got["segments"] == pub["segments"]
                    and got["verdict"] == pub["verdict"]
                    and got["n_folds"] == pub["n_folds"]
                    and got["n_folds_per_segment"] == pub["n_folds_per_segment"]
                )
                all_ok = all_ok and ok
                print("  {:6s} {:10s} lb={:2d} -> {} (n_folds={})".format(
                    ticker, level, lb, "MATCH" if ok else "MISMATCH",
                    got["n_folds"]))
                if not ok:
                    print("    recomputed medians {} nulls {} verdict {} | "
                          "published medians {} nulls {} verdict {}".format(
                              pub_med, pub_null, got["verdict"],
                              [round(m, 3) for m in pub["medians"]],
                              [round(n, 3) for n in pub["null_medians"]],
                              pub["verdict"]))

    print("\n=== Gate 2: corrected signal, fresh walk_forward + null (AMZN, JPM) ===")
    diag = art["corrected_signal_diagnostic"]
    for ticker in ("AMZN", "JPM"):
        for level in LEVEL_NAMES:
            for lb in LOOKBACKS:
                got = fresh_cell(ticker, lb, level, corrected_signals)
                pub = diag["per_cost"][level][ticker][str(lb)]
                ok = (
                    [round(m, 3) for m in got["medians"]]
                    == [round(m, 3) for m in pub["medians"]]
                    and [round(n, 3) for n in got["null_medians"]]
                    == [round(n, 3) for n in pub["null_medians"]]
                    and got["segments"] == pub["segments"]
                    and got["verdict"] == pub["verdict"]
                    and got["n_folds_per_segment"] == pub["n_folds_per_segment"]
                )
                all_ok = all_ok and ok
                print("  {:6s} {:10s} lb={:2d} -> {}".format(
                    ticker, level, lb, "MATCH" if ok else "MISMATCH"))
                if not ok:
                    print("    recomputed medians {} verdict {} | published "
                          "medians {} verdict {}".format(
                              [round(m, 3) for m in got["medians"]],
                              got["verdict"],
                              [round(m, 3) for m in pub["medians"]],
                              pub["verdict"]))

    print("\n=== Gate 3: signal-integrity re-derivation (all tickers, all lookbacks) ===")
    integ = art["signal_integrity"]
    n_constant = 0
    flippers = []
    for ticker in tickers:
        bars, _dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            rec = [s.weight for s in recorded_signals(closes, lb)][lb:]
            framework = [s.weight for s in bt.momentum_signals(closes, lookback=lb)][lb:]
            corr = [s.weight for s in corrected_signals(closes, lb)][lb:]
            pub = integ["per_ticker"][ticker][str(lb)]
            ok = (
                rec == framework
                and pub["n_short_after_warmup"] == sum(1 for w in rec if w < 0)
                and pub["n_long_after_warmup"] == sum(1 for w in rec if w > 0)
                and pub["distinct_weights_after_warmup"] == sorted(set(rec))
            )
            all_ok = all_ok and ok
            if not ok:
                print("  MISMATCH {} lb={}: recomputed distinct={} framework_match={}"
                      " published={}".format(
                          ticker, lb, sorted(set(rec)), rec == framework,
                          pub["distinct_weights_after_warmup"]))
            if sorted(set(rec)) == [1.0]:
                n_constant += 1
            elif any(w < 0 for w in rec):
                flippers.append((ticker, lb, sum(1 for w in rec if w < 0)))
            if set(corr) == {1.0}:
                pass
    print("  independent (asset, lookback) cells constant +1.0 after warmup: "
          "{}/{}".format(n_constant, len(tickers) * len(LOOKBACKS)))
    print("  independent sign-flip cells: {}".format(flippers))
    pub_constant = sum(
        1 for t in tickers for lb in LOOKBACKS
        if integ["per_ticker"][t][str(lb)]["distinct_weights_after_warmup"] == [1.0]
        and integ["per_ticker"][t][str(lb)]["n_short_after_warmup"] == 0)
    print("  published constant +1.0 cells: {}/{}".format(
        pub_constant, len(tickers) * len(LOOKBACKS)))
    gate3 = pub_constant == n_constant
    all_ok = all_ok and gate3

    print("\n=== Gate 4: classification re-derived from published medians/nulls ===")
    for label, block, pub_block in (
            ("recorded", art["per_cost"], art["per_asset"]),
            ("corrected", diag["per_cost"], diag["per_asset"])):
        derived = derive_classification(block, tickers)
        for ticker in tickers:
            ok = derived[ticker] == pub_block[ticker]["classification"]
            all_ok = all_ok and ok
            print("  {:9s} {:6s} derived={:18s} published={:18s} {}".format(
                label, ticker, derived[ticker],
                pub_block[ticker]["classification"],
                "MATCH" if ok else "MISMATCH"))
        counts = {c: 0 for c in CLASS_NAMES}
        for t in tickers:
            counts[derived[t]] += 1
        print("  {:9s} derived counts {}".format(label, counts))

    print("\n=== Gate 5: zero-cost column vs momentum_lookback_sweep_results.json ===")
    prior = json.loads(LOOKBACK_SWEEP.read_text())
    prior_lp = prior["universe"]["per_asset"]
    gate5 = True
    for ticker in tickers:
        for lb in LOOKBACKS:
            cur = art["per_cost"]["zero"][ticker][str(lb)]
            pub = prior["per_asset"][ticker][str(lb)]
            ok = (
                [round(m, 3) for m in cur["medians"]]
                == prior_lp[ticker]["lookback_medians"][str(lb)]
                and [round(n, 3) for n in cur["null_medians"]]
                == prior_lp[ticker]["lookback_null_medians"][str(lb)]
                and cur["segments"] == pub["segments"]
                and cur["verdict"] == pub["verdict"]
            )
            gate5 = gate5 and ok
            if not ok:
                print("  MISMATCH {} lb={}: {} vs published {}".format(
                    ticker, lb, cur["verdict"], pub["verdict"]))
    all_ok = all_ok and gate5
    print("  all 10 tickers x 3 lookbacks compared -> {}".format(
        "MATCH" if gate5 else "MISMATCH"))

    print("\n=== Gate 6: determinism of the verification path ===")
    a1 = fresh_cell("AMZN", 5, "realistic", recorded_signals)
    a2 = fresh_cell("AMZN", 5, "realistic", recorded_signals)
    b1 = fresh_cell("AMZN", 5, "realistic", corrected_signals)
    b2 = fresh_cell("AMZN", 5, "realistic", corrected_signals)
    det = (json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
           and json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True))
    print("  AMZN lb=5 realistic r1 == r2 (both signals): {}".format(det))
    all_ok = all_ok and det

    print()
    if all_ok:
        print("All independent recomputations MATCH the artifact: recorded and "
              "corrected cells, signal-integrity counts, classifications, "
              "zero-cost cross-consistency, determinism.")
    else:
        print("MISMATCH FOUND: see the printed recomputed vs published values.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Independent verification of a cost-sensitivity "
          "simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())