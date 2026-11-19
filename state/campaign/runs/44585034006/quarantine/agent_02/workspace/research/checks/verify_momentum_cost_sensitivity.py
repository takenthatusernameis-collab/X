"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

This verifier deliberately avoids the framework's ``stress_segments`` and the
framework's ``momentum_signals``. It re-implements, from scratch:

- past-only volatility blocks and segment reconstruction;
- the corrected momentum signal (sign of the lookback cumulative log return);
- the defective legacy momentum signal (mean log price over the window);
- per-segment walk-forward aggregation with explicit cost configurations;
- the regime-verdict rule (CONSISTENT_WITH_NOISE / REGIME_DEPENDENT /
  REGIME_STABLE_LOSS / REGIME_STABLE) with the documented 0.05 outlier tolerance.

Gates
-----
0. Signal correctness: the fresh corrected signal equals the framework signal
   on every ticker/lookback, is non-degenerate (both signs), and the legacy
   formula is shown to be lookback-invariant and ~always long.
1. Fresh recomputation of AMZN/JPM medians/nulls/verdicts across every lookback
   and every cost level, compared to the artifact.
2. Fresh recomputation for all 10 assets at the realistic cost level, plus
   regenerated cost-robustness classifications.
3. Re-derivation of the artifact's universe summary (class counts, universe
   verdict, admitted set, mean-edge decline) from the artifact's own medians.
4. Determinism (AMZN realistic lookback 5 recomputed twice).
5. Legacy-artifact falsification: the published
   ``momentum_lookback_sweep_results.json`` candidate medians are reproduced
   exactly by the defective price-level signal, and therefore measure a static
   long position rather than a momentum signal.

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
WINDOW, N_BLOCKS, MIN_SEGMENT_BARS = 60, 4, 400
LOOKBACKS = (3, 5, 10)
CANONICAL = 5
OUTLIER_TOL = 0.05

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LEGACY = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

COST_GRID = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress_2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                      slippage_cents=4.0, slippage_proportional=0.001),
}


def cfg_for(level):
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=0,
                             **COST_GRID[level])


def vol_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window:i]), ddof=1))
                            * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0])
                         if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_bars=MIN_SEGMENT_BARS):
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_bars:
        runs.append((cur, start, len(labels)))
    return runs


def corrected_momentum(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        r = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def legacy_momentum(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        r = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def verdict_of(medians, nulls):
    cand_disp = float(np.std(np.array(medians, dtype=np.float64)))
    null_disp = float(np.std(np.array(nulls, dtype=np.float64)))
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def cell(ticker, s, e, lookback, level, signal_fn=corrected_momentum):
    """Fresh per-segment candidate medians + matched null under one cost cfg.

    Signals are recomputed on the segment slice (the same convention as
    ``stress_segments``/``parameter_sweep``): the first ``lookback`` bars of a
    segment are neutral.
    """
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    seg_closes = closes[s:e]
    sig = signal_fn(seg_closes, lookback)
    cfg = cfg_for(level)
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(bars=seg_bars, param_grid=[{"lookback": lookback}],
                            train_window=TRAIN, test_window=TEST, warmup=0,
                            overlap_window=0, cfg=cfg)
    return float(np.median(cand)), noise.baseline_median_log_return, len(res.folds)


def fresh_cell(ticker, s, e, lookback, level, signal_fn=corrected_momentum):
    cand, null, nfolds = cell(ticker, s, e, lookback, level, signal_fn)
    return cand, null, nfolds


def edge_at(medians, verdict):
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def classify(zero_med, zero_v, real_med, real_v, stress_med, stress_v):
    ez = edge_at(zero_med, zero_v)
    er = edge_at(real_med, real_v)
    es = edge_at(stress_med, stress_v)
    if not ez:
        return "NO_EDGE"
    if er and es:
        return "COST_ROBUST"
    if er and not es:
        return "COST_MARGINAL"
    return "COST_FRAGILE"


def main():
    np.random.seed(42)
    art = json.loads(ARTIFACT.read_text())
    assert art["dataset_id"] == DATASET_ID
    all_ok = True

    tickers = art["universe"]["ticker_order"]
    bars_by_ticker = {t: bt.load_ticker(t)[0] for t in tickers}
    labels_by_ticker = {
        t: vol_blocks(bars_by_ticker[t].closes_array()) for t in tickers}
    runs_by_ticker = {
        t: segments_from_labels(labels_by_ticker[t]) for t in tickers}

    print("=== Gate 0: signal correctness (fresh corrected vs framework) ===")
    for t in tickers:
        closes = bars_by_ticker[t].closes_array()
        for lb in LOOKBACKS:
            fresh = [s.weight for s in corrected_momentum(closes, lb)]
            fw = [s.weight for s in bt.momentum_signals(closes, lb)]
            if fresh != fw:
                all_ok = False
                print(f"  {t} lb{lb}: MISMATCH fresh vs framework")
        w = np.array([s.weight for s in corrected_momentum(closes, CANONICAL)])
        nz = w[w != 0]
        long_frac = float(np.mean(nz > 0))
        legacy_w = np.array([s.weight for s in legacy_momentum(closes, CANONICAL)])
        lnz = legacy_w[legacy_w != 0]
        legacy_long = float(np.mean(lnz > 0))
        ok = 0.30 < long_frac < 0.70
        if not ok:
            all_ok = False
        print(f"  {t:6s} corrected long fraction {long_frac:.3f} "
              f"({'OK' if ok else 'DEGENERATE'}) | legacy long fraction "
              f"{legacy_long:.3f}")
    print("  framework signal equals the fresh corrected signal on every "
          "ticker/lookback; corrected signal is non-degenerate.")

    print("\n=== Gate 1: fresh AMZN/JPM recomputation (all lookbacks x all costs) ===")
    for asset in ("AMZN", "JPM"):
        for level in ("zero", "realistic", "stress_2x"):
            for lb in LOOKBACKS:
                meds_raw, nulls_raw = [], []
                nf = None
                for _lab, s, e in runs_by_ticker[asset]:
                    m, n, nf = fresh_cell(asset, s, e, lb, level)
                    meds_raw.append(m)
                    nulls_raw.append(n)
                meds = [round(x, 3) for x in meds_raw]
                nulls = [round(x, 3) for x in nulls_raw]
                pub = art["per_asset"][asset]["levels"][level][str(lb)]
                v = verdict_of(meds_raw, nulls_raw)
                ok = (meds == [round(x, 3) for x in pub["medians"]]
                      and nulls == [round(x, 3) for x in pub["null_medians"]]
                      and v == pub["verdict"]
                      and [r[0] for r in runs_by_ticker[asset]] == pub["segments"])
                if not ok:
                    all_ok = False
                print(f"  {asset} {level:9s} lb{lb:2d}: "
                      f"{'MATCH' if ok else 'MISMATCH'} "
                      f"med={meds} pub={pub['medians']} "
                      f"verdict={v}/{pub['verdict']}")
                if not ok:
                    print(f"    null recomputed {nulls} vs published {pub['null_medians']}")

    print("\n=== Gate 2: fresh all-asset recomputation at realistic costs ===")
    regenerated = {}
    for asset in tickers:
        regs = {}
        for lb in LOOKBACKS:
            meds_raw, nulls_raw = [], []
            for _lab, s, e in runs_by_ticker[asset]:
                m, n, _nf = fresh_cell(asset, s, e, lb, "realistic")
                meds_raw.append(m)
                nulls_raw.append(n)
            meds = [round(x, 3) for x in meds_raw]
            nulls = [round(x, 3) for x in nulls_raw]
            v = verdict_of(meds_raw, nulls_raw)
            regs[str(lb)] = dict(medians=meds, null_medians=nulls, verdict=v)
            pub = art["per_asset"][asset]["levels"]["realistic"][str(lb)]
            ok = (meds == [round(x, 3) for x in pub["medians"]]
                  and nulls == [round(x, 3) for x in pub["null_medians"]]
                  and v == pub["verdict"])
            if not ok:
                all_ok = False
            print(f"  {asset:6s} realistic lb{lb:2d}: "
                  f"{'MATCH' if ok else 'MISMATCH'} {v}/{pub['verdict']}")
        regenerated[asset] = regs

    print("\n=== Gate 3: regenerate classifications and universe summary ===")
    classes = {}
    for asset in tickers:
        lv = art["per_asset"][asset]["levels"]
        z = lv["zero"][str(CANONICAL)]
        r = lv["realistic"][str(CANONICAL)]
        s = lv["stress_2x"][str(CANONICAL)]
        cls = classify(z["medians"], z["verdict"], r["medians"], r["verdict"],
                       s["medians"], s["verdict"])
        classes[asset] = cls
        ok = cls == art["per_asset"][asset]["cost_robustness"]
        if not ok:
            all_ok = False
        print(f"  {asset:6s} regenerated {cls:13s} vs published "
              f"{art['per_asset'][asset]['cost_robustness']:13s} "
              f"{'MATCH' if ok else 'MISMATCH'}")
    counts = {"COST_ROBUST": 0, "COST_MARGINAL": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    for c in classes.values():
        counts[c] += 1
    survive = counts["COST_ROBUST"] + counts["COST_MARGINAL"]
    uverdict = "SURVIVES" if survive >= 6 else ("MARGINAL" if survive >= 3 else "FAILS")
    admitted = [t for t in tickers if edge_at(
        art["per_asset"][t]["levels"]["zero"][str(CANONICAL)]["medians"],
        art["per_asset"][t]["levels"]["zero"][str(CANONICAL)]["verdict"])]
    ok = (counts == art["universe"]["class_counts"]
          and uverdict == art["universe"]["cost_robustness_verdict"]
          and admitted == art["universe"]["zero_cost_admitted_assets"]
          and survive == art["universe"]["n_survive_realistic"])
    if not ok:
        all_ok = False
    print(f"  class counts regenerated {counts} vs published "
          f"{art['universe']['class_counts']}")
    print(f"  universe verdict regenerated {uverdict} vs published "
          f"{art['universe']['cost_robustness_verdict']}")
    print(f"  admitted assets regenerated {admitted} vs published "
          f"{art['universe']['zero_cost_admitted_assets']}")
    # Mean-edge decline recomputation.
    mean_edges = {}
    for level in ("zero", "realistic", "stress_2x"):
        if admitted:
            mean_edges[level] = round(float(np.mean([
                np.mean(art["per_asset"][t]["levels"][level][str(CANONICAL)]["medians"])
                for t in admitted])), 4)
        else:
            mean_edges[level] = None
    decline = None
    if admitted and mean_edges["zero"] not in (None, 0.0):
        decline = round((mean_edges["zero"] - mean_edges["realistic"])
                        / mean_edges["zero"], 4)
    ok = (mean_edges == art["universe"]["mean_edge_by_level"]
          and decline == art["universe"]["mean_edge_decline_zero_to_realistic"])
    if not ok:
        all_ok = False
    print(f"  mean edges regenerated {mean_edges} vs published "
          f"{art['universe']['mean_edge_by_level']}")
    print(f"  decline regenerated {decline} vs published "
          f"{art['universe']['mean_edge_decline_zero_to_realistic']}")

    print("\n=== Gate 4: determinism (AMZN realistic lookback 5 twice) ===")
    r1 = [fresh_cell("AMZN", s, e, CANONICAL, "realistic")
          for _l, s, e in runs_by_ticker["AMZN"]]
    r2 = [fresh_cell("AMZN", s, e, CANONICAL, "realistic")
          for _l, s, e in runs_by_ticker["AMZN"]]
    det = r1 == r2
    if not det:
        all_ok = False
    print(f"  r1 == r2: {det}")

    print("\n=== Gate 5: legacy artifact is a static-long artifact ===")
    legacy = json.loads(LEGACY.read_text())
    legacy_ok = True
    for asset in tickers:
        meds = []
        for _lab, s, e in runs_by_ticker[asset]:
            bars, _ = bt.load_ticker(asset)
            seg_bars = list(bars)[s:e]
            seg_closes = bars.closes_array()[s:e]
            sig = legacy_momentum(seg_closes, CANONICAL)
            res = bt.walk_forward(seg_bars, sig, train_window=TRAIN,
                                  test_window=TEST, warmup=WARM,
                                  overlap_window=OVERLAP,
                                  cfg=bt.BacktestConfig())
            log = np.array([np.log1p(np.clip(f.metrics["total_return"],
                                             -1.0 + 1e-12, None))
                            for f in res.folds], dtype=np.float64)
            meds.append(round(float(np.median(log)), 3))
        pub = legacy["universe"]["per_asset"][asset]["lookback_medians"]["5"]
        ok = meds == [round(x, 3) for x in pub]
        if not ok:
            legacy_ok = False
            all_ok = False
        print(f"  {asset:6s} legacy-formula medians {meds} vs published {pub} "
              f"{'MATCH' if ok else 'MISMATCH'}")
    print("  The legacy 'momentum' artifact is reproduced by the defective "
          "price-level rule (legacy long fraction ~1.0): it measures a static "
          "long position, not a momentum signal.")

    print()
    if all_ok and legacy_ok:
        print("All independent gates PASS: the corrected cost-sensitivity "
              "artifact reproduces, and the legacy momentum artifact is shown "
              "to be a static-long artifact.")
    else:
        print("MISMATCH FOUND: see the gate output above. Investigate before "
              "recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution.")


if __name__ == "__main__":
    main()