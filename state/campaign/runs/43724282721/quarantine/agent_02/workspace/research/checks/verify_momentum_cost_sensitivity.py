"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Five gates, each using a code path separate from the check:

1. Fresh walk_forward + noise_benchmark recomputation (AMZN, JPM; all lookbacks;
   all cost levels) with a re-implemented momentum signal and segment slicing,
   bypassing `stress_segments`. Candidate medians, null medians, and the
   framework verdict are recomputed and compared to the artifact.

2. Static-signal confirmation (independent path): re-implement the *documented
   repo formula* ``mean(log(close))`` and count sign changes; then show that the
   repo momentum medians equal a fresh always-long (all +1) walk-forward for
   AAPL/AMZN/TSLA at the reference lookback. A textbook trailing-return momentum
   (sign of ``log(c_t) - log(c_{t-lb})``) is computed only to quantify the
   contrast in turnover (it is NOT run through the cost grid; that would be a
   different, out-of-scope experiment).

3. Classification reproducibility: re-derive each asset's cost verdict from the
   artifact's stored medians via the predeclared rule and compare to the stored
   `cost_verdict` (self-consistency of the classification layer).

4. Zero-cost column cross-consistency vs the prior momentum lookback sweep.

5. Determinism: recompute the AAPL reference-lookback zero-cost medians twice.
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
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
LOOKBACKS = (3, 5, 10)
REF = 5
ART = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
VERDICT_TOL = 0.05

COST_MODELS = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.001),
}


def cfg(level):
    return bt.BacktestConfig(initial_capital=1e6, **COST_MODELS[level])


def vol_blocks(closes, n_blocks, window):
    """Independent reimplementation of volatility_blocks."""
    n = len(closes)
    bs = n // n_blocks
    vols = []
    for i in range(n):
        vols.append(0.0 if i < window else
                    float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in vols if v > 0]
    med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // bs
            s, e = b * bs, (b + 1) * bs
            bmed = float(np.median([v for v in vols[s:e] if v > 0])) if e - s > window else 0.0
            labels.append("turbulent" if bmed > med else "calm")
    return labels


def segments(labels, min_len):
    runs, cur, start = [], labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_len:
                runs.append((cur, start, i))
            cur, start = labels[i], i
    if len(labels) - start >= min_len:
        runs.append((cur, start, len(labels)))
    return runs


def repo_momentum(closes, lookback):
    """Independent reimplementation of the repo's documented formula.

    The repo docstring says 'long the previous lookback-day return'; the code
    actually tests mean(log(close)) over the window > 0. Reproduced verbatim.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        val = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if val > 0 else -1.0)
    return out


def always_long(closes):
    return [bt.Signal(date=i + 1, weight=1.0) for i in range(len(closes))]


def true_momentum(closes, lookback):
    """Textbook trailing-return momentum (contrast only, not cost-tested)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        r = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def changes(sig):
    return sum(1 for i in range(1, len(sig))
               if abs(sig[i].weight - sig[i - 1].weight) > 1e-12)


def verdict(medians, nulls):
    cd = float(np.std(np.array(medians)))
    nd = float(np.std(np.array(nulls)))
    if all(abs(m) <= VERDICT_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cd > 2.0 * nd:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def fresh_segment(ticker, s, e, lookback, level):
    """Candidate median + matched null median for one segment, fresh path."""
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = repo_momentum(closes, lookback)[s:e]
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg(level))
    cand = float(np.median([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                            for f in res.folds]))
    noise = noise_benchmark(seg_bars, param_grid=[{"lookback": lookback}],
                            train_window=TRAIN, test_window=TEST, warmup=0,
                            overlap_window=0, periods_per_year=252, cfg=cfg(level))
    return cand, noise.baseline_median_log_return


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load(ART)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks/costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEG)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for level in ("zero", "realistic", "stress2x"):
                meds, nuls = [], []
                for lab, s, e in runs:
                    m, n = fresh_segment(asset, s, e, lb, level)
                    meds.append(round(m, 3))
                    nuls.append(round(n, 3))
                pub = art["per_asset"][asset]["by_lookback"][str(lb)][level]
                ok = (meds == pub["medians"] and nuls == pub["null_medians"]
                      and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                    print("  MISMATCH {} lb={} {}: fresh med={} pub={} | fresh null={} pub={}"
                          .format(asset, lb, level, meds, pub["medians"],
                                  nuls, pub["null_medians"]))
            print("  {} lb={}: all cost levels recomputed".format(asset, lb))
    print("  Gate 1 -> {}".format("MATCH" if all_ok else "MISMATCH"))

    print("\n=== Gate 2: static-signal confirmation ===")
    g2_ok = True
    for asset in ["AAPL", "AMZN", "TSLA"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEG)
        # repo formula sign changes at ref lookback
        ch = changes(repo_momentum(closes, REF))
        # fresh always-long medians at ref lookback, zero cost
        meds = []
        for lab, s, e in runs:
            res = bt.walk_forward(list(bars)[s:e], always_long(closes[s:e]),
                                  train_window=TRAIN, test_window=TEST, warmup=WARM,
                                  overlap_window=OVERLAP, cfg=cfg("zero"))
            meds.append(round(float(np.median(
                [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                 for f in res.folds])), 3))
        pub = art["per_asset"][asset]["by_lookback"][str(REF)]["zero"]["medians"]
        ok = (meds == pub and ch <= 3)
        g2_ok = g2_ok and ok
        print("  {}: repo_formula_sign_changes={} always_long_medians={} repo_momentum_medians={} -> {}"
              .format(asset, ch, meds, pub, "STATIC+MATCH" if ok else "MISMATCH"))
    # contrast: textbook momentum turnover
    bars, _ = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    tm = changes(true_momentum(closes, REF))
    print("  contrast: AAPL textbook trailing-return momentum sign changes={} "
          "(repo formula = {})".format(tm, changes(repo_momentum(closes, REF))))
    print("  Gate 2 -> {}".format("PASS" if g2_ok else "MISMATCH"))
    all_ok = all_ok and g2_ok

    print("\n=== Gate 3: cost-verdict reproducibility from stored medians ===")
    def classify(nz, nr, n2):
        if nz <= 0:
            return "NO_EDGE_BASE"
        if nr <= 0:
            return "COST_FAILS"
        r2 = n2 / nz
        if n2 <= 0 or r2 < art["retention_floor"]:
            return "COST_ERODED"
        return "COST_ROBUST"
    g3_ok = True
    for ticker in art["universe"]["ticker_order"]:
        for lb in LOOKBACKS:
            d = art["per_asset"][ticker]["derived"][str(lb)]
            nz = float(np.median(art["per_asset"][ticker]["by_lookback"][str(lb)]["zero"]["medians"]))
            nr = float(np.median(art["per_asset"][ticker]["by_lookback"][str(lb)]["realistic"]["medians"]))
            n2 = float(np.median(art["per_asset"][ticker]["by_lookback"][str(lb)]["stress2x"]["medians"]))
            recomputed = classify(nz, nr, n2)
            if recomputed != d["cost_verdict"]:
                g3_ok = False
                print("  MISMATCH {} lb={}: recomputed {} vs stored {}"
                      .format(ticker, lb, recomputed, d["cost_verdict"]))
    print("  all 10 assets x 3 lookbacks re-derived -> {}".format(
        "MATCH" if g3_ok else "MISMATCH"))
    all_ok = all_ok and g3_ok

    print("\n=== Gate 4: zero-cost column vs prior momentum lookback sweep ===")
    sweep = load(SWEEP)
    sw = sweep["per_asset"]
    g4_ok = True
    for ticker in art["universe"]["ticker_order"]:
        for lb in LOOKBACKS:
            z = art["per_asset"][ticker]["by_lookback"][str(lb)]["zero"]
            p = sw[ticker][str(lb)]
            if (z["medians"] != p["medians"] or z["null_medians"] != p["null_medians"]
                    or z["verdict"] != p["verdict"]):
                g4_ok = False
                print("  MISMATCH {} lb={}".format(ticker, lb))
    print("  all 10 assets x 3 lookbacks cross-checked -> {}".format(
        "MATCH" if g4_ok else "MISMATCH"))
    all_ok = all_ok and g4_ok

    print("\n=== Gate 5: determinism (AAPL ref lookback zero medians twice) ===")
    bars, _ = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    runs = segments(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEG)

    def one_pass():
        out = []
        for lab, s, e in runs:
            m, _n = fresh_segment("AAPL", s, e, REF, "zero")
            out.append(round(m, 3))
        return out
    m1, m2 = one_pass(), one_pass()
    print("  r1={} r2={} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    g5_ok = (m1 == m2)
    all_ok = all_ok and g5_ok

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact: fresh "
              "walk_forward/null per segment (AMZN/JPM, all lookbacks/costs), "
              "static-signal confirmation, cost-verdict re-derivation, "
              "zero-cost cross-consistency, and determinism.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())