"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates (all independent of ``stress_segments`` / ``noise_benchmark``):

1. Fresh recomputation of the AMZN/JPM candidate and matched-null medians for
   every lookback {3,5,10} and cost level {0.0,0.5,1.0,2.0}, using a separate
   code path: re-implemented volatility blocks, segment reconstruction,
   momentum signals, an inlined coin-flip null, and ``walk_forward`` directly.
   Every median/null/segment/verdict is compared to the artifact.

2. Zero-cost cross-consistency: the zero-cost column for all 10 tickers is
   recomputed from scratch and must match BOTH the new artifact and the prior
   `momentum_lookback_sweep_results.json` (the independent regression tie to
   the already-admitted zero-cost momentum evidence).

3. Classification reproducibility: the a-priori cost classes and edge counts
   are re-derived from the artifact medians/verdicts and compared to the
   artifact's ``classification.predeclared_rule``.

4. Instrument audit reproduction: independently rebuild the NVDA lookback-5
   m=0.5 null first fold and confirm the matched null goes insolvent (negative
   OOS equity) so that ``total_return = end/start - 1`` is a negative/negative
   ratio, which is what invalidates the cost-robustness classification.

5. Determinism of this verification path.
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
REALISTIC = dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005)

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
PRIOR = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def cfg_for(mult):
    return bt.BacktestConfig(**{k: v * float(mult) for k, v in REALISTIC.items()})


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    bs = n // n_blocks
    bv = []
    for i in range(n):
        bv.append(0.0 if i < window else
                  float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bv if v > 0]
    med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // bs
            s, e = b * bs, (b + 1) * bs
            bmed = float(np.median([v for v in bv[s:e] if v > 0]) if e - s > window else 0.0)
            labels.append("turbulent" if bmed > med else "calm")
    return labels


def segments_from_labels(labels, mb):
    runs, cur, st = [], labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - st >= mb:
                runs.append((cur, st, i))
            cur, st = labels[i], i
    if len(labels) - st >= mb:
        runs.append((cur, st, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    """Fresh momentum implementation (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lb = int(lookback)
    for i in range(lb, n):
        ret = float(np.mean(np.log(closes[i - lb + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def random_signals(n, seed):
    """Inlined coin-flip null (independent of perturbation.random_signals)."""
    rng = np.random.default_rng(seed)
    w = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(x)) for i, x in enumerate(w)]


def _median_log(res):
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def cand_segment(seg_bars, lookback, mult):
    closes = np.array([b.close for b in seg_bars], dtype=np.float64)
    sig = momentum_signals(closes, lookback)
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg_for(mult))
    return _median_log(res)


def null_segment(seg_bars, lookback, mult):
    n = len(seg_bars)
    seed = 42 + int(lookback) * 1000
    sig = random_signals(n, seed)
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=0, overlap_window=0, cfg=cfg_for(mult))
    return _median_log(res)


def regen_verdict(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def is_edge(medians, verdict):
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def classify(edge_counts):
    E = edge_counts
    if E["0.0"] == 0:
        return "NO_EDGE_AT_ZERO_COST"
    if E["1.0"] == 0:
        return "COST_FRAGILE"
    if E["2.0"] >= 1:
        return "COST_ROBUST"
    return "COST_ERODING"


def load_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load_json(ARTIFACT)
    prior = load_json(PRIOR)
    assert art["dataset_id"] == DATASET_ID
    all_ok = True

    print("=== Gate 1: fresh walk_forward + inlined null (AMZN, JPM; all lookbacks x costs) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            for mult in COST_MULTIPLIERS:
                meds, nulls = [], []
                for lab, s, e in runs:
                    seg = list(bars)[s:e]
                    m, _ = cand_segment(seg, lb, mult)
                    nm, _ = null_segment(seg, lb, mult)
                    meds.append(round(m, 3))
                    nulls.append(round(nm, 3))
                rv = regen_verdict(meds, nulls)
                pub = art["per_cell"][str(lb)][str(mult)]["per_asset"][ticker]
                ok = (meds == pub["medians"] and nulls == pub["null_medians"]
                      and lbls == pub["segments"] and rv == pub["verdict"])
                if not ok:
                    all_ok = False
                    print("  MISMATCH {} lb={} m={}: meds {} vs {} | nulls {} vs {} | "
                          "verdict {} vs {}".format(
                              ticker, lb, mult, meds, pub["medians"], nulls,
                              pub["null_medians"], rv, pub["verdict"]))
        print("  {}: all lookbacks x costs recomputed".format(ticker))

    print("\n=== Gate 2: zero-cost column cross-consistency (all 10 tickers) ===")
    prior_assets = prior["universe"]["per_asset"]
    for entry in art["ticker_order"]:
        ticker = entry
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        for lb in LOOKBACKS:
            meds = [round(cand_segment(list(bars)[s:e], lb, 0.0)[0], 3)
                    for lab, s, e in runs]
            nulls = [round(null_segment(list(bars)[s:e], lb, 0.0)[0], 3)
                     for lab, s, e in runs]
            pub_new = art["per_cell"][str(lb)]["0.0"]["per_asset"][ticker]
            ok_new = (meds == pub_new["medians"] and nulls == pub_new["null_medians"])
            ok_prior = (meds == prior_assets[ticker]["lookback_medians"][str(lb)]
                        and nulls == prior_assets[ticker]["lookback_null_medians"][str(lb)])
            if not (ok_new and ok_prior):
                all_ok = False
                print("  MISMATCH {} lb={}: meds {} (new {} / prior {})".format(
                    ticker, lb, meds, pub_new["medians"],
                    prior_assets[ticker]["lookback_medians"][str(lb)]))
    print("  all 10 tickers x 3 lookbacks zero-cost recomputed -> new + prior")

    print("\n=== Gate 3: classification reproducibility ===")
    pre = art["classification"]["predeclared_rule"]
    for lb in LOOKBACKS:
        ec = {}
        for mult in COST_MULTIPLIERS:
            cell = art["per_cell"][str(lb)][str(mult)]["per_asset"]
            ec[str(mult)] = sum(
                1 for t in cell
                if is_edge(cell[t]["medians"], regen_verdict(cell[t]["medians"],
                                                             cell[t]["null_medians"])))
        cls = classify(ec)
        ok = (cls == pre["per_lookback"][str(lb)]
              and ec == pre["edge_counts"][str(lb)])
        if not ok:
            all_ok = False
        print("  lookback {}: edge_counts={} class={} (pub {} {}) -> {}".format(
            lb, ec, cls, pre["edge_counts"][str(lb)],
            pre["per_lookback"][str(lb)], "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 4: instrument audit reproduction (NVDA null insolvency) ===")
    bars, _ = bt.load_ticker("NVDA")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    lab, s, e = runs[0]
    seg = list(bars)[s:e]
    n = len(seg)
    sig = random_signals(n, 42 + 5 * 1000)
    fold_bars = seg[: TRAIN + TEST]
    fold_sig = sig[: TRAIN + TEST]
    res = bt.Backtest(cfg_for(0.5)).run(fold_bars, fold_sig)
    eq = res.equity_curve
    oos = eq[-TEST:]
    start_eq, end_eq = float(oos[0]), float(oos[-1])
    tr = end_eq / start_eq - 1.0
    audit = art["classification"]["instrument_audit"]
    ok = (round(start_eq, 1) == audit["oos_start_equity"]
          and round(end_eq, 1) == audit["oos_end_equity"]
          and round(tr, 3) == audit["total_return"]
          and start_eq < 0.0 and tr > 0.0)
    if not ok:
        all_ok = False
    print("  NVDA lb=5 m=0.5 seg0: OOS start={:.1f} end={:.1f} total_return={:+.3f} "
          "(audit {} {} {}) -> {}".format(
              start_eq, end_eq, tr, audit["oos_start_equity"],
              audit["oos_end_equity"], audit["total_return"],
              "MATCH" if ok else "MISMATCH"))
    print("  decision_relevant_classification in artifact: {}".format(
        art["classification"]["decision_relevant"]["classification"]))

    print("\n=== Gate 5: determinism ===")
    bars, _ = bt.load_ticker("AMZN")
    runs = segments_from_labels(vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW),
                                MIN_SEGMENT_BARS)
    m1 = [round(cand_segment(list(bars)[s:e], 5, 1.0)[0], 3) for lab, s, e in runs]
    m2 = [round(cand_segment(list(bars)[s:e], 5, 1.0)[0], 3) for lab, s, e in runs]
    det = m1 == m2
    if not det:
        all_ok = False
    print("  AMZN lb=5 m=1.0 r1={} r2={} -> {}".format(m1, m2, "identical" if det else "DIFFERENT"))

    print()
    if all_ok:
        print("All independent recomputations MATCH the artifact; the zero-cost "
              "column reproduces the prior sweep; the a-priori classes reproduce; "
              "and the null-insolvency instrument audit is independently confirmed.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()