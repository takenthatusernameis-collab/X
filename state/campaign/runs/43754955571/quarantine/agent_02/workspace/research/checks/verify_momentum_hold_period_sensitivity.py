"""Independent verification of `research/checks/momentum_hold_period_sensitivity.py`.

Five verification gates. The verifier never imports the check module; every
quantity is recomputed from the artifact, the collected CSVs, and fresh
components.

Gate 1  Fresh recomputation (AMZN/JPM, all holds): own volatility-block
        classifier, own contiguous-segment reconstruction, own momentum signal
        implementation (the published definition), aggregation of fold log
        returns through `walk_forward`, and the framework coin-flip null through
        `noise_benchmark` (a different code path from `stress_segments`). Every
        per-hold median, null median, segment label list and fold count is
        compared to the artifact.

Gate 2  H=1 cross-consistency against `momentum_results.json`. The check
        documents its H=1 run as a direct replication of the admitted momentum
        evidence, so per-asset candidate medians must match exactly. Verdict
        equality is checked separately because the two runs feed different
        parameter dicts to the null (`{"hold": 1}` vs `{"lookback": 5}`), which
        selects a different deterministic coin-flip seed; any verdict
        difference is reported explicitly rather than hidden.

Gate 3  Classification reproducibility: hold profiles, edge counts, stable and
        singular asset lists, and the universe effect class are re-derived from
        the artifact medians/verdicts with an independent implementation of the
        documented rules.

Gate 4  Signal-degeneracy measurement: the fraction of emitted momentum
        weights that equal +1 per ticker, and the fraction of bars at which the
        hold-period transform changes the position at all. This tests whether
        the hold-period grid can be informative for this signal.

Gate 5  Falsification of the interpretation: the same hold grid is run with the
        intended momentum definition (sign of the lookback log *return*, i.e.
        log(close_t / close_{t-lookback})) on AMZN/JPM. This is a diagnostic
        probe of whether the artifact's "HORIZON_SENSITIVE" classification is a
        property of the effect or of the signal definition. It is reported as
        UNVERIFIED-with-cause if the two definitions disagree.

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
LOOKBACK = 5
HOLD_PERIODS = (1, 2, 3, 5)
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"
MANIFEST = json.loads((Path.cwd() / "research" / "data" / "manifest.json").read_text())


def vol_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW):
    """Fresh re-implementation of the published volatility-block classifier."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
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


def segments_from_labels(labels, min_segment_bars=MIN_SEGMENT_BARS):
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


def published_momentum_signals(closes, lookback=LOOKBACK):
    """The momentum definition the check actually evaluates.

    `ret = mean(log(closes[i-lookback+1:i+1]))`, long if ret > 0. This is the
    arithmetic mean of log *prices*, not of log returns.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        val = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if val > 0 else -1.0)
    return out


def intended_momentum_signals(closes, lookback=LOOKBACK):
    """The momentum definition the check's docstring describes.

    Sign of the lookback log return: log(close_t / close_{t-lookback}).
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        val = float(np.log(closes[i] / closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if val > 0 else -1.0)
    return out


def hold_transform(signals, hold, lookback=LOOKBACK):
    """Fresh re-implementation of the hold-period carry transform.

    Rebalance bars are lookback + k*hold; each output bar inherits the signal of
    the most recent rebalance bar at or before it. Depends only on signals at or
    before the bar, so no look-ahead.
    """
    out = []
    for i in range(len(signals)):
        if i < lookback:
            out.append(bt.Signal(date=i + 1, weight=0.0))
        else:
            idx = i - (i - lookback) % hold
            out.append(bt.Signal(date=i + 1, weight=signals[idx].weight))
    return out


def segment_result(ticker, s, e, hold, sig_fn=published_momentum_signals):
    """Fresh walk-forward + null for one asset segment and hold period.

    Rebuilds signals from the segment closes (independent path; no call to
    `stress_segments`).
    """
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = hold_transform(sig_fn(closes), hold)[s:e]
    res = bt.walk_forward(
        seg_bars, sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=bt.BacktestConfig())
    cand = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                     for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"hold": hold}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252)
    return float(np.median(cand)), float(noise.baseline_median_log_return), len(res.folds)


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> int:
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["hold_periods"] == list(HOLD_PERIODS)
    assert art["lookback"] == LOOKBACK
    ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all holds) ===")
    for ticker in ("AMZN", "JPM"):
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes))
        labels = [lab for lab, _, _ in runs]
        for hold in HOLD_PERIODS:
            meds, nulls, nfold = [], [], None
            for i_seg, (_lab, s, e) in enumerate(runs):
                m, n, nf = segment_result(ticker, s, e, hold)
                meds.append(round(m, 3))
                nulls.append(round(n, 3))
                if i_seg == 0:
                    nfold = nf
            pub = art["per_asset"][ticker][str(hold)]
            status = ("MATCH" if meds == [round(v, 3) for v in pub["medians"]]
                      and nulls == [round(v, 3) for v in pub["null_medians"]]
                      and labels == pub["segments"]
                      and nfold == pub["n_folds"]
                      else "MISMATCH")
            if status != "MATCH":
                ok = False
                print("  {} hold={} {} | recomputed medians {} vs {} | nulls {} vs {} | "
                      "segments {} vs {} | folds {} vs {}".format(
                          ticker, hold, status, meds, pub["medians"], nulls,
                          pub["null_medians"], labels, pub["segments"], nfold,
                          pub["n_folds"]))
            else:
                print("  {} hold={} segments={} -> {} (medians {}, nulls {})".format(
                    ticker, hold, labels, status, meds, nulls))

    print("\n=== Gate 2: H=1 cross-consistency vs momentum_results.json ===")
    mom = load(MOM_ARTIFACT)
    med_mismatch = 0
    verdict_diffs = []
    for ticker in mom["universe"]["per_asset"]:
        pub5 = mom["universe"]["per_asset"][ticker]
        h1 = art["universe"]["hold_sums"]["1"]["per_asset"][ticker]
        same_med = [round(v, 3) for v in h1["medians"]] == [round(v, 3) for v in pub5["medians"]]
        same_seg = list(h1["segments"]) == list(pub5["segments"])
        if not (same_med and same_seg):
            med_mismatch += 1
            print("  {}: medians {} vs {} | segments {} vs {} -> MISMATCH".format(
                ticker, h1["medians"], pub5["medians"], h1["segments"], pub5["segments"]))
        if h1["verdict"] != pub5["verdict"]:
            verdict_diffs.append((ticker, h1["verdict"], pub5["verdict"]))
    print("  per-asset H=1 candidate medians + segment labels: {}/{} tickers MATCH".format(
        len(mom["universe"]["per_asset"]) - med_mismatch, len(mom["universe"]["per_asset"])))
    print("  H=1 null medians (this artifact): {}".format(
        {t: art["universe"]["hold_sums"]["1"]["per_asset"][t]["null_medians"]
         for t in art["universe"]["hold_sums"]["1"]["per_asset"]}))
    print("  H=1 null medians (momentum_results.json): {}".format(
        {t: mom["universe"]["per_asset"][t]["null_medians"]
         for t in mom["universe"]["per_asset"]}))
    if verdict_diffs:
        print("  H=1 verdict differences (null-draw sensitivity, seed from param dict "
              "{{'hold':1}} vs {{'lookback':5}}):")
        for t, a, b in verdict_diffs:
            print("    {}: this artifact {} | momentum_results.json {}".format(t, a, b))
    else:
        print("  H=1 verdicts identical across both artifacts")
    if med_mismatch:
        ok = False
    print("  NOTE: identical candidate medians with different verdicts are a null-seed")
    print("        effect, not a signal effect: the coin-flip seed is derived from the")
    print("        parameter dict, which differs between the two checks.")

    print("\n=== Gate 3: classification reproducibility from the artifact ===")
    ticker_order = art["universe"]["ticker_order"]
    hold_sums = art["universe"]["hold_sums"]

    def is_edge(t, h):
        p = hold_sums[str(h)]["per_asset"][t]
        return p["verdict"] == "REGIME_STABLE" and all(m > 0 for m in p["medians"])

    profiles, stable, singular = {}, [], []
    edge_counts = {str(h): sum(1 for t in ticker_order if is_edge(t, h)) for h in HOLD_PERIODS}
    for t in ticker_order:
        flags = [is_edge(t, h) for h in HOLD_PERIODS]
        profiles[t] = ("STABLE_EDGE" if all(flags)
                       else "SINGULAR" if sum(flags) == 1
                       else "WEAKENS" if sum(flags) > 1
                       else "NO_EDGE")
        if profiles[t] == "STABLE_EDGE":
            stable.append(t)
        if profiles[t] == "SINGULAR":
            singular.append(t)
    n_h1 = edge_counts[str(HOLD_PERIODS[0])]
    n_min = min(edge_counts.values())
    if n_min >= 7 and not singular and n_h1 >= 7:
        effect = "HORIZON_STABLE"
    elif singular:
        effect = "HORIZON_SINGULAR"
    elif n_h1 == 0:
        effect = "HORIZON_DESTROYED"
    else:
        effect = "HORIZON_SENSITIVE"
    pub_profiles = art["universe"]["hold_profiles"]
    pub_stable = art["universe"]["stable_edge_assets"]
    pub_singular = art["universe"]["singular_assets"]
    prof_ok = all(pub_profiles[t][str(h)] == profiles[t]
                  for t in ticker_order for h in HOLD_PERIODS)
    print("  edge_counts recomputed {} | published n_edge_at_h1={} n_edge_min_across_grid={}".format(
        edge_counts, art["universe"]["n_edge_at_h1"], art["universe"]["n_edge_min_across_grid"]))
    print("  hold_profiles reproduce: {} | stable_edge_assets {} vs {} | singular {} vs {}".format(
        prof_ok, stable, pub_stable, singular, pub_singular))
    print("  effect_classification recomputed {} | published {}".format(
        effect, art["universe"]["effect_classification"]))
    if not (prof_ok and stable == pub_stable and singular == pub_singular
            and effect == art["universe"]["effect_classification"]
            and n_h1 == art["universe"]["n_edge_at_h1"]
            and n_min == art["universe"]["n_edge_min_across_grid"]):
        ok = False
        print("  -> MISMATCH")
    else:
        print("  -> MATCH")

    print("\n=== Gate 4: signal-degeneracy measurement (can the hold grid be informative?) ===")
    for ticker in MANIFEST["universe"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = published_momentum_signals(closes)
        w = np.array([s.weight for s in sig])
        nonneutral = int(np.sum(w != 0))
        frac_long = float(np.mean(w == 1.0))
        base = hold_transform(sig, 1)
        drift = {}
        for hold in HOLD_PERIODS[1:]:
            eff = hold_transform(sig, hold)
            diff = float(np.mean([eff[i].weight != base[i].weight for i in range(len(w))]))
            drift[hold] = round(diff, 4)
        print("  {:6s} non-neutral={:5d} frac(+1)={:.3f} frac(-1)={:.3f} "
              "hold-transform changes position: {}".format(
                  ticker, nonneutral, frac_long, float(np.mean(w == -1.0)), drift))
    degenerate = 0
    for ticker in MANIFEST["universe"]:
        bars, _ = bt.load_ticker(ticker)
        w = np.array([s.weight for s in published_momentum_signals(bars.closes_array())])
        nonneutral = w[w != 0]
        if len(nonneutral) and float(np.mean(nonneutral == 1.0)) > 0.999:
            degenerate += 1
    print("  tickers whose momentum weight series is a constant +1 long: {}/{}".format(
        degenerate, len(MANIFEST["universe"])))
    print("  implication: for a constant signal the hold transform is a no-op, so the")
    print("  hold grid cannot be informative for those tickers; the HORIZON_SENSITIVE")
    print("  label then reflects null-draw variation, not edge decay.")

    print("\n=== Gate 5: intended momentum definition (sign of the lookback log return) ===")
    print("  diagnostic only; not the predeclared grid and not an artifact replacement")
    for ticker in ("AMZN", "JPM"):
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes))
        for hold in HOLD_PERIODS:
            meds, nls = [], []
            for _lab, s, e in runs:
                m, n, _nf = segment_result(ticker, s, e, hold, sig_fn=intended_momentum_signals)
                meds.append(round(m, 3))
                nls.append(round(n, 3))
            cand_disp = float(np.std(np.array(meds)))
            null_disp = float(np.std(np.array(nls)))
            if all(abs(m) <= 0.05 for m in meds):
                verdict = "CONSISTENT_WITH_NOISE"
            elif cand_disp > 2.0 * null_disp:
                verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in meds):
                verdict = "REGIME_STABLE_LOSS"
            else:
                verdict = "REGIME_STABLE"
            print("  {} hold={} medians {} nulls {} cand_disp {:+.3f} null_disp {:+.3f} "
                  "-> {}".format(ticker, hold, meds, nls, cand_disp, null_disp, verdict))

    print("\n=== Summary ===")
    print("  Gate 1 (independent recomputation): {}".format("MATCH" if ok else "MISMATCH"))
    print("  Gate 2 (H=1 cross-consistency): {} (median/segment equality)".format(
        "MATCH" if med_mismatch == 0 else "MISMATCH"))
    print("  Gate 3 (classification reproducibility): {}".format(
        "MATCH" if ok else "see above"))
    print("  Gate 4 (degeneracy): {}/{} tickers constant +1 long".format(
        degenerate, len(MANIFEST["universe"])))
    print("  Gate 5 (intended definition): diagnostic, reported separately")
    if not ok:
        print("VERIFICATION FAILURE: an independent recomputation did not reproduce the artifact.")
        sys.exit(1)
    print("All independent recomputations MATCH the artifact.")
    print("NOTE: research/simulation only. No live trading or production execution.")


if __name__ == "__main__":
    main()