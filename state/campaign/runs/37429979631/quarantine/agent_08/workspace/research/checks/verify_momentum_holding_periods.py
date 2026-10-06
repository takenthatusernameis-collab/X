"""Independent verification of the momentum holding-period check
(`research/checks/momentum_holding_periods.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh
`walk_forward` implementation (not ``stress_segments``), then compares against
the artifact written by the momentum holding-period check
(`state/check_artifacts/momentum_holding_periods_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- The matched coin-flip null is recomputed via a fresh RNG code path seeded to
  the check's deterministic convention, using the same segment lengths and
  walk-forward windows, so it can be compared against the artifact's nulls.
- All candidate values are loaded from the check's artifact and compared;
  verdicts are also regenerated from the recomputed medians and the artifact's
  matched coin-flip nulls.

This keeps the same inputs (dataset, seed, windows, segments, holds) so a
match confirms the check computed from those inputs; a mismatch would flag an
error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.data import Bar

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
HOLD_PERIODS = (1, 2, 5)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_holding_periods_results.json"


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


def coin_flip_null_hold(seg_closes, hold, seed):
    """Independent re-implementation of the matched coin-flip null at one
    hold. The null signals use the check's own weight convention
    (choice over {-1, 0, +1} with p=[1/3,1/3,1/3]) seeded by the same
    deterministic scheme (base seed 42 plus the parameter sum times 1000),
    run through walk_forward with the same windows and the same hold.
    Only the RNG generation code is a separate path from the check's
    random_signals."""
    rng = np.random.default_rng(seed)
    n = len(seg_closes)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1/3, 1/3, 1/3])
    signals = [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]
    seg_bars = [Bar(date=i + 1, open=float(c), high=float(c), low=float(c),
                    close=float(c), volume=0.0) for i, c in enumerate(seg_closes)]
    cfg = bt.BacktestConfig(hold_period=hold)
    res = bt.walk_forward(
        seg_bars, signals, train_window=TRAIN, test_window=TEST, warmup=0,
        overlap_window=0, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def verdict_from(medians, null):
    """Regenerate a regime-stability verdict from medians and a null (same
    logic as research.backtest.regime_stability)."""
    cand = np.array(medians)
    null = np.array(null)
    cand_disp = round(float(np.std(cand)), 3)
    null_disp = round(float(np.std(null)), 3)
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK, "lookback mismatch"
    assert artifact["holds"] == list(HOLD_PERIODS), "hold grid mismatch"
    expected_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]
    assert sorted(artifact["universe"]["per_asset"].keys()) == sorted(expected_tickers)

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset (AMZN/JPM): momentum segment medians and matched null medians
    # for each hold, recomputed independently, vs artifact.
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        hold_medians = []
        hold_nulls = []
        hold_nfolds = []
        for h in HOLD_PERIODS:
            medians = []
            nulls = []
            nfolds = []
            for lab, s, e in runs:
                seg_signals = momentum_signals(closes, LOOKBACK)[s:e]
                cfg = bt.BacktestConfig(hold_period=h, warmup_periods=WARM)
                res = bt.walk_forward(
                    list(bars)[s:e], seg_signals, train_window=TRAIN,
                    test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
                    cfg=cfg)
                log = np.array([np.log1p(np.clip(f.metrics["total_return"],
                                -1.0 + 1e-12, None))
                                for f in res.folds], dtype=np.float64)
                medians.append(round(float(np.median(log)), 3))
                nfolds.append(len(res.folds))
                # null seeded by the check's deterministic scheme for params
                # {"lookback": 5} -> 42 + 5000 = 5042
                null_med, _ = coin_flip_null_hold(
                    closes[s:e], hold=h, seed=42 + int(round(5.0)) * 1000)
                nulls.append(round(null_med, 3))
            hold_medians.append((lbls, medians, nfolds))
            hold_nulls.append((lbls, nulls))

        for idx, h in enumerate(HOLD_PERIODS):
            lbls, medians, nfolds = hold_medians[idx]
            _, nulls = hold_nulls[idx]
            pub = artifact["per_asset"][asset][str(h)]
            pub_n = pub["n_folds"]

            med_status = "MATCH" if medians == pub["medians"] else "MISMATCH"
            null_status = "MATCH" if nulls == pub["null_medians"] else "MISMATCH"
            nf_status = "MATCH" if nfolds[0] == pub_n else "MISMATCH"
            if med_status != "MATCH" or null_status != "MATCH" or nf_status != "MATCH":
                all_ok = False
            print("  {}: hold={} medians {} vs pub {} -> {}"
                  .format(asset, h, medians, pub["medians"], med_status))
            print("       nulls    {} vs pub {} -> {}"
                  .format(nulls, pub["null_medians"], null_status))
            print("       n_folds  {} vs pub {} -> {}"
                  .format(nfolds, pub_n, nf_status))
            # Regenerate the verdict from recomputed medians + artifact nulls.
            regen_verdict = verdict_from(medians, pub["null_medians"])
            verdict_status = "MATCH" if regen_verdict == pub["verdict"] else "MISMATCH"
            if verdict_status != "MATCH":
                all_ok = False
            print("       verdict  {} (pub: {}) -> {}".format(
                regen_verdict, pub["verdict"], verdict_status))

    # Universe-level: fresh walk_forward medians for every ticker at every hold,
    # compared against the artifact, with verdicts regenerated from
    # recomputed medians + artifact nulls.
    print("\n=== Universe-level recomputation (all 10 tickers, holds 1/2/5) ===")
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {}
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for ticker in artifact["universe"]["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        results = {}
        for h in HOLD_PERIODS:
            medians = []
            unrounded = []
            for lab, s, e in runs:
                seg_signals = momentum_signals(closes, LOOKBACK)[s:e]
                res = bt.walk_forward(
                    list(bars)[s:e], seg_signals, train_window=TRAIN,
                    test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
                    cfg=bt.BacktestConfig(hold_period=h, warmup_periods=WARM))
                log = np.array([np.log1p(np.clip(f.metrics["total_return"],
                                -1.0 + 1e-12, None))
                                for f in res.folds], dtype=np.float64)
                med = float(np.median(log))
                medians.append(round(med, 3))
                unrounded.append(med)
            results[h] = (lbls, medians, unrounded)

        for h in HOLD_PERIODS:
            lbls, medians, unrounded = results[h]
            pub = artifact["per_asset"][ticker][str(h)]
            pub_lbls = pub["segments"]
            status = "MATCH" if medians == pub["medians"] else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print("  [DEBUG] ticker={}: pub_lbls={} re-lbls={}".format(
                    ticker, pub_lbls, lbls))
            regen_verdict = verdict_from(unrounded, pub["null_medians"])
            verdict_status = ("MATCH" if regen_verdict
                              == pub["verdict"]
                              else "MISMATCH")
            if verdict_status != "MATCH":
                all_ok = False
            print("  {}: hold={} medians={} segments={} -> {} | "
                  "verdict {} (pub: {}) -> {}".format(
                ticker, h, medians, lbls, status, regen_verdict,
                pub["verdict"],
                verdict_status))

    # Determinism of this verification path: recompute AMZN medians at all
    # holds twice and assert identical output.
    print("\nDeterminism: recompute AMZN medians for all holds (r1 vs r2).")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    r1 = {}
    r2 = {}
    for h in HOLD_PERIODS:
        m1 = []
        m2 = []
        for lab, s, e in runs:
            seg_signals = momentum_signals(closes, LOOKBACK)[s:e]
            res1 = bt.walk_forward(
                list(bars)[s:e], seg_signals, train_window=TRAIN,
                test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
                cfg=bt.BacktestConfig(hold_period=h, warmup_periods=WARM))
            log1 = np.array([np.log1p(np.clip(f.metrics["total_return"],
                            -1.0 + 1e-12, None))
                            for f in res1.folds], dtype=np.float64)
            m1.append(round(float(np.median(log1)), 3))
            res2 = bt.walk_forward(
                list(bars)[s:e], seg_signals, train_window=TRAIN,
                test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
                cfg=bt.BacktestConfig(hold_period=h, warmup_periods=WARM))
            log2 = np.array([np.log1p(np.clip(f.metrics["total_return"],
                            -1.0 + 1e-12, None))
                            for f in res2.folds], dtype=np.float64)
            m2.append(round(float(np.median(log2)), 3))
        r1[h] = m1
        r2[h] = m2
    for h in HOLD_PERIODS:
        ident = "identical" if r1[h] == r2[h] else "DIFFERENT"
        print("  hold={} r1: {}  r2: {} -> {}".format(h, r1[h], r2[h], ident))
        if ident != "identical":
            all_ok = False
            print("  [DEBUG] verification path not deterministic at hold={}".format(h))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum holding-period independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
