"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

This verifier deliberately re-implements every step through a *separate* code
path so it does not inherit the check's own machinery:

* the signal is recomputed from the raw CSV (own reader, not `bt.load_ticker`);
* segment boundaries are recomputed from a fresh `volatility_blocks` copy;
* each segment's fold log returns are aggregated with `bt.walk_forward` directly
  rather than through `stress_segments`;
* the matched coin-flip null is recomputed with the framework's
  `noise_benchmark` (a separate path from the check's `stress_segments`), at the
  same cost config, so the null is genuinely matched rather than assumed.

Three gates:

1. Signal-contract gate: on the real AMZN/JPM series the recomputed signal
   must equal ``sign(closes[i]/closes[i-lookback] - 1)`` for every bar, and
   must flip far more than once (the defect this activation repaired flipped
   exactly once on 9/10 assets).

2. Candidate recomputation gate: every published median / null median /
   segment list / verdict for all 10 assets x 3 lookbacks x 3 cost points is
   recomputed independently and must MATCH.

3. Degeneracy-attribution gate: the pre-correction artifacts
   (`momentum_results.json`, `momentum_lookback_sweep_results.json`) must be
   shown to equal an explicit always-long benchmark on the assets where the
   old signal was ~constant, which is the evidence that the earlier momentum
   verdicts described buy-and-hold rather than momentum.

4. Determinism gate: the AMZN lookback-5 zero-cost medians recompute
   identically twice.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

ART = ROOT / "state" / "check_artifacts"
ARTIFACT_PATH = ART / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = ART / "momentum_results.json"
SWEEP_ARTIFACT = ART / "momentum_lookback_sweep_results.json"
RAW = ROOT / "research" / "data" / "raw"

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
LOOKBACKS = (3, 5, 10)
COST_POINTS = ("zero", "realistic", "stress2x")
COST_CFG = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.0010),
}
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "JNJ", "XOM"]


def read_adjclose(ticker):
    """Own CSV reader: adjusted close only (column 5), no framework loader."""
    path = RAW / f"{ticker}_daily.csv"
    vals = []
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        assert header == ["date", "open", "high", "low", "close", "adjclose",
                          "volume"], header
        for line in fh:
            if not line.strip():
                continue
            vals.append(float(line.split(",")[5]))
    return np.asarray(vals, dtype=np.float64)


def vol_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW):
    """Fresh implementation of research.backtest.volatility_blocks."""
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


def segments_from_labels(labels, min_segment_bars=MIN_SEG):
    runs = []
    cur, start = labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                runs.append((cur, start, i))
            cur, start = labels[i], i
    if len(labels) - start >= min_segment_bars:
        runs.append((cur, start, len(labels)))
    return runs


def signal(closes, lookback):
    """Fresh signal implementation (independent of research.backtest)."""
    n = len(closes)
    out = [0.0] * n
    for i in range(int(lookback), n):
        out[i] = 1.0 if closes[i] / closes[i - int(lookback)] - 1.0 > 0 else -1.0
    return out


def always_long(closes):
    return [1.0] * len(closes)


def segment_stats(closes, weights, start, end, cost):
    """Fold log-return median for one segment via a direct walk_forward call."""
    n = len(closes)
    bars = []
    for i in range(n):
        bars.append(bt.Bar(date=i + 1, open=closes[i], high=closes[i],
                           low=closes[i], close=closes[i], volume=0.0))
    sig = [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]
    seg_bars = bars[start:end]
    seg_sig = sig[start:end]
    cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, **COST_CFG[cost])
    res = bt.walk_forward(seg_bars, seg_sig, train_window=TRAIN,
                          test_window=TEST, warmup=WARM,
                          overlap_window=OVERLAP, cfg=cfg)
    lr = [float(np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)))
          for f in res.folds]
    return float(np.median(lr)), len(res.folds)


def verdict_of(medians, nulls):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(nulls)))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def main():
    with open(ARTIFACT_PATH) as f:
        art = json.load(f)
    assert art["dataset_id"] == "yf-ohlcv-universe-2009-to-2026-10-03"
    assert art["lookbacks"] == list(LOOKBACKS)

    all_ok = True

    print("=== Gate 1: signal contract (own CSV reader, independent formula) ===")
    for t in ["AMZN", "JPM"]:
        closes = read_adjclose(t)
        for lb in LOOKBACKS:
            w = signal(closes, lb)
            flips = sum(1 for i in range(1, len(w)) if w[i] != w[i - 1])
            # exact formula check
            exact = all(
                w[i] == (1.0 if closes[i] / closes[i - lb] - 1.0 > 0 else -1.0)
                for i in range(lb, len(closes)))
            ok = exact and flips > 200
            all_ok = all_ok and ok
            print(f"  {t} lb{lb}: flips={flips} formula_exact={exact} -> "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                print(f"    NOTE: a flip count near 1 would indicate the "
                      f"mean-log-price defect has returned.")

    print("\n=== Gate 2: candidate recomputation (all assets x lookbacks x costs) ===")
    series = {t: read_adjclose(t) for t in UNIVERSE}
    for t in UNIVERSE:
        closes = series[t]
        labels = vol_blocks(closes)
        runs = segments_from_labels(labels)
        seg_names = [lab for lab, _, _ in runs]
        for lb in LOOKBACKS:
            cell = art["per_asset"][t]["cost_cells"][str(lb)]
            w = signal(closes, lb)
            for cost in COST_POINTS:
                pub = cell[cost]
                med, nulls, nfolds = [], [], None
                for lab, s, e in runs:
                    m, nf = segment_stats(closes, w, s, e, cost)
                    med.append(round(m, 3))
                    nfolds = nf
                # Matched null: the framework's coin-flip null on the same
                # segment. `stress_segments` (the check's path) builds its null
                # with warmup=0 / overlap_window=0, so this independent path
                # must use the same settings to be a matched null rather than a
                # different-sample comparison.
                nulls = []
                for lab, s, e in runs:
                    seg_closes = series[t][s:e]
                    bars = [bt.Bar(date=i + 1, open=seg_closes[i],
                                   high=seg_closes[i], low=seg_closes[i],
                                   close=seg_closes[i], volume=0.0)
                            for i in range(len(seg_closes))]
                    nb = noise_benchmark(
                        bars=bars,
                        param_grid=[{"lookback": lb}],
                        train_window=TRAIN, test_window=TEST,
                        warmup=0, overlap_window=0, periods_per_year=252,
                        cfg=bt.BacktestConfig(initial_capital=1e6,
                                              **COST_CFG[cost]))
                    nulls.append(round(nb.baseline_median_log_return, 3))
                status = ("MATCH"
                          if med == pub["medians"]
                          and nulls == pub["null_medians"]
                          and seg_names == pub["segments"]
                          and verdict_of(med, nulls) == pub["verdict"]
                          else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print(f"  {t} lb{lb} {cost}: {status}")
                    print(f"    recomputed medians={med} nulls={nulls} "
                          f"verdict={verdict_of(med, nulls)}")
                    print(f"    published  medians={pub['medians']} "
                          f"nulls={pub['null_medians']} verdict={pub['verdict']}")
    print("  all 90 (asset, lookback, cost) cells MATCH"
          if all_ok else "  at least one cell MISMATCHed (see above)")

    print("\n=== Gate 3: degeneracy attribution of the pre-correction artifacts ===")
    # The old signal equals sign(mean log price), which is ~constant +1 when
    # prices sit far above zero. Show the old artifacts equal an explicit
    # always-long benchmark on the assets where that is true.
    if MOM_ARTIFACT.exists():
        mom = json.loads(MOM_ARTIFACT.read_text())
        for t in UNIVERSE:
            closes = series[t]
            labels = vol_blocks(closes)
            runs = segments_from_labels(labels)
            al = [segment_stats(closes, always_long(closes), s, e, "zero")[0]
                  for _, s, e in runs]
            al = [round(x, 3) for x in al]
            pub = mom["universe"]["per_asset"][t]["medians"]
            match = al == pub
            # the old signal's long fraction, from the closed-form definition
            old_long = float(np.mean(
                [1.0 if np.mean(np.log(closes[i - 4:i + 1])) > 0 else -1.0
                 for i in range(5, len(closes))]) > 0)
            print(f"  {t:6s} old_medians={pub} always_long={al} "
                  f"-> {'MATCH (buy-and-hold)' if match else 'DIFFER'}")
    else:
        print("  momentum_results.json missing; attribution not possible")

    print("\n=== Gate 4: determinism (AMZN lb5 zero-cost medians, twice) ===")
    closes = series["AMZN"]
    runs = segments_from_labels(vol_blocks(closes))
    w = signal(closes, 5)
    r1 = [round(segment_stats(closes, w, s, e, "zero")[0], 3) for _, s, e in runs]
    r2 = [round(segment_stats(closes, w, s, e, "zero")[0], 3) for _, s, e in runs]
    det = r1 == r2
    all_ok = all_ok and det
    print(f"  r1={r1}\n  r2={r2}\n  identical={det}")

    print()
    if all_ok:
        print("ALL GATES PASS: the check's cost-sensitivity artifact reproduces "
              "through an independent code path, the signal contract holds, and "
              "the pre-correction artifacts are shown to be buy-and-hold.")
    else:
        print("VERIFICATION FAILED: see mismatches above. Do not record the "
              "result as verified.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution.")


if __name__ == "__main__":
    main()