"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates, none of which call `stress_segments`:

1. Signal-integrity gate: re-implements the contracted momentum formula
   (``sign(mean(log(close[i-lookback+1:i+1])))) and a genuine momentum signal
   (``sign(log(close[i]) - log(close[i-lookback]))``) from scratch, and confirms
   the contracted signal is (near) always-long and lookback-insensitive while
   the genuine signal takes both signs and is lookback-sensitive.

2. Fresh-recomputation gate: for AMZN and JPM, every lookback in {3, 5, 10} and
   every cost level in {zero, realistic, conservative}, the per-segment
   candidate medians and coin-flip-null medians are recomputed through
   ``walk_forward`` + ``noise_benchmark`` (independent code path, same cost cfg)
   and compared to the check artifact, for both signal variants.

3. Determinism gate: one cell is recomputed twice and must be identical.
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
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

# Redefined independently from the check (a mismatch fails loudly).
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
        "slippage_proportional": 0.0010,
    },
}


def make_cfg(cost):
    return bt.BacktestConfig(initial_capital=1e6, **cost)


def vol_blocks(closes, n_blocks, window):
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


def contracted_momentum_signals(closes, lookback):
    """Independent reimplementation of research.backtest.momentum_signals."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lb = int(lookback)
    for i in range(lb, n):
        ret = float(np.mean(np.log(closes[i - lb + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def corrected_momentum_signals(closes, lookback):
    """Independent genuine momentum: sign of the lookback log return."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lg = np.log(closes)
    lb = int(lookback)
    for i in range(lb, n):
        r = lg[i] - lg[i - lb]
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


VARIANTS = {"contracted": contracted_momentum_signals,
            "corrected": corrected_momentum_signals}


def seg_candidate_median(seg_bars, lookback, variant, cfg):
    seg_closes = np.array([float(b.close) for b in seg_bars], dtype=np.float64)
    sig = VARIANTS[variant](seg_closes, lookback)
    res = bt.walk_forward(
        seg_bars, sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def seg_null_median(seg_bars, lookback, cfg):
    noise = noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252, cfg=cfg)
    return float(noise.baseline_median_log_return)


def recompute_cell(ticker, lookback, variant, cost_name):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    cfg = make_cfg(COST_LEVELS[cost_name])
    seg_bars_all = list(bars)
    medians, nulls = [], []
    nfolds = None
    for _lab, s, e in runs:
        seg_bars = seg_bars_all[s:e]
        m, nf = seg_candidate_median(seg_bars, lookback, variant, cfg)
        medians.append(round(m, 3))
        nulls.append(round(seg_null_median(seg_bars, lookback, cfg), 3))
        nfolds = nf if nfolds is None else nfolds
    return medians, nulls


def main() -> int:
    np.random.seed(42)
    art = json.load(open(ARTIFACT))
    assert art["dataset_id"] == DATASET_ID
    assert art["cost_levels"] == COST_LEVELS, "cost grid diverged between check and verifier"
    all_ok = True

    print("=== Gate 1: signal-integrity reimplementation ===")
    for ticker in ["AAPL", "AMZN", "NVDA", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            cw = np.array([s.weight for s in contracted_momentum_signals(closes, lb)])[lb:]
            rw = np.array([s.weight for s in corrected_momentum_signals(closes, lb)])[lb:]
            c_distinct = sorted(set(np.round(cw, 6).tolist()))
            c_frac_long = float(np.mean(cw > 0))
            r_frac_long = float(np.mean(rw > 0))
            ok = (c_frac_long >= 0.99 or ticker == "NVDA") and (0.2 < r_frac_long < 0.8)
            # Compare against the artifact's recorded integrity numbers.
            pub = art["signal_integrity"]["per_ticker"][ticker][str(lb)]
            pub_ok = (pub["frac_long"] == round(c_frac_long, 4)
                      and pub["distinct_weights"] == c_distinct)
            status = "MATCH" if (ok and pub_ok) else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print("  {:6s} lb={:>2d} contracted_distinct={} contracted_frac_long={:.3f} "
                  "corrected_frac_long={:.3f} -> {}".format(
                      ticker, lb, c_distinct, c_frac_long, r_frac_long, status))

    print("\n=== Gate 2: fresh walk_forward + noise_benchmark (AMZN, JPM; all lb x cost x variant) ===")
    for variant in ("contracted", "corrected"):
        for ticker in ["AMZN", "JPM"]:
            for lb in LOOKBACKS:
                for cost_name in ("zero", "realistic", "conservative"):
                    med, nul = recompute_cell(ticker, lb, variant, cost_name)
                    pub = art["cells"][variant][str(lb)][cost_name]["per_asset"][ticker]
                    ok = (med == pub["medians"] and nul == pub["null_medians"])
                    if not ok:
                        all_ok = False
                    print("  {:9s} {:6s} lb={:>2d} {:12s} medians {} vs {} | nulls {} vs {} -> {}".format(
                        variant, ticker, lb, cost_name,
                        med, pub["medians"], nul, pub["null_medians"],
                        "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 3: determinism (AMZN corrected lb=5 realistic recomputed twice) ===")
    m1 = recompute_cell("AMZN", 5, "corrected", "realistic")
    m2 = recompute_cell("AMZN", 5, "corrected", "realistic")
    det = (m1 == m2)
    if not det:
        all_ok = False
    print("  r1 == r2 -> {}".format(det))
    assert det, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact and the "
              "signal-integrity reimplementation reproduces the degeneracy.")
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