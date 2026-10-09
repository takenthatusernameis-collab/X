"""Independent verification of the momentum cost check (`research/checks/momentum_cost.py`).

This is a separate implementation path: it recomputes each segment's walk-forward
median under realistic costs from a fresh `walk_forward` implementation
(not `stress_segments_across_tickers`), then compares against the artifact
written by the momentum cost check (`state/check_artifacts/momentum_cost_results.json`).
A mismatch would flag a defect in the cost check.

Design of the independent path:
- Regime labels are reconstructed from `closes` using a re-implemented
  `volatility_blocks` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using `min_segment_bars`.
- Segment medians are computed via `walk_forward` directly on the segment
  slice, aggregating fold log returns -- no call to `stress_segments_across_tickers`
  or `regime_stability.py`.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments, cost config)
so a match confirms the check computed from those inputs; a mismatch would
flag an error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_results.json"

COST_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


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


def segment_median_cost(ticker, s, e, train, test, warm, overlap, lb, cost_cfg):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lb)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cost_cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(SEED)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == LOOKBACKS

    print("=== Independent recomputation under realistic costs ===")
    all_ok = True

    # Universe-level recomputation per lookback
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for lb in LOOKBACKS:
        print(f"\n=== Lookback {lb} ===")
        all_verdicts_match = True

        lb_str = str(lb)
        for ticker in artifact["per_lookback"][lb_str]["per_asset"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            medians = []
            for lab, s, e in runs:
                m, nf = segment_median_cost(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb, COST_CFG)
                medians.append(round(m, 3))

            pub_medians = artifact["per_lookback"][lb_str]["per_asset"][ticker]["medians"]
            pub_null_medians = artifact["per_lookback"][lb_str]["per_asset"][ticker]["null_medians"]
            pub_lbls = artifact["per_lookback"][lb_str]["per_asset"][ticker]["segments"]

            status = "MATCH" if medians == pub_medians and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  [DEBUG] ticker={ticker}: len(medians)={len(medians)} medians={medians}")
                print(f"  [DEBUG] ticker={ticker}: pub_medians={pub_medians}")

            # Regenerate verdict from recomputed medians and null medians
            cand = np.array(medians)
            null = np.array(pub_null_medians)
            cand_disp = round(float(np.std(cand)), 3)
            null_disp = round(float(np.std(null)), 3)
            if all(abs(m) <= 0.05 for m in medians):
                regen_verdict = "CONSISTENT_WITH_NOISE"
            elif cand_disp > 2.0 * null_disp:
                regen_verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                regen_verdict = "REGIME_STABLE_LOSS"
            else:
                regen_verdict = "REGIME_STABLE"

            verdict_status = "MATCH" if regen_verdict == artifact["per_lookback"][lb_str]["per_asset"][ticker]["verdict"] else "MISMATCH"
            if verdict_status != "MATCH":
                all_verdicts_match = False
                all_ok = False

            print(f"  {ticker}: medians {medians} segments={pub_lbls} -> {status} | verdict {regen_verdict} (pub: {artifact['per_lookback'][lb_str]['per_asset'][ticker]['verdict']})")

        print(f"    verdict counts for lookback {lb}:")
        print(f"      {artifact['per_lookback'][lb_str]['verdict_counts']}")

    print()

    # Determinism of this verification path
    print("Determinism: recomputing universe medians for one asset again.")
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)

    m1 = [round(segment_median_cost("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP, 5, COST_CFG)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median_cost("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP, 5, COST_CFG)[0], 3)
          for lab, s, e in runs]
    print(f"  AAPL r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations under realistic costs MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the cost check artifact does not reproduce via the independent path.")
        print("Investigate before accepting the results.")
        sys.exit(1)

    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost robustness independent verification on the collected universe: exploratory simulation.")


if __name__ == "__main__":
    main()
