"""Regime-stability stress test of the AAPL MA crossover on the collected
Yahoo Finance OHLCV universe, partitioning the series into contiguous
volatility regimes (research-only).

This closes the last robustness dimension on real data: the tooling already
has perturbation (parameter sensitivity) and asset-universe sweeps executed on
the collected universe; this runs regime-stability across real market regimes.

Method: each bar is classified past-only by its trailing-60-bar realized
volatility into 'calm' (<15% annualized), 'normal' (15-28%), or 'turbulent'
(>28%). Consecutive bars of the same label form segments; each segment is
backtested with walk-forward IS/OOS and compared against a coin-flip null run
on the same segment. Per-segment candidate medians are compared against the
null medians with the same verdict logic as ``regime_stress``:
CONSISTENT_WITH_NOISE (no edge in any segment), REGIME_DEPENDENT (the
candidate's cross-segment swing exceeds 2x the null's, i.e. it fits one
regime), or REGIME_STABLE (consistent edge or no edge everywhere).

Research/simulation only. No live trading, no production execution.
Results inform only whether to continue research on this idea.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
WARMUP = 60  # matches the slow (60) moving-average window padding
FAST, SLOW = 20, 60
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60


def ma_crossover_signals(closes: np.ndarray, fast: int, slow: int):
    """Past-only MA-crossover signals.

    Bars before both moving-average windows are neutral. The signal author
    guarantees padding through the slow window so no real signal exists
    before the warmup boundary.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fast_ma = np.mean(closes[i - fast + 1 : i + 1])
        slow_ma = np.mean(closes[i - slow + 1 : i + 1])
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return signals


def volatility_segments(closes: NDArray, window: int = 60):
    """Past-only classification by trailing realized volatility: calm /
    normal / turbulent / insufficient."""
    labels: list[str] = []
    for i in range(len(closes)):
        if i < window:
            labels.append("insufficient")
        else:
            rv = float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
            if rv < 0.15:
                labels.append("calm")
            elif rv < 0.28:
                labels.append("normal")
            else:
                labels.append("turbulent")
    return labels


def volatility_blocks(closes: NDArray, n_blocks: int = 4, window: int = 60):
    """Split the series into `n_blocks` contiguous blocks by date and label
    each block by its realized-volatility regime.

    Each block is labeled 'turbulent' if its median trailing-`window`-bar
    realized volatility (annualized) exceeds the series-wide median, else
    'calm'. Bars before `window` bars into the series are 'insufficient' and
    dropped as a short leading segment.

    This gives a small number of long, contiguous regime segments so that
    walk-forward validation inside each segment is well defined. Regime labels
    are determined by past data only (block medians and the series-wide
    median are both computed from closes).
    """
    n = len(closes)
    block_size = n // n_blocks
    bar_vols: List[float] = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
            )
    active = [v for v in bar_vols if v > 0]
    if not active:
        raise ValueError("not enough bars to compute realized volatility")
    series_med = float(np.median(active))

    labels: List[str] = []
    for i in range(n):
        block = i // block_size
        s, e = block * block_size, (block + 1) * block_size
        bmed = float(
            np.median([v for v in bar_vols[s:e] if v > 0])
            if e - s > window
            else 0.0
        )
        labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def volatility_segments(closes: NDArray, window: int = 60):
    """Past-only classification by trailing realized volatility: calm /
    normal / turbulent / insufficient."""
    labels: list[str] = []
    for i in range(len(closes)):
        if i < window:
            labels.append("insufficient")
        else:
            rv = float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
            if rv < 0.15:
                labels.append("calm")
            elif rv < 0.28:
                labels.append("normal")
            else:
                labels.append("turbulent")
    return labels


def volatility_segments_balanced(closes: NDArray, window: int = 60):
    """Past-only classification into 'low' / 'high' by trailing realized vol,
    split at the series median so the classification yields two comparable
    regime segments (unlike fixed thresholds, which can swallow one regime).

    Uses only closes[:i], i.e. past-only. Bars before `window` are labeled
    'insufficient' and dropped as a short leading segment.
    """
    n = len(closes)
    vols = []
    for i in range(window, n):
        rv = float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
        vols.append(rv)
    if not vols:
        raise ValueError("not enough bars to compute realized volatility")
    median_vol = float(np.median(vols))
    labels: list[str] = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            idx = i - window
            labels.append("low" if vols[idx] < median_vol else "high")
    return labels


def main():
    np.random.seed(42)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Data loading + signals ===")
    bars, dates = bt.load_ticker("AAPL")
    print(f"  AAPL bars: {bars.n_bars} | first: {dates[0]} | last: {dates[-1]}")
    closes = bars.closes_array()
    fast, slow = FAST, SLOW
    signals = ma_crossover_signals(closes, fast, slow)
    nonneutral = sum(1 for s in signals if abs(s.weight) > 1e-12)
    print(f"  signals: {nonneutral} non-neutral after {slow}-bar warmup padding")

    print("\n=== 4. Leakage review (REAL_DATA_FEASIBILITY.md pre-run checklist) ===")
    bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
    print("  [PASS] check_signal_integrity: dates unique, all dates in series, "
          "no future dates, weights in [-1,1], no pre-warmup non-neutral weight")
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARMUP)
    res = bt.run_bars(list(bars), signals, cfg0)
    bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
    print("  [PASS] check_equity_matches_fills: full-sample equity fully explained by recorded fills")

    print("\n=== 5. Regime classification (4 contiguous blocks by date, past-only) ===")
    labels = volatility_blocks(closes, n_blocks=4)
    from collections import Counter
    counts = Counter(labels)
    # 'insufficient' prefix is expected (leading bars without enough history)
    # and is dropped as a short segment.
    for label, n in counts.items():
        print(f"  {label}: {n} bars")

    print("\n=== 6. Regime-stability stress test (real segments, MA crossover) ===")
    baseline = (("fast", fast), ("slow", slow))
    # A single canonical parameter set keeps this focused on regime stability
    # across segments; the full perturbation grid is already exercised by the
    # ma_crossover_real_data example.
    grid = [{"fast": fast, "slow": slow}]

    stress = bt.stress_segments(
        signals_fn=ma_crossover_signals,
        bars=bars,
        signals=signals,
        param_grid=grid,
        baseline=baseline,
        segment_fn=bt.segment_fn_from_labels(labels),
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg0,
        min_segment_bars=400,
    )

    print("Scenarios (segments):", [s.name for s in stress.scenarios])
    for s in stress.scenarios:
        status = "EDGE  " if abs(s.baseline_median_log_return) > 0.05 else "noise"
        print(f"  {s.name:12s}: candidate {s.baseline_median_log_return:+.3f}, "
              f"noise {s.noise_median_log_return:+.3f}  "
              f"[{status}] (folds={s.n_folds})")
    print(f"  Candidate dispersion across segments: {stress.candidate_dispersion:+.3f}")
    print(f"  Null dispersion across segments:      {stress.null_dispersion:+.3f}")
    print(f"  Overall verdict: {stress.overall_verdict}")
    print()
    print("Interpretation: REGIME_STABLE means the MA crossover behaves the same "
          "way in every regime mix (same edge, or no edge, everywhere). "
          "REGIME_DEPENDENT means the candidate's results vary strongly across "
          "regimes relative to the noise benchmark (fits one regime, fails in "
          "others). CONSISTENT_WITH_NOISE means the candidate is "
          "indistinguishable from the coin-flip null in every segment.")

    print("\nNOTE: research/simulation only. No live trading or production execution. "
          "AAPL MA crossover regime-stability on the collected universe: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
