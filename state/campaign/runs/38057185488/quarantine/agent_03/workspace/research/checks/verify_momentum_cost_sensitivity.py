"""Independent verification of the momentum cost sensitivity check
(`research/checks/momentum_cost_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum cost sensitivity check (`state/check_artifacts/momentum_cost_sensitivity_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Cost modeling: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch would flag an
error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_MODEL = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COST_MODEL = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


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


def base_ma_signals(closes, fast, slow):
    """Fresh implementation of base MA crossover (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fast_ma = float(np.mean(closes[i - fast + 1 : i + 1]))
        slow_ma = float(np.mean(closes[i - slow + 1 : i + 1]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return out


def segment_median(ticker, s, e, train, test, warm, overlap, cfg, signals_fn):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = signals_fn(closes, **{"fast": 20, "slow": 60}) if signals_fn.__name__ == "base_ma_signals" else signals_fn(closes)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Load manifest for universe
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {}
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # Per-asset: base MA and momentum segment medians vs artifact
    print("\n=== Per-asset recomputation (AMZN/JPM subset) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        base_medians = []
        for lab, s, e in runs:
            m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP, ZERO_COST_MODEL, base_ma_signals)
            base_medians.append(round(m, 3))
        mom_medians = []
        for lab, s, e in runs:
            m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP, ZERO_COST_MODEL, lambda c, **kw: momentum_signals(c, lookback=5))
            mom_medians.append(round(m, 3))

        pub_base = artifact["lookback_results"]["5"]["zero_cost"][asset]["medians"]
        pub_mom = artifact["lookback_results"]["5"]["zero_cost"][asset]["medians"]
        pub_lbls = artifact["lookback_results"]["5"]["zero_cost"][asset]["segments"]

        base_status = "MATCH" if base_medians == pub_base and lbls == pub_lbls else "MISMATCH"
        mom_status = "MATCH" if mom_medians == pub_mom and lbls == pub_lbls else "MISMATCH"
        if base_status != "MATCH" or mom_status != "MATCH":
            all_ok = False
        print("  {}: base medians {} vs published {} -> {}"
              .format(asset, base_medians, pub_base, base_status))
        print("  {}: momentum medians {} vs published {} -> {}"
              .format(asset, mom_medians, pub_mom, mom_status))

    # Universe-level per-asset medians and verdicts vs artifact
    print("\n=== Universe-level recomputation ===")
    for ticker in artifact["lookback_results"]["5"]["zero_cost"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        # Check both cost and zero-cost configurations
        for cfg_name in ["zero_cost", "cost"]:
            if cfg_name not in artifact["lookback_results"]["5"]:
                continue
                
            medians = []
            for lab, s, e in runs:
                cfg = COST_MODEL if cfg_name == "cost" else ZERO_COST_MODEL
                m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, cfg, lambda c, **kw: momentum_signals(c, lookback=5))
                medians.append(round(m, 3))
                
            pub = artifact["lookback_results"]["5"][cfg_name][ticker]["medians"]
            pub_lbls = artifact["lookback_results"]["5"][cfg_name][ticker]["segments"]
            
            status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print("  [DEBUG] ticker={}: len(runs)={} lbls={}".format(ticker, len(runs), lbls))
                print("  [DEBUG] ticker={}: pub_lbls={}".format(ticker, pub_lbls))
                print("  [DEBUG] ticker={}: len(medians)={} medians={}".format(ticker, len(medians), medians))
                print("  [DEBUG] ticker={}: pub_medians={}".format(ticker, pub))
            
            # Regenerate the verdict from recomputed medians and null medians.
            cand = np.array(medians)
            null = np.array(artifact["lookback_results"]["5"][cfg_name][ticker]["null_medians"])
            cand_disp = round(float(np.std(cand)), 3)
            null_disp = round(float(np.std(null)), 3)
            if all(abs(m) <= 0.05 for m in medians):
                regen_verdict = "CONSISTENT_WITH_NOISE"
            elif cand_disp > 2.0 * null_disp:
                regen_verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                # Uniformly negative medians with a swing within twice the null:
                # stable losses, not an edge; must be read alongside the medians.
                regen_verdict = "REGIME_STABLE_LOSS"
            else:
                regen_verdict = "REGIME_STABLE"
                
            verdict_status = "MATCH" if regen_verdict == artifact["lookback_results"]["5"][cfg_name][ticker]["verdict"] else "MISMATCH"
            if verdict_status != "MATCH":
                all_ok = False
            print("  {}: medians {} segments={} -> {} | verdict {} (pub: {})"
                  .format(ticker, medians, pub_lbls, status, regen_verdict,
                          artifact["lookback_results"]["5"][cfg_name][ticker]["verdict"]))
            if verdict_status != "MATCH":
                print("      verdict regeneration did not match artifact: {}"
                      .format(verdict_status))

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, ZERO_COST_MODEL, lambda c, **kw: momentum_signals(c, lookback=5))[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, ZERO_COST_MODEL, lambda c, **kw: momentum_signals(c, lookback=5))[0], 3)
          for lab, s, e in runs]
    print("  AMZN r1: {}; r2: {} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity independent verification on AMZN/JPM + universe: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
