"""Test momentum lookback-5 across a predeclared small holding-period grid.

This is a focused-task (R-002) as per the higher-order objective: "Test a fixed
lookback-5 momentum signal with a predeclared small holding-period set using the
existing walk-forward/null methodology."

The holding-period grid is predeclared: [1, 2, 3, 5, 7] days.

Primary question: Does the qualified momentum effect survive a small
predeclared holding-period variation without becoming a single-point timing
artifact?

The momentum cell from activation 37318950814 (lookback-5, 1-day hold) was
SUPPORTED with a REGIME_STABLE positive edge in 7/10 assets. This test
extends that analysis to a holding-period grid.

Methodology:
- Use the existing walk-forward regime-stability gate from regime_stability.py
- Same 10-asset collected universe (yf-ohlcv-universe-2009-to-2026-10-03)
- Same walk-forward parameters: train=252d, test=84d, warmup=60d, overlap=60d, min 400 bars/segment
- Fixed lookback = 5 days (NO POST-INSPECT RESOLUTION)
- Variable holding periods: [1, 2, 3, 5, 7] days (predeclared before execution)
- Same volatility-blocks regime classification (n_blocks=4, window=60)
- Coin-flip null on each segment (sign-flipped signal)
- Both per-asset verdict counts and universe-level verdict
- Independent verification path
- Deterministic assertion

Falsification prediction: momentum either shows CONSISTENT_WITH_NOISE across
holding periods (no edge in any holding period) or REGIME_DEPENDENT (edge
squanders on one holding period, revealing a single-point timing artifact),
or REGIME_STABLE_LOSS (uniformly negative, reversal wins on both sides).

Success criterion: The holding-period sensitivity is classified and any
positive finding is reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLDING_PERIODS = [1, 2, 3, 5, 7]  # PREDECLARED BEFORE EXECUTION
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "holding_period_momentum_results.json"

def create_momentum_signals(closes, lookback, hold_days):
    """Generate momentum signals with variable holding periods.
    
    Long the previous lookback-day return and hold for hold_days.
    At each bar, if the historical return was positive, go long;
    if negative, go short. The position is held for hold_days
    from the bar where the signal was generated.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        weight = 1.0 if ret > 0 else -1.0
        # Hold the position for hold_days
        for j in range(i, min(i + hold_days, n)):
            out[j] = bt.Signal(date=j + 1, weight=weight)
    return out

def run_holding_period_test(asset, tickers):
    """Test momentum with all holding periods for one asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    results = {}
    
    for hold_days in HOLDING_PERIODS:
        signals = create_momentum_signals(closes, LOOKBACK, hold_days)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} hold={hold_days}: signals {len(signals)} != bars {len(closes)}")
        
        res = bt.stress_segments(
            signals_fn=lambda c, hd=hold_days: create_momentum_signals(c, LOOKBACK, hd),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK, "hold_days": hold_days}],
            baseline=(("lookback", LOOKBACK), ("hold_days", hold_days)),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        
        results[f"hold_{hold_days}"] = dict(
            segments=[s.name for s in res.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=res.overall_verdict,
            n_folds=res.n_folds,
        )
    
    return results

def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    
    print("\n=== 3. Leakage review per asset ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        
        for hold_days in HOLDING_PERIODS:
            signals = create_momentum_signals(closes, LOOKBACK, hold_days)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        
        total_return = (bars.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, hold grid {HOLDING_PERIODS}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    
    print("\n=== 4. Momentum lookback-5 across holding-period grid (predeclared) ===")
    print("Holding periods: {}".format(HOLDING_PERIODS))
    print("Momentum: long the previous {}-day return, hold variable days, daily "
          "rebalanced. Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, compared vs coin-flip null on the same segments\n".format(
          LOOKBACK, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    
    results_all = {}
    for asset in target:
        res = run_holding_period_test(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:10s} {:14s} {:28s} {:12s}  {:12s}  verdict".format(
            "hold", "segments", "candidate_medians", "cand_disp", "null_disp"))
        for hold_days, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ", ".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:10d} {:14s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    int(hold_days[5:]), segs, meds, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
        print("")
    
    # Full-universe generalization
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    
    print("\n=== 5. Momentum lookback-5 across collected universe (all holding periods) ===")
    summary_all = {}
    for hold_days in HOLDING_PERIODS:
        univ_summary = bt.stress_segments_across_tickers(
            tickers=tickers_univ,
            signals_fn=lambda c, hd=hold_days: create_momentum_signals(c, LOOKBACK, hd),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", LOOKBACK), ("hold_days", hold_days)),
            param_grid=[{"lookback": LOOKBACK, "hold_days": hold_days}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        summary_all[f"hold_{hold_days}"] = univ_summary
        print("Hold {} days: {}".format(hold_days, univ_summary.inspect().replace("\n", "\n  ")))
    
    print("\n=== 6. Determinism: rerun the 2-asset comparison and assert identical output ===")
    res2 = {a: run_holding_period_test(a, tickers) for a in target}
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"
    
    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        holding_periods=HOLDING_PERIODS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=results_all,
        universe=dict(
            summaries=summary_all,
            overall_verdict_counts={},
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("Holding-period sweep complete.")
    return 0

def sha256_file(path):
    """Simple sha256 for file verification."""
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

if __name__ == "__main__":
    sys.exit(main())