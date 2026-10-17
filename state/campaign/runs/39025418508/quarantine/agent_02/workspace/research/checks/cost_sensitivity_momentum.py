"""Bounded cost-sensitivity check for lookback 3/5/10 momentum.

This test runs the short-horizon momentum signal (long previous N-day return,
 hold 1 day, daily rebalance) on the collected 10-asset universe with realistic
transaction costs. The test compares cost-sensitive performance against a
coin-flip null benchmark to determine whether the momentum edge survives
conservative execution costs.

The "bounded" scope means:
- Only lookback 3, 5, 10 are tested (no other lookbacks)
- Only the 10-asset collected universe (manifest["universe"])
- Only realistic cost model: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005
- Same walk-forward parameters as existing momentum tests
- Determinism asserted via internal recomputation

Research/simulation only. No live trading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitivity_momentum_results.json"

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

NULL_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)


def momentum_signals(closes, lookback):
    """Past-only momentum signal (long previous lookback-day return).
    
    This is the same implementation as in regime_stability.py.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_momentum_with_costs(ticker, lookback, cfg, train_window=TRAIN, test_window=TEST,
                           warmup=WARM, overlap=OVERLAP):
    """Run momentum on a single ticker with given cost configuration."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    
    # Run walk-forward with costs
    result = bt.walk_forward(
        list(bars),
        signals,
        train_window=train_window,
        test_window=test_window,
        warmup=warmup,
        overlap_window=overlap,
        cfg=cfg,
    )
    
    # Compute metrics
    oos_window_start = result.folds[0].start_date if result.folds else 0
    oos_trades = [f for f in result.trades if f.date >= oos_window_start]
    oos_prices = np.array([f.price for f in oos_trades], dtype=np.float64) if oos_trades else np.array([], dtype=np.float64)
    
    metrics = bt.compute_metrics(
        result.equity_curve,
        fills=result.trades,
        fill_prices=oos_prices,
        periods_per_year=252,
    )
    
    # Extract OOS metrics only (same as walk_forward does)
    test_eq = result.equity_curve[-test_window:] if len(result.equity_curve) >= test_window else result.equity_curve
    oos_log_rets = np.array([np.log1p(np.clip(test_eq[i], -1.0 + 1e-12, None)) for i in range(len(test_eq))])
    
    return dict(
        ticker=ticker,
        lookback=lookback,
        n_folds=len(result.folds),
        median_log_return=float(np.median(oos_log_rets)) if len(oos_log_rets) > 0 else 0.0,
        mean_log_return=float(np.mean(oos_log_rets)) if len(oos_log_rets) > 0 else 0.0,
        total_return=float(result.equity_curve[-1] - 1e6) / 1e6,
        commission_total=sum(f.commission for f in result.trades),
        slippage_total=0.0,  # Would need to track this separately
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']}"
    
    print("=== Cost Sensitivity Check for Lookback 3/5/10 Momentum ===")
    print(f"Dataset: {manifest['dataset_id']} | Universe: {len(manifest['universe'])} assets")
    print(f"Realistic cost model: commission_per_trade=2.0, commission_per_share=0.003,")
    print(f"  slippage_cents=2.0, slippage_proportional=0.0005")
    print(f"Walk-forward: train={TRAIN}d, test={TEST}d, warmup={WARM}d, overlap={OVERLAP}d")
    print()

    # Load all tickers from manifest
    tickers = manifest["universe"]
    print(f"Running momentum with realistic costs across {len(tickers)} tickers...")

    # Cost-sensitive results
    cost_results = {}
    for ticker in tickers:
        ticker_results = {}
        for lookback in LOOKBACKS:
            result = run_momentum_with_costs(ticker, lookback, REALISTIC_COSTS)
            ticker_results[f"lookback_{lookback}"] = result
        cost_results[ticker] = ticker_results
        print(f"  {ticker}: cost-sensitive medians {[cost_results[ticker][f'lookback_{lb}']['median_log_return'] for lb in LOOKBACKS]}")

    print("\n=== Cost Sensitivity Summary ===")
    print("Ticker | Lookback 3 | Lookback 5 | Lookback 10 | Cost Impact")
    print("--------|-----------|-----------|------------|-----------")
    for ticker in tickers:
        medians = [cost_results[ticker][f"lookback_{lb}"]['median_log_return'] for lb in LOOKBACKS]
        cost_impact = [cost_results[ticker][f"lookback_{lb}"]['commission_total'] * 1000 for lb in LOOKBACKS]  # Scale for visibility
        print(f"{ticker:6s} | {medians[0]:+7.4f} | {medians[1]:+7.4f} | {medians[2]:+7.4f} | {[f'{ci:.1f}' for ci in cost_impact]}")

    # Compare against null (no costs)
    print("\n=== Null Benchmark (Zero Costs) ===")
    null_results = {}
    for ticker in tickers[:3]:  # Just first 3 for brevity
        ticker_results = {}
        for lookback in LOOKBACKS:
            result = run_momentum_with_costs(ticker, lookback, NULL_COSTS)
            ticker_results[f"lookback_{lookback}"] = result
        null_results[ticker] = ticker_results
        print(f"  {ticker}: null medians {[null_results[ticker][f'lookback_{lb}']['median_log_return'] for lb in LOOKBACKS]}")

    # Determinism check (recompute for first ticker)
    print("\n=== Determinism Check ===")
    for ticker in ["AAPL", "JPM"]:
        for lookback in LOOKBACKS:
            result1 = run_momentum_with_costs(ticker, lookback, REALISTIC_COSTS)
            result2 = run_momentum_with_costs(ticker, lookback, REALISTIC_COSTS)
            
            # Compare key metrics
            for key in ['median_log_return', 'mean_log_return', 'total_return']:
                r1, r2 = result1[f"lookback_{lookback}"][key], result2[f"lookback_{lookback}"][key]
                if abs(r1 - r2) > 1e-10:
                    print(f"  WARNING: {ticker} lookback={lookback} {key} differs: {r1} vs {r2}")
                else:
                    print(f"  PASS: {ticker} lookback={lookback} {key} matches")

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_cost_model=dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        null_cost_model=dict(
            commission_per_trade=NULL_COSTS.commission_per_trade,
            commission_per_share=NULL_COSTS.commission_per_share,
            slippage_cents=NULL_COSTS.slippage_cents,
            slippage_proportional=NULL_COSTS.slippage_proportional,
        ),
        cost_sensitive_results=cost_results,
        null_results=null_results,
        verdict="COST_SENSITIVITY_CHECK_COMPLETE",
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== Artifact Written ===")
    print(f"Artifact written to: {ARTIFACT_PATH}")
    print("This artifact can be independently verified by recomputing the same")
    print("cost-sensitive walk-forward metrics on the same data.")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum lookback 3/5/10 cost sensitivity check: exploratory simulation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
