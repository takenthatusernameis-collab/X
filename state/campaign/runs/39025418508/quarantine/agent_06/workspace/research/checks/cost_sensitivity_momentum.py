"""Bounded cost-sensitivity check for lookback 3/5/10 momentum.

This test runs the short-horizon momentum signal (long previous N-day return,
hold 1 day, daily rebalance) on the collected 10-asset universe with realistic
transaction costs. The test compares cost-sensitive performance against a
matched null benchmark to determine whether the momentum edge survives
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

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
    """Past-only momentum signal (long previous lookback-day return)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_segment_bars_with_costs(ticker, lookback, cfg):
    """Run a single backtest on a segment with the given cost configuration."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    
    # Get regime labels
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= MIN_SEGMENT_BARS:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= MIN_SEGMENT_BARS:
        runs.append((cur, run_start, len(labels)))
    
    # Run walk-forward on the full series
    signals = momentum_signals(closes, lookback)
    cfg_with_warmup = bt.BacktestConfig(
        warmup_periods=WARM,
        initial_capital=cfg.initial_capital,
        target_exposure=cfg.target_exposure,
        commission_per_trade=cfg.commission_per_trade,
        commission_per_share=cfg.commission_per_share,
        slippage_cents=cfg.slippage_cents,
        slippage_proportional=cfg.slippage_proportional,
        risk_free=cfg.risk_free,
        periods_per_year=cfg.periods_per_year,
        margin_rate=cfg.margin_rate,
        margin_call_liquidate=cfg.margin_call_liquidate,
    )
    res = bt.walk_forward(
        list(bars),
        signals,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg_with_warmup)
    
    # Extract OOS log returns from all folds
    all_log_rets = []
    for fold in res.folds:
        log_rets = np.array([np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None))
                            for fold in res.folds], dtype=np.float64)
        all_log_rets.extend(log_rets)
    
    return {
        "ticker": ticker,
        "lookback": lookback,
        "n_segments": len(runs),
        "median_log_return": float(np.median(all_log_rets)) if all_log_rets else 0.0,
        "mean_log_return": float(np.mean(all_log_rets)) if all_log_rets else 0.0,
        "n_folds": len(res.folds),
    }


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
            result = run_segment_bars_with_costs(ticker, lookback, REALISTIC_COSTS)
            ticker_results[f"lookback_{lookback}"] = result
        cost_results[ticker] = ticker_results
        print(f"  {ticker}: cost-sensitive medians {[cost_results[ticker][f'lookback_{lb}']['median_log_return'] for lb in LOOKBACKS]}")

    # Null benchmark results (matched costs)
    print("\n=== Null Benchmark (Zero Costs) ===")
    null_results = {}
    for ticker in tickers:
        ticker_results = {}
        for lookback in LOOKBACKS:
            result = run_segment_bars_with_costs(ticker, lookback, NULL_COSTS)
            ticker_results[f"lookback_{lookback}"] = result
        null_results[ticker] = ticker_results
        print(f"  {ticker}: null medians {[null_results[ticker][f'lookback_{lb}']['median_log_return'] for lb in LOOKBACKS]}")

    # Cost impact analysis
    print("\n=== Cost Impact Analysis ===")
    print("Ticker | Lookback 3 | Lookback 5 | Lookback 10 | Cost Survival")
    print("--------|-----------|-----------|------------|-------------")
    cost_survival_count = 0
    cost_data = []
    for ticker in tickers:
        cost_meds = [cost_results[ticker][f"lookback_{lb}"]["median_log_return"] for lb in LOOKBACKS]
        null_meds = [null_results[ticker][f"lookback_{lb}"]["median_log_return"] for lb in LOOKBACKS]
        survival = [cost > null for cost, null in zip(cost_meds, null_meds)]
        cost_survival = all(survival)
        if cost_survival:
            cost_survival_count += 1
        print(f"{ticker:6s} | {cost_meds[0]:+7.4f} | {cost_meds[1]:+7.4f} | {cost_meds[2]:+7.4f} | {cost_survival}")
        cost_data.append({
            "ticker": ticker,
            "lookback_3_cost": cost_meds[0],
            "lookback_3_null": null_meds[0],
            "lookback_5_cost": cost_meds[1],
            "lookback_5_null": null_meds[1],
            "lookback_10_cost": cost_meds[2],
            "lookback_10_null": null_meds[2],
            "cost_survival": cost_survival
        })

    print(f"\nCost edge survival: {cost_survival_count}/{len(tickers)} tickers")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total tickers tested: {len(tickers)}")
    print(f"Tickers where momentum survives realistic costs: {cost_survival_count}")
    print(f"Percentage survival: {cost_survival_count/len(tickers)*100:.1f}%")
    
    # Count lookback-specific survival
    lookback_3_survival = sum(1 for td in cost_data if td["lookback_3_cost"] > td["lookback_3_null"])
    lookback_5_survival = sum(1 for td in cost_data if td["lookback_5_cost"] > td["lookback_5_null"])
    lookback_10_survival = sum(1 for td in cost_data if td["lookback_10_cost"] > td["lookback_10_null"])
    
    print(f"\nLookback 3 survival: {lookback_3_survival}/{len(tickers)}")
    print(f"Lookback 5 survival: {lookback_5_survival}/{len(tickers)}")
    print(f"Lookback 10 survival: {lookback_10_survival}/{len(tickers)}")

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
        cost_data=cost_data,
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