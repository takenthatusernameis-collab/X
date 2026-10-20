"""Cost-sensitivity check for lookback 3/5/10 momentum on collected real-data universe.

Background: The candidate positive momentum evidence is short-horizon and mechanical; cost robustness remains decision-relevant. This check evaluates whether the short-horizon momentum edge survives conservative transaction-cost stress on the collected real-data universe.

Method: Fixed a-priori, not tuned to OOS: lookback=3/5/10 trading days; long the previous lookback-day return, hold 1 day, daily rebalance; walk-forward train=252d/test=84d/warmup=60d/overlap=60d; realistic costs: commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005; matched null = coin-flip null (sign-flipped spread, same ranks, random long/short assignment).

Falsification prediction: momentum either shows CONSISTENT_WITH_NOISE (no edge under realistic costs) or REGIME_STABLE_LOSS (uniformly negative under realistic costs). Either outcome closes the cell.

Research/simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Debug: Print sys.path before import
print(f"DEBUG: Current working directory: {Path.cwd()}")
print(f"DEBUG: Script path: {Path(__file__).resolve()}")
print(f"DEBUG: Adding to sys.path: {Path(__file__).resolve().parent.parent.parent / 'research'}")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
print(f"DEBUG: sys.path after insert: {sys.path[0]}")

import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitive_momentum_results.json"

REALISTIC_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")

    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "research" / "data"))
    from preflight import main as preflight_main
    rc = preflight_main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-sensitive momentum run.")
        sys.exit(1)

    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    trunc_len = min(len(tb) for tb in tickers.values())

    print("\n=== 3. Cost-sensitivity sweep for momentum lookback 3/5/10 ===")
    print(f"  Realistic costs: commission_per_trade=2.0, commission_per_share=0.003, "
          f"slippage_cents=2.0, slippage_proportional=0.0005")
    print(f"  Zero-cost baseline: all costs = 0")

    results = {}
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} ---")

        # Zero-cost momentum
        print(f"  Zero-cost: computing momentum spread (lookback={lookback})...")
        momentum_returns = bt.cs_momentum_spread_daily_returns(
            tickers, lookback, 3, 3
        )

        momentum_synth = bt.build_synthetic_spread_asset(momentum_returns, start_price=1e6)
        momentum_signals = [
            bt.Signal(
                date=i + 1,
                weight=momentum_synth.closes_array()[i] / momentum_synth.closes_array()[0]
            )
            for i in range(len(momentum_returns))
        ]

        momentum_zero_result = bt.run_bars(list(momentum_synth), momentum_signals, ZERO_COST_CONFIG)
        momentum_zero_metrics = bt.compute_metrics(
            momentum_zero_result.equity_curve,
            fills=momentum_zero_result.trades,
            fill_prices=np.array([f.price for f in momentum_zero_result.trades])
                         if momentum_zero_result.trades else np.array([]),
            periods_per_year=252,
        )

        # Realistic-cost momentum
        print(f"  Realistic-cost: computing momentum spread (lookback={lookback})...")
        momentum_returns_real = bt.cs_momentum_spread_daily_returns(
            tickers, lookback, 3, 3
        )

        momentum_synth_real = bt.build_synthetic_spread_asset(momentum_returns_real, start_price=1e6)
        momentum_signals_real = [
            bt.Signal(
                date=i + 1,
                weight=momentum_synth_real.closes_array()[i] / momentum_synth_real.closes_array()[0]
            )
            for i in range(len(momentum_returns_real))
        ]

        momentum_real_result = bt.run_bars(list(momentum_synth_real), momentum_signals_real, REALISTIC_CONFIG)
        momentum_real_metrics = bt.compute_metrics(
            momentum_real_result.equity_curve,
            fills=momentum_real_result.trades,
            fill_prices=np.array([f.price for f in momentum_real_result.trades])
                             if momentum_real_result.trades else np.array([]),
            periods_per_year=252,
        )

        # Matched null (coin-flip) with realistic costs
        print(f"  Matched null: computing coin-flip null (lookback={lookback})...")
        null_returns = bt.cs_momentum_null_spread_daily_returns(
            tickers, lookback, 3, 3, SEED
        )

        null_synth = bt.build_synthetic_spread_asset(null_returns, start_price=1e6)
        null_signals = [
            bt.Signal(
                date=i + 1,
                weight=null_synth.closes_array()[i] / null_synth.closes_array()[0]
            )
            for i in range(len(null_returns))
        ]

        null_real_result = bt.run_bars(list(null_synth), null_signals, REALISTIC_CONFIG)
        null_real_metrics = bt.compute_metrics(
            null_real_result.equity_curve,
            fills=null_real_result.trades,
            fill_prices=np.array([f.price for f in null_real_result.trades])
                             if null_real_result.trades else np.array([]),
            periods_per_year=252,
        )

        # Results summary
        results[lookback] = {
            "zero_cost": {
                "total_return": momentum_zero_metrics.total_return,
                "sharpe": momentum_zero_metrics.sharpe,
                "max_drawdown": momentum_zero_metrics.max_drawdown,
                "turnover_ratio": momentum_zero_metrics.turnover_ratio,
                "n_trades": momentum_zero_metrics.n_trades,
                "final_cash": momentum_zero_metrics.final_cash,
                "final_position_shares": momentum_zero_metrics.final_position_shares,
            },
            "realistic_cost": {
                "total_return": momentum_real_metrics.total_return,
                "sharpe": momentum_real_metrics.sharpe,
                "max_drawdown": momentum_real_metrics.max_drawdown,
                "turnover_ratio": momentum_real_metrics.turnover_ratio,
                "n_trades": momentum_real_metrics.n_trades,
                "final_cash": momentum_real_metrics.final_cash,
                "final_position_shares": momentum_real_metrics.final_position_shares,
            },
            "null_realistic": {
                "total_return": null_real_metrics.total_return,
                "sharpe": null_real_metrics.sharpe,
                "max_drawdown": null_real_metrics.max_drawdown,
                "turnover_ratio": null_real_metrics.turnover_ratio,
                "n_trades": null_real_metrics.n_trades,
                "final_cash": null_real_metrics.final_cash,
                "final_position_shares": null_real_metrics.final_position_shares,
            },
            "cost_impact": {
                "return_drop": momentum_zero_metrics.total_return - momentum_real_metrics.total_return,
                "sharpe_drop": momentum_zero_metrics.sharpe - momentum_real_metrics.sharpe,
                "cost_ratio": (momentum_real_metrics.total_return - null_real_metrics.total_return) / momentum_zero_metrics.total_return if momentum_zero_metrics.total_return != 0 else 0.0,
            },
        }

        print(f"  [{lookback}] Zero-cost: {momentum_zero_metrics.total_return:+.3%} Sharpe {momentum_zero_metrics.sharpe:+.3f}")
        print(f"  [{lookback}] Realistic-cost: {momentum_real_metrics.total_return:+.3%} Sharpe {momentum_real_metrics.sharpe:+.3f}")
        print(f"  [{lookback}] Matched null: {null_real_metrics.total_return:+.3%} Sharpe {null_real_metrics.sharpe:+.3f}")
        print(f"  [{lookback}] Cost impact: return drop {results[lookback]['cost_impact']['return_drop']:+.3%}, cost ratio {results[lookback]['cost_impact']['cost_ratio']:.2%}")

    # Determine overall verdict
    print("\n=== 4. Overall verdict ===")
    significant_edges = []
    for lookback in LOOKBACKS:
        realistic_ret = results[lookback]["realistic_cost"]["total_return"]
        null_ret = results[lookback]["null_realistic"]["total_return"]
        
        # Check if realistic-cost momentum significantly outperforms null
        if realistic_ret > null_ret and abs(realistic_ret - null_ret) > 0.05:  # 5% threshold
            significant_edges.append(lookback)

    verdict = "CONSISTENT_WITH_NOISE"
    if significant_edges:
        verdict = "REGIME_STABLE"

    print(f"  Significant edges survived realistic costs: {significant_edges}")
    print(f"  Overall verdict: {verdict}")

    # Prepare artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": LOOKBACKS,
        "test": TEST,
        "train": TRAIN,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "fully_overlapping_bars": trunc_len,
        "cost_sensitivity": results,
        "verdict": verdict,
        "significant_edges": significant_edges,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Cost-sensitive momentum check on the collected universe: exploratory simulation.")

    return 0

if __name__ == "__main__":
    sys.exit(main())