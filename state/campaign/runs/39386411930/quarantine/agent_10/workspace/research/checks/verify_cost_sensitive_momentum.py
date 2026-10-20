"""Independent verification of the cost-sensitive momentum check.

This recomputes everything using the same bt.cs_momentum_spread_daily_returns
function to verify that the check's artifact is reproducible.

Design:
- Uses the same bt.cs_momentum_spread_daily_returns function to compute spreads
- Recomputes zero-cost and realistic-cost results using the same logic
- Compares all results against the artifact written by the check.

This validates that the check's artifact is reproducible.
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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "cost_sensitive_momentum_results.json"

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


def main() -> int:
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting verification cost-sensitive momentum run.")
        sys.exit(1)

    print("\n=== 3. Independent verification of cost-sensitive momentum check ===")

    # Load tickers from manifest
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    # Recompute the cost-sensitive momentum check
    print("=== 4. Recomputing cost-sensitive momentum check ===")
    
    results = {}
    
    for lookback in LOOKBACKS:
        print(f"\n--- Recomputing lookback {lookback} ---")
        
        # Zero-cost momentum
        print(f"  Zero-cost: computing momentum spread (lookback={lookback})...")
        momentum_returns = bt.cs_momentum_spread_daily_returns(tickers, lookback, 3, 3)
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
        momentum_returns_real = bt.cs_momentum_spread_daily_returns(tickers, lookback, 3, 3)
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
        null_returns = bt.cs_momentum_null_spread_daily_returns(tickers, lookback, 3, 3, SEED)
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

    # Load artifact for comparison
    print("\n=== 5. Artifact comparison ===")
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)

    # Compare results
    all_match = True
    for lookback in LOOKBACKS:
        print(f"\n--- Comparing lookback {lookback} ---")
        for config_type in ["zero_cost", "realistic_cost", "null_realistic"]:
            artifact_metrics = artifact["cost_sensitivity"][str(lookback)][config_type]
            verified_metrics = results[lookback][config_type]
            print(f"  {config_type}:")
            for metric in ["total_return", "sharpe", "max_drawdown", "turnover_ratio", "n_trades"]:
                artifact_val = artifact_metrics[metric]
                verified_val = verified_metrics[metric]
                match = abs(artifact_val - verified_val) < 1e-6
                if not match:
                    all_match = False
                print(f"    {metric}: artifact {artifact_val:.6f} vs verified {verified_val:.6f} -> {match}")

        # Check cost impact
        artifact_cost = artifact["cost_sensitivity"][str(lookback)]["cost_impact"]
        verified_cost = results[lookback]["cost_impact"]
        print(f"  cost_impact:")
        for metric in ["return_drop", "sharpe_drop", "cost_ratio"]:
            artifact_val = artifact_cost[metric]
            verified_val = verified_cost[metric]
            match = abs(artifact_val - verified_val) < 1e-6
            if not match:
                all_match = False
            print(f"    {metric}: artifact {artifact_val:.6f} vs verified {verified_val:.6f} -> {match}")

    # Overall verdict comparison
    print("\n=== 6. Overall verdict comparison ===")
    artifact_verdict = artifact["verdict"]
    artifact_significant_edges = artifact["significant_edges"]

    # Determine verdict from verified results
    significant_edges = []
    for lookback in LOOKBACKS:
        realistic_ret = results[lookback]["realistic_cost"]["total_return"]
        null_ret = results[lookback]["null_realistic"]["total_return"]
        if realistic_ret > null_ret and abs(realistic_ret - null_ret) > 0.05:
            significant_edges.append(lookback)

    verified_verdict = "CONSISTENT_WITH_NOISE"
    if significant_edges:
        verified_verdict = "REGIME_STABLE"

    verdict_match = artifact_verdict == verified_verdict and artifact_significant_edges == significant_edges
    if not verdict_match:
        all_match = False
    print(f"  Artifact verdict: {artifact_verdict}, Artifact significant edges: {artifact_significant_edges}")
    print(f"  Verified verdict: {verified_verdict}, Verified significant edges: {significant_edges}")
    print(f"  Verdict match: {verdict_match}")

    if all_match:
        print("\n✓ All independent recomputations MATCH the check artifact.")
    else:
        print("\n✗ MISMATCH FOUND: The check artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)

    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Cost-sensitive momentum independent verification on collected real-data universe.")
    return 0


if __name__ == "__main__":
    main()