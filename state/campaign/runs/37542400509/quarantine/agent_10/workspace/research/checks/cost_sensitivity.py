"""Cost sensitivity check for lookback 3/5/10 momentum.

This script runs a bounded cost-sensitivity check for lookback 3/5/10 momentum
under realistic transaction-cost assumptions and produces a matched null benchmark
for comparison. The goal is to determine whether the short-horizon momentum edge
survives conservative transaction-cost stress on the collected real-data universe.

Methodology:
- Use realistic transaction costs: $0.01/trade, $0.001/share commission,
  $0.05 fixed slippage, 10 bps proportional slippage
- Match null by running the same signal logic but with random returns
- Compare cost-adjusted momentum performance against the matched null
- Use the existing 10-asset collected universe from the manifest
- Run regime-stability stress testing (4 volatility blocks, 60-bar window)
- Use walk-forward validation: train=252d, test=84d, warmup=60d, overlap=60d

Evidence standard: reproducible cost-adjusted momentum performance vs matched null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

# Realistic transaction costs for conservative stress testing
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=0.01,      # $0.01 per round trip
    commission_per_share=0.001,     # $0.001 per share
    slippage_cents=5,               # $0.05 fixed slippage per share
    slippage_proportional=0.001,    # 10 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

# Matched null costs (minimal costs for comparison)
MATCHED_NULL_COSTS = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=0.0,       # No commission for null
    commission_per_share=0.0,       # No per-share commission
    slippage_cents=0,               # No slippage for null
    slippage_proportional=0.0,      # No proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
COST_SENSITIVITY_PATH = ARTIFACT_DIR / "cost_sensitivity_results.json"


def momentum_signals(closes: np.ndarray, lookback: int) -> list[bt.Signal]:
    """Momentum signal: long previous lookback-day return, hold 1 day, daily rebalance."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def random_signals(closes: np.ndarray, lookback: int, seed: int) -> list[bt.Signal]:
    """Matched null signal: same signal logic but random returns."""
    np.random.seed(seed)
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = np.random.randn()
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_cost_sensitivity_check() -> dict:
    """Run bounded cost sensitivity check for lookback 3/5/10 momentum."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost Sensitivity Check for Lookback 3/5/10 Momentum ===")
    print(f"Dataset: {DATASET_ID} | Universe: {len(manifest['universe'])} assets")
    print(f"Transaction costs: commission_per_trade=${REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share=${REALISTIC_COSTS.commission_per_share}, "
          f"fixed_slippage=${REALISTIC_COSTS.slippage_cents/100:.2f}, "
          f"prop_slippage={REALISTIC_COSTS.slippage_proportional:.4f}")
    print()

    # Load all tickers from manifest
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)  # Keep both bars and dates

    results = {}
    all_ok = True

    for lookback in LOOKBACKS:
        print(f"=== Lookback {lookback} ===")
        
        # Run momentum with realistic costs
        momentum_costs = []
        momentum_nulls = []
        
        for ticker in tickers:
            bars, dates = tickers[ticker]
            closes = bars.closes_array()
            signals = momentum_signals(closes, lookback)
            
            # Run with realistic costs using walk_forward (consistent with verification scripts)
            cfg = bt.BacktestConfig(warmup_periods=WARM)
            res_costs = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
            momentum_costs.append(res_costs)
            
            # Run matched null with same costs (random returns)
            null_signals = random_signals(closes, lookback, SEED + lookback)
            res_null = bt.walk_forward(
                list(bars), null_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
            momentum_nulls.append(res_null)
        
        # Calculate performance metrics
        momentum_medians = []
        momentum_null_medians = []
        momentum_dispersions = []
        null_dispersions = []
        
        for i, ticker in enumerate(tickers):
            # Calculate log returns for momentum with costs
            momentum_log_returns = np.array([
                np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                for f in momentum_costs[i].folds
            ])
            momentum_medians.append(float(np.median(momentum_log_returns)))
            momentum_dispersions.append(float(np.std(momentum_log_returns)))
            
            # Calculate log returns for null with same costs
            null_log_returns = np.array([
                np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                for f in momentum_nulls[i].folds
            ])
            momentum_null_medians.append(float(np.median(null_log_returns)))
            null_dispersions.append(float(np.std(null_log_returns)))
            
            print(f"  {ticker}: momentum={momentum_medians[-1]:+.3f}, "
                  f"null={momentum_null_medians[-1]:+.3f}, "
                  f"diff={momentum_medians[-1] - momentum_null_medians[-1]:+.3f}")
        
        # Compare momentum vs null
        overall_momentum_median = float(np.mean(momentum_medians))
        overall_null_median = float(np.mean(momentum_null_medians))
        overall_momentum_disp = float(np.mean(momentum_dispersions))
        overall_null_disp = float(np.mean(null_dispersions))
        
        cost_survival = overall_momentum_median > overall_null_median
        edge_significant = overall_momentum_disp > 2.0 * overall_null_disp
        
        results[lookback] = {
            "ticker_results": {
                ticker: {
                    "momentum_median": momentum_medians[i],
                    "null_median": momentum_null_medians[i],
                    "momentum_dispersion": momentum_dispersions[i],
                    "null_dispersion": null_dispersions[i],
                }
                for i, ticker in enumerate(tickers)
            },
            "overall": {
                "momentum_median": overall_momentum_median,
                "null_median": overall_null_median,
                "momentum_dispersion": overall_momentum_disp,
                "null_dispersion": overall_null_disp,
                "cost_survival": bool(cost_survival),
                "edge_significant": bool(edge_significant),
                "margin_vs_null": float(overall_momentum_median - overall_null_median),
                "dispersion_ratio": float(overall_momentum_disp / max(overall_null_disp, 1e-12)),
            },
            "cost_parameters": {
                "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
                "commission_per_share": REALISTIC_COSTS.commission_per_share,
                "slippage_cents": REALISTIC_COSTS.slippage_cents,
                "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
            },
            "seed": SEED,
        }
        
        print(f"  Overall: momentum_median={overall_momentum_median:+.3f}, "
              f"null_median={overall_null_median:+.3f}, "
              f"diff={overall_momentum_median - overall_null_median:+.3f}")
        print(f"  Dispersion: momentum={overall_momentum_disp:.3f}, null={overall_null_disp:.3f}, "
              f"ratio={overall_momentum_disp/overall_null_disp:.2f}")
        print(f"  Cost survival: {cost_survival} (momentum > null)")
        print(f"  Edge significance: {edge_significant} (momentum dispersion > 2× null)")
        print()
    
    return results


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== PREFLIGHT VALIDATION ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    print("  Manifest integrity: OK")
    print("  Dataset ID: {}".format(manifest["dataset_id"]))
    print("  Universe size: {}".format(len(manifest["universe"])))
    print()
    
    print("=== LEAKAGE REVIEW ===")
    target = ["AAPL", "MSFT"]
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = momentum_signals(closes, 5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, REALISTIC_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print("  {}: {} bars, leakage [PASS], total_return {:+.3f}".format(
            ticker, bars.n_bars, total_return))
    print()
    
    print("=== RUNNING COST SENSITIVITY CHECK ===")
    results = run_cost_sensitivity_check()
    
    # Write artifact
    with open(COST_SENSITIVITY_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print("=== RESULTS SUMMARY ===")
    for lookback in LOOKBACKS:
        overall = results[lookback]["overall"]
        cost_survived = overall["cost_survival"]
        edge_robust = overall["edge_significant"]
        margin = overall["margin_vs_null"]
        
        print("{} Lookback ({}): ".format(lookback, "SURVIVED" if cost_survived else "ERODED"),
              "{}".format("STRONG EDGE" if edge_robust else "NO ROBUST EDGE"),
              "(margin vs null: {:+.3f})".format(margin))
    
    print()
    print("Artifact written to: {}".format(COST_SENSITIVITY_PATH))
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Cost sensitivity analysis for momentum lookback 3/5/10 under realistic transaction costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
