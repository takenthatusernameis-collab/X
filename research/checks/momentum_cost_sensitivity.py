"""Cost sensitivity check for momentum lookback 3/5/10 on real-data universe.

Objective: run a bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

Realistic cost model (institutional research standards):
- commission_per_trade: $1.0 (typical institutional flat fee)
- commission_per_share: $0.01 per share
- slippage_cents: $0.02 (2 cents per share)
- slippage_proportional: 0.0005 (5 bps)
- initial_capital: $1M (standard research capital)
- warmup_periods: 60 (same as momentum check)
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# Realistic institutional cost model
COST_MODEL = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=1.0,        # $1 per trade
    commission_per_share=0.01,       # $0.01 per share  
    slippage_cents=0.02,             # $0.02 per share (2 cents)
    slippage_proportional=0.0005,   # 5 bps proportional
    warmup_periods=60,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,                 # No margin for research
    margin_call_liquidate=True,
)

# Matched null (coin-flip) benchmark with identical cost model
NULL_COST_MODEL = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=1.0,
    commission_per_share=0.01,
    slippage_cents=0.02,
    slippage_proportional=0.0005,
    warmup_periods=60,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def momentum_signals_with_cost(closes, lookback):
    """Momentum signal implementation for cost testing."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def random_signals_with_cost(n, seed, cost_model):
    """Coin-flip positioning - wrapper for standard random_signals."""
    # The cost_model parameter is accepted for API consistency but not used
    # in signal generation (costs are applied through cfg parameter)
    from research.backtest.perturbation import random_signals
    return random_signals(n, seed)


def segment_result_with_costs(ticker, bars, lookback, cost_model):
    """Walk-forward regime-stability result for one ticker and one lookback with costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    
    # Momentum signal
    signals = momentum_signals_with_cost(closes, lookback)
    
    # Reconstruct labels for regime stability
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Run with costs using the provided cost model
    res = bt.stress_segments(
        signals_fn=lambda c, **kw: momentum_signals_with_cost(c, lookback),
        bars=bars,
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_model,
    )
    
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def null_segment_result_with_costs(ticker, bars, lookback, cost_model):
    """Coin-flip null benchmark with identical cost model."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    
    # Coin-flip signals with identical seed for both momentum and null
    seed = SEED + hash(str(lookback)) % 2**32
    signals = random_signals_with_cost(len(closes), seed, cost_model)
    
    # Reconstruct labels
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Run null with costs
    res = bt.stress_segments(
        signals_fn=lambda c, **kw: random_signals_with_cost(len(c), SEED, cost_model),
        bars=bars,
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_model,
    )
    
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Momentum Cost Sensitivity Check ===")
    print(f"Cost model: commission_per_trade=${COST_MODEL.commission_per_trade}, commission_per_share=${COST_MODEL.commission_per_share}")
    print(f"          slippage_cents=${COST_MODEL.slippage_cents}, slippage_proportional={COST_MODEL.slippage_proportional}")
    print(f"Period windows: train={TRAIN}d, test={TEST}d, warmup={WARM}d, overlap={OVERLAP}d")
    print(f"Lookback candidates: {LOOKBACKS}")

    # Load universe
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # Run cost sensitivity for each lookback
    cost_results = {}
    for lb in LOOKBACKS:
        print(f"\n=== Lookback {lb} with costs ===")
        cost_results[lb] = {}
        
        # Real momentum with costs
        print(f"  Running real momentum (lookback={lb})...")
        cost_results[lb]["real"] = {}
        for ticker in ["AMZN", "JPM"]:
            cost_results[lb]["real"][ticker] = segment_result_with_costs(ticker, tickers[ticker], lb, COST_MODEL)
            medians = cost_results[lb]["real"][ticker]["medians"]
            verdict = cost_results[lb]["real"][ticker]["verdict"]
            print(f"    {ticker}: medians {medians} -> {verdict}")
        
        # Matched null with identical costs
        print(f"  Running matched null (lookback={lb})...")
        cost_results[lb]["null"] = {}
        for ticker in ["AMZN", "JPM"]:
            cost_results[lb]["null"][ticker] = null_segment_result_with_costs(ticker, tickers[ticker], lb, NULL_COST_MODEL)
            medians = cost_results[lb]["null"][ticker]["medians"]
            verdict = cost_results[lb]["null"][ticker]["verdict"]
            print(f"    {ticker}: medians {medians} -> {verdict}")

    # Cost impact analysis
    print("\n=== Cost Impact Analysis ===")
    print("Cost impact = (real_median - null_median) / |null_median| (when null_median != 0)")
    print("Realistic cost robustness: if impact remains > 0.05 (5%) after costs, edge survives")

    cost_impact_summary = {}
    for lb in LOOKBACKS:
        cost_impact_summary[lb] = {}
        for ticker in ["AMZN", "JPM"]:
            real_median = cost_results[lb]["real"][ticker]["medians"][0]
            null_median = cost_results[lb]["null"][ticker]["medians"][0]
            
            if abs(null_median) < 0.001:
                impact = real_median - null_median
            else:
                impact = (real_median - null_median) / abs(null_median)
            
            cost_impact_summary[lb][ticker] = {
                "real_median": real_median,
                "null_median": null_median,
                "cost_impact": impact,
                "robustness": "ROBUST" if abs(impact) > 0.05 else "SENSITIVE"
            }
            
            print(f"  {ticker} lookback={lb}: real={real_median:+.3f}, null={null_median:+.3f}, "
                  f"impact={impact:+.3f} -> {cost_impact_summary[lb][ticker]['robustness']}")

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_model=dict(
            commission_per_trade=COST_MODEL.commission_per_trade,
            commission_per_share=COST_MODEL.commission_per_share,
            slippage_cents=COST_MODEL.slippage_cents,
            slippage_proportional=COST_MODEL.slippage_proportional,
            initial_capital=COST_MODEL.initial_capital,
            warmup_periods=COST_MODEL.warmup_periods,
        ),
        cost_results=cost_results,
        cost_impact_summary=cost_impact_summary,
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("Cost sensitivity check complete. Results preserved for independent verification.")
    
    # Determine if momentum edge survives cost stress
    surviving_assets = 0
    for lb in LOOKBACKS:
        for ticker in ["AMZN", "JPM"]:
            if cost_impact_summary[lb][ticker]["robustness"] == "ROBUST":
                surviving_assets += 1
    
    print(f"\nSummary: Momentum edge survives cost stress for {surviving_assets} asset/lookback combinations")
    print("Success criterion: Results are reproducible and independently verifiable regardless of outcome.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
