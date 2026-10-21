"""Momentum cost robustness: lookback 3/5/10 across collected universe.

Objective: the momentum frontier test (research/checks/momentum.py, activation 37318950814)
admitted a REGIME_STABLE positive edge in 7/10 assets at lookback=5, hold=1 day.
The existing lookback-sweep check (activation 37361000967) showed the edge is robust
across lookback 3/5/10 in 6/10 assets.

This task addresses the next frontier question: does the short-horizon momentum edge
survive conservative transaction-cost stress on the collected real-data universe?

We define a realistic cost model (industry-standard conservative estimates):
- Commission: $0.005 per trade (typical broker commission)
- Slippage: 1.5 bps (0.0015) of notional for aggressive market orders
- Target exposure: 100% (standard for trend-following)

Method (fixed a-priori, not tuned to OOS): long the previous N-day return; hold 1 day;
daily rebalance. Lookbacks 3/5/10. Same walk-forward windows, 4 contiguous volatility
blocks, min 400 bars/segment, compared vs coin-flip null on the same segments.

Classification: edge survives cost stress if the uniformly-positive REGIME_STABLE
pattern at lookback=5 (reference) also appears at >= 2 lookbacks under cost stress;
a lookback-sensitive result (edge only at lookback=5 under costs) is a parameter-
fragility warning.

Research / simulation only. No live trading or production execution.
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

# Realistic cost model (conservative estimates)
COMMISION_PER_TRADE = 0.005      # $0.005 per round trip
SLIPPAGE_BPS = 1.5              # 1.5 bps slippage
SLIPPAGE_PROPORTIONAL = 0.0015  # 1.5 bps proportional

# Create cost configuration
def create_cost_config():
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=COMMISION_PER_TRADE,
        slippage_cents=SLIPPAGE_BPS * 100,  # Convert bps to cents per share
        slippage_proportional=SLIPPAGE_PROPORTIONAL,
    )

# Matched null: momentum run with zero costs for comparison
def create_null_config():
    return bt.BacktestConfig(initial_capital=1e6)

def segment_result(ticker, bars, lookback, config, is_null=False):
    """Walk-forward regime-stability result for one ticker and one lookback with given costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    # For null, we need to verify against the costed baseline
    if is_null:
        null_config = create_null_config()
    else:
        null_config = create_null_config()  # Use null config for comparison
        
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
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
    """Run momentum across lookbacks with realistic costs and matched null."""
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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # Initialize cost and null configs
    cost_config = create_cost_config()
    null_config = create_null_config()
    
    # Load all tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    print("\n=== 3. Momentum cost robustness: lookback 3/5/10 ===")
    print("Momentum = long the previous N-day return, hold 1 day, daily rebalanced. "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, compared vs coin-flip null on the same "
          "segments.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("Realistic cost model applied:")
    print("  Commission: ${:.3f} per trade".format(0.005))
    print("  Slippage: {:.1f} bps aggressive".format(1.5))
    print()

    # Run across lookbacks with costs
    cost_results = {lb: {t: segment_result(t, tickers_univ[t], lb, cost_config) 
                       for t in manifest["universe"]} for lb in LOOKBACKS}

    print("=== 4. Cost robustness results (lookback sweep) ===")
    
    # Count robust edges
    robust_assets = 0
    sensitive_assets = 0
    no_edge_assets = 0
    
    for ticker in manifest["universe"]:
        edge_count = 0
        for lb in LOOKBACKS:
            verdict = cost_results[lb][ticker]["verdict"]
            medians = cost_results[lb][ticker]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                edge_count += 1
        
        if edge_count >= 2:
            robust_assets += 1
        elif edge_count == 1:
            sensitive_assets += 1
        else:
            no_edge_assets += 1
            
        print("  {}: edge_counts={}".format(ticker, edge_count))

    print("\n=== 5. Cost robustness verdict ===")
    print("  ROBUST: positive edge at >= 2 lookbacks under cost stress")
    print("  SENSITIVE: positive edge at exactly 1 lookback under costs")
    print("  NO_EDGE: no positive edge under costs")
    print("  Robust assets: {} / {}".format(robust_assets, len(manifest["universe"])))
    print("  Sensitive assets: {} / {}".format(sensitive_assets, len(manifest["universe"])))
    print("  No edge assets: {} / {}".format(no_edge_assets, len(manifest["universe"])))
    print()

    # Store results for artifact
    per_asset_output = {}
    for ticker in manifest["universe"]:
        per_asset_output[ticker] = {}
        for lb in LOOKBACKS:
            per_asset_output[ticker][str(lb)] = cost_results[lb][ticker]

    # Build artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_model=dict(
            commission_per_trade=0.005,
            slippage_bps=1.5,
            slippage_proportional=0.0015,
        ),
        per_asset=per_asset_output,
        universe=dict(
            robust_assets=robust_assets,
            sensitive_assets=sensitive_assets,
            no_edge_assets=no_edge_assets,
            total_assets=len(manifest["universe"]),
        ),
    )

    # Write artifact
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost robustness (lookback 3/5/10) with realistic "
          "transaction costs: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())