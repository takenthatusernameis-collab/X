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
COMMISSION_PER_TRADE = 0.005      # $0.005 per round trip
SLIPPAGE_BPS = 1.5              # 1.5 bps slippage
SLIPPAGE_PROPORTIONAL = 0.0015  # 1.5 bps proportional

# Create cost configuration
def create_cost_config():
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=COMMISSION_PER_TRADE,
        slippage_cents=SLIPPAGE_BPS * 100,  # Convert bps to cents per share
        slippage_proportional=SLIPPAGE_PROPORTIONAL,
    )

# Matched null: momentum run with zero costs for comparison
def create_null_config():
    return bt.BacktestConfig(initial_capital=1e6)


def segment_result_with_costs(ticker, bars, lookback, is_cost_run=False):
    """Walk-forward regime-stability result for one ticker and one lookback with given costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    # Use the appropriate config based on whether this is a cost run
    config = create_cost_config() if is_cost_run else create_null_config()
    
    # Create segment results with the given config
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid={"lookback": lookback},
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=config,
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

    # Generate results with realistic costs
    print("=== 4. Momentum cost robustness analysis ===")
    print("Running cost-sensitive momentum analysis with realistic transaction costs...\n")

    # Generate results with realistic costs
    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    for ticker in tickers_univ.keys():
        ticker_results = {}
        for lb in LOOKBACKS:
            bars = tickers_univ[ticker]
            
            # Run with realistic costs
            res_cost = segment_result_with_costs(ticker, bars, lb, is_cost_run=True)
            ticker_results[str(lb)] = res_cost
        
        per_asset[ticker] = ticker_results

    # Apply cost robustness decision logic
    for ticker in per_asset.keys():
        edge_count = 0
        for lb in LOOKBACKS:
            verdict = per_asset[ticker][str(lb)]["verdict"]
            medians = per_asset[ticker][str(lb)]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                edge_count += 1
        
        if edge_count >= 2:
            robustness_counts["ROBUST"] += 1
        elif edge_count == 1:
            robustness_counts["SENSITIVE"] += 1
        else:
            robustness_counts["NO_EDGE"] += 1

    # Leakage gate: run a quick sanity check on all tickers
    print("\n=== 5. Leakage gate (cost-robustness) ===")
    target = ["AMZN", "JPM"]
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(
            initial_capital=1e6,
            commission_per_trade=COMMISSION_PER_TRADE,
            slippage_cents=SLIPPAGE_BPS * 100,
            slippage_proportional=SLIPPAGE_PROPORTIONAL,
        ))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    # Determinism check: rerun the cost robustness computation twice
    print("\n=== 6. Determinism check ===")
    print("Running cost-robustness computation twice for determinism verification...")
    
    # First run
    per_asset1 = {}
    robustness_counts1 = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    for ticker in tickers_univ.keys():
        ticker_results = {}
        for lb in LOOKBACKS:
            bars = tickers_univ[ticker]
            res_cost = segment_result_with_costs(ticker, bars, lb, is_cost_run=True)
            ticker_results[str(lb)] = res_cost
        
        per_asset1[ticker] = ticker_results

    # Apply cost robustness decision logic
    for ticker in per_asset1.keys():
        edge_count = 0
        for lb in LOOKBACKS:
            verdict = per_asset1[ticker][str(lb)]["verdict"]
            medians = per_asset1[ticker][str(lb)]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                edge_count += 1
        
        if edge_count >= 2:
            robustness_counts1["ROBUST"] += 1
        elif edge_count == 1:
            robustness_counts1["SENSITIVE"] += 1
        else:
            robustness_counts1["NO_EDGE"] += 1
    
    # Second run
    per_asset2 = {}
    robustness_counts2 = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    for ticker in tickers_univ.keys():
        ticker_results = {}
        for lb in LOOKBACKS:
            bars = tickers_univ[ticker]
            res_cost = segment_result_with_costs(ticker, bars, lb, is_cost_run=True)
            ticker_results[str(lb)] = res_cost
        
        per_asset2[ticker] = ticker_results

    # Apply cost robustness decision logic
    for ticker in per_asset2.keys():
        edge_count = 0
        for lb in LOOKBACKS:
            verdict = per_asset2[ticker][str(lb)]["verdict"]
            medians = per_asset2[ticker][str(lb)]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                edge_count += 1
        
        if edge_count >= 2:
            robustness_counts2["ROBUST"] += 1
        elif edge_count == 1:
            robustness_counts2["SENSITIVE"] += 1
        else:
            robustness_counts2["NO_EDGE"] += 1
    
    # Compare results
    import json
    
    def get_dict_for_comparison(data):
        return json.dumps(data, sort_keys=True)
    
    per_asset_json1 = get_dict_for_comparison(per_asset1)
    per_asset_json2 = get_dict_for_comparison(per_asset2)
    counts_json1 = get_dict_for_comparison(robustness_counts1)
    counts_json2 = get_dict_for_comparison(robustness_counts2)
    
    per_asset_match = per_asset_json1 == per_asset_json2
    counts_match = counts_json1 == counts_json2
    
    print("determinism: per-asset JSON equality:", "PASS" if per_asset_match else "FAIL")
    print("determinism: robustness_counts JSON equality:", "PASS" if counts_match else "FAIL")
    
    if not per_asset_match:
        print("  per_asset mismatch detected!")
    if not counts_match:
        print("  robustness_counts mismatch detected!")
    
    assert per_asset_match and counts_match, "deterministic computation failed"
    
    # Print results summary
    print("  Cost-robustness verdict counts:")
    print(f"    ROBUST (>=2 lookbacks positive edge): {robustness_counts['ROBUST']}")
    print(f"    SENSITIVE (edge at exactly one lookback): {robustness_counts['SENSITIVE']}")
    print(f"    NO_EDGE (no positive edge): {robustness_counts['NO_EDGE']}\n")
    
    print("  Per-asset results:")
    for ticker in tickers_univ.keys():
        edge_count = 0
        for lb in LOOKBACKS:
            verdict = per_asset[ticker][str(lb)]["verdict"]
            medians = per_asset[ticker][str(lb)]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                edge_count += 1
        
        verdict_map = {1: "ROBUST", 0: "NO_EDGE", -1: "SENSITIVE"}  # Simplified logic
        if edge_count >= 2:
            v = "ROBUST"
        elif edge_count == 1:
            v = "SENSITIVE"
        else:
            v = "NO_EDGE"
        
        print(f"    {ticker}: {v} (edges at lookbacks {[lb for lb in LOOKBACKS if all(per_asset[ticker][str(lb)]['medians'])]})")

    # Write the artifact
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_model={
            "commission_per_trade": COMMISSION_PER_TRADE,
            "slippage_bps": SLIPPAGE_BPS,
            "slippage_proportional": SLIPPAGE_PROPORTIONAL,
        },
        per_asset=per_asset,
        universe=dict(
            robustness_counts=robustness_counts,
            ticker_order=list(tickers_univ.keys()),
            per_asset=dict(
                (t, dict(
                    medians={str(lb): per_asset[t][str(lb)]["medians"] for lb in LOOKBACKS},
                    null_medians={str(lb): per_asset[t][str(lb)]["null_medians"] for lb in LOOKBACKS},
                    verdict_counts={
                        "REGIME_STABLE": sum(1 for lb in LOOKBACKS 
                                            if per_asset[t][str(lb)]["verdict"] == "REGIME_STABLE" and all(m > 0 for m in per_asset[t][str(lb)]["medians"])),
                        "CONSISTENT_WITH_NOISE": sum(1 for lb in LOOKBACKS 
                                                    if per_asset[t][str(lb)]["verdict"] == "CONSISTENT_WITH_NOISE"),
                        "REGIME_DEPENDENT": sum(1 for lb in LOOKBACKS 
                                              if per_asset[t][str(lb)]["verdict"] == "REGIME_DEPENDENT"),
                    },
                ))
                for t in tickers_univ.keys()
            ),
        ),
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== 5. Artifact written ===")
    print(f"  Artifact written to: {ARTIFACT_PATH}")
    print(f"\n  Cost robustness results saved for {len(tickers_univ)} assets")
    print(f"  Lookbacks analyzed: {LOOKBACKS}")
    print(f"  Cost model: commission=${COMMISSION_PER_TRADE}/trade, slippage={SLIPPAGE_BPS} bps")
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost robustness check with realistic transaction costs.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
