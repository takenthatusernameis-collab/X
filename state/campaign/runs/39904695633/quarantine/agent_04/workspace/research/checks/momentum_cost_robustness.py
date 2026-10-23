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

# "Existing realistic cost model" from BacktestConfig defaults
# Zero costs in this repository, but represents the existing baseline
REALISTIC_COST_CONFIG = bt.BacktestConfig(initial_capital=1e6)

# Conservative transaction cost grid to test robustness
# Based on typical retail trading costs: $0.01/share commission, 5bps slippage
CONSERVATIVE_COST_GRID = [
    bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=0.0,
        commission_per_share=0.01,  # $0.01 per share
        slippage_cents=0.0,         # 0 cents fixed slippage
        slippage_proportional=0.0005,  # 5bps proportional slippage
    ),
    bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=5.0,   # $5 per round trip
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.001,   # 10bps
    ),
    bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=0.0,
        commission_per_share=0.02,  # $0.02 per share
        slippage_cents=10.0,        # $0.10 fixed slippage per share
        slippage_proportional=0.002,   # 20bps
    ),
]


def run_cost_sensitivity_test(ticker, bars, lookback, cost_config):
    """Run momentum test with specific cost configuration."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
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
        cfg=cost_config,
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
        cost_config={
            "commission_per_trade": cost_config.commission_per_trade,
            "commission_per_share": cost_config.commission_per_share,
            "slippage_cents": cost_config.slippage_cents,
            "slippage_proportional": cost_config.slippage_proportional,
        },
    )


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

    print("\n=== 3. Momentum cost-robustness test (lookback 3/5/10) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs coin-flip null."
          "Cost grid includes realistic retail transaction cost assumptions."
          "\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Test each lookback with each cost configuration
    all_lookback_costs = {}
    
    for cost_idx, cost_config in enumerate(CONSERVATIVE_COST_GRID):
        print(f"\n=== Cost configuration {cost_idx + 1} ===")
        print(f"  commission_per_trade: ${cost_config.commission_per_trade}")
        print(f"  commission_per_share: ${cost_config.commission_per_share}")
        print(f"  slippage_cents: ${cost_config.slippage_cents/100:.3f}")
        print(f"  slippage_proportional: {cost_config.slippage_proportional*10000:.1f} bps")
        
        cost_key = f"cost_{cost_idx}"
        all_lookback_costs[cost_key] = {}
        
        for lookback in LOOKBACKS:
            print(f"\n  Lookback {lookback}:")
            lookback_results = {}
            
            for ticker in manifest["universe"]:
                bars = bt.load_ticker(ticker)[0]
                result = run_cost_sensitivity_test(ticker, bars, lookback, cost_config)
                lookback_results[ticker] = result
                verdict = result["verdict"]
                edge = "EDGE" if verdict == "REGIME_STABLE" and all(m > 0 for m in result["medians"]) else "NO_EDGE"
                print(f"    {ticker}: {verdict} (edge: {edge})")
            
            all_lookback_costs[cost_key][f"lookback_{lookback}"] = lookback_results

    # Determinism: rerun the reference cost (index 0) and assert identical output
    print("\n=== 4. Determinism (reference cost config) ===")
    reference_costs = CONSERVATIVE_COST_GRID[0]
    determinism_passed = True
    
    for lookback in LOOKBACKS:
        for ticker in manifest["universe"]:
            bars = bt.load_ticker(ticker)[0]
            r1 = run_cost_sensitivity_test(ticker, bars, lookback, reference_costs)
            r2 = run_cost_sensitivity_test(ticker, bars, lookback, reference_costs)
            if r1 != r2:
                print(f"    NON-DETERMINISTIC: {ticker} lookback={lookback}")
                determinism_passed = False
    
    if determinism_passed:
        print("  ✓ All cost-robustness runs are deterministic")
    else:
        print("  ✗ Non-deterministic output detected")

    # Classification: does the edge survive conservative transaction costs?
    print("\n=== 5. Cost robustness classification ===")
    classification_results = {}
    
    for cost_idx, cost_config in enumerate(CONSERVATIVE_COST_GRID):
        cost_key = f"cost_{cost_idx}"
        robust_lookbacks = []
        sensitive_lookbacks = []
        no_edge_lookbacks = []
        
        for lookback in LOOKBACKS:
            edge_count = 0
            for ticker in manifest["universe"]:
                result = all_lookback_costs[cost_key][f"lookback_{lookback}"][ticker]
                if result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in result["medians"]):
                    edge_count += 1
            
            if edge_count >= len(manifest["universe"]) * 0.6:  # 60% threshold
                robust_lookbacks.append(lookback)
            elif edge_count > 0:
                sensitive_lookbacks.append(lookback)
            else:
                no_edge_lookbacks.append(lookback)
        
        # Overall cost robustness verdict
        if len(robust_lookbacks) >= 2:  # At least 2 lookbacks show robust edge
            cost_verdict = "ROBUST"
            cost_reason = f"Positive edge survives across {len(robust_lookbacks)}/{len(LOOKBACKS)} lookbacks under conservative costs"
        elif len(sensitive_lookbacks) > 0:
            cost_verdict = "SENSITIVE"
            cost_reason = f"Positive edge present but not robust under conservative costs (sensitive lookbacks: {sensitive_lookbacks})"
        else:
            cost_verdict = "DESTROYED"
            cost_reason = f"Positive edge eliminated by conservative transaction costs"
        
        classification_results[cost_key] = {
            "verdict": cost_verdict,
            "reason": cost_reason,
            "robust_lookbacks": robust_lookbacks,
            "sensitive_lookbacks": sensitive_lookbacks,
            "no_edge_lookbacks": no_edge_lookbacks,
        }
        
        print(f"  Cost {cost_idx}: {cost_verdict} - {cost_reason}")

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_cost_config={
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
        },
        conservative_cost_grid=[{
            "commission_per_trade": c.commission_per_trade,
            "commission_per_share": c.commission_per_share,
            "slippage_cents": c.slippage_cents,
            "slippage_proportional": c.slippage_proportional,
        } for c in CONSERVATIVE_COST_GRID],
        cost_robustness_results=all_lookback_costs,
        classification=classification_results,
        determinism_passed=determinism_passed,
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost robustness test across lookbacks 3/5/10: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())