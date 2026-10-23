"""Cost robustness check for momentum lookbacks 3/5/10.

Task: Run a bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

Uses existing 10-asset collected universe and existing momentum implementation/check framework.
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

# Realistic cost model based on repository test suite
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.003,      # 3 basis points per share
    slippage_cents=5.0,               # $0.05 per share
    commission_per_trade=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    periods_per_year=252,
)

# Matched null: same costs, same universe, but momentum signal randomized
# Using the repository's existing coin-flip null mechanism
REALISTIC_NULL_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.003,      # Same costs
    slippage_cents=5.0,               # Same slippage
    commission_per_trade=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    periods_per_year=252,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

def load_collected_universe():
    """Load the collected 10-asset universe from manifest."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"
    
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars
    return tickers

def run_matched_null_bars(bars, lookback, seed_offset=0):
    """Generate a matched null by using the repository's coin-flip mechanism.
    
    This replicates the existing null methodology from regime_stability.py.
    """
    closes = bars.closes_array()
    n = len(closes)
    rng = np.random.default_rng(SEED + seed_offset)
    
    # Generate coin-flip randomized signal: same magnitude as momentum,
    # but with random sign for each bar
    signals = []
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        # Coin-flip sign for each bar (independent random)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        signals.append(bt.Signal(date=i + 1, weight=sign * abs(ret) * 0.01))  # scale down to match momentum magnitude
    
    return signals

def compute_fold_medians(tickers, lookback, costs, signals_fn=None):
    """Compute per-segment medians for a given lookback and costs.
    
    Uses the existing walk-forward regime-stability framework from the repository.
    """
    # Use the same pattern as momentum.py and momentum_sweep.py
    # Use stress_segments_across_tickers for simplicity
    return bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=costs,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
        
        results[ticker] = {
            "segments": [s.name for s in res.scenarios],
            "medians": [round(s.baseline_median_log_return, 4) for s in res.scenarios],
            "null_medians": [round(s.noise_median_log_return, 4) for s in res.scenarios],
            "candidate_dispersion": round(res.candidate_dispersion, 4),
            "null_dispersion": round(res.null_dispersion, 4),
            "verdict": res.overall_verdict,
            "n_folds": res.n_folds,
        }
    
    return results

def coin_flip_null_matched(tickers, lookback):
    """Create a matched null using the repository's existing null mechanism."""
    # Use the same pattern as the existing null benchmark from perturbation.py
    # Generate coin-flip randomized signals using the existing mechanism
    def null_signals_fn(closes, lookback):
        """Generate null signals using coin-flip randomization (matched to momentum)."""
        # Use the repository's existing null signal generation approach
        # This mimics the existing null benchmark in perturbation.py
        n = len(closes)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        rng = np.random.default_rng(SEED)
        
        for i in range(lookback, n):
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            # Coin-flip sign for each bar (independent random)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            # Scale to match momentum magnitude
            out[i] = bt.Signal(date=i + 1, weight=sign * abs(ret) * 0.01)
        
        return out
    
    # Use stress_segments_across_tickers like the main momentum function
    return bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: null_signals_fn(c, lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_NULL_COSTS,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
        
        null_results[ticker] = {
            "segments": [s.name for s in res.scenarios],
            "medians": [round(s.baseline_median_log_return, 4) for s in res.scenarios],
            "null_medians": [round(s.noise_median_log_return, 4) for s in res.scenarios],
            "candidate_dispersion": round(res.candidate_dispersion, 4),
            "null_dispersion": round(res.null_dispersion, 4),
            "verdict": res.overall_verdict,
            "n_folds": res.n_folds,
        }
    
    return null_results

def main() -> int:
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
        print("PREFLIGHT FAILED; aborting cost robustness run.")
        sys.exit(1)
    
    print("\n=== 3. Momentum cost robustness check (lookback 3 / 5 / 10) ===")
    print("Momentum = long the previous N-day return, hold 1 day, daily rebalanced.")
    print(f"Params: realistic costs (commission_per_share=0.003, slippage_cents=5.0), "
          f"walk-forward per segment train={TRAIN}d / test={TEST}d / warmup={WARM}d / "
          f"overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs matched null.")
    
    # Load the collected universe
    tickers = load_collected_universe()
    
    # Run momentum with realistic costs for each lookback
    print("\n=== 4. Momentum with realistic costs ===")
    momentum_summary = {}
    for lb in LOOKBACKS:
        print(f"  lookback {lb}: running momentum with realistic costs...")
        results = compute_fold_medians(tickers, lb, REALISTIC_COSTS)
        momentum_summary[lb] = results
        
        # Summary statistics
        edges = []
        losses = []
        noise = []
        for ticker, r in results.assets.items():
            if hasattr(r, 'baseline_median_log_return') and abs(r.baseline_median_log_return) > 0.01:
                edges.append(ticker)
            verdict = r.overall_verdict
            if verdict == "REGIME_STABLE_LOSS":
                losses.append(ticker)
            elif verdict == "CONSISTENT_WITH_NOISE":
                noise.append(ticker)
        
        print(f"    edges: {len(edges)} non-zero median returns")
        print(f"    REGIME_STABLE edges: {len([r for r in results.assets.values() if r.overall_verdict == 'REGIME_STABLE'])} / {len(results.assets)}")
        print(f"    REGIME_STABLE_LOSS: {len(losses)} assets")
        print(f"    CONSISTENT_WITH_NOISE: {len(noise)} assets")
    
    # Run matched null with same realistic costs
    print("\n=== 5. Matched null with same realistic costs ===")
    null_summary = {}
    for lb in LOOKBACKS:
        print(f"  lookback {lb}: running matched null with realistic costs...")
        # Use the repository's existing coin-flip null mechanism
        null_summary[lb] = coin_flip_null_matched(tickers, lb)
        
        # Summary statistics for null
        edges = []
        losses = []
        noise = []
        for ticker, r in null_summary[lb].assets.items():
            if hasattr(r, 'baseline_median_log_return') and abs(r.baseline_median_log_return) > 0.01:
                edges.append(ticker)
            verdict = r.overall_verdict
            if verdict == "REGIME_STABLE_LOSS":
                losses.append(ticker)
            elif verdict == "CONSISTENT_WITH_NOISE":
                noise.append(ticker)
        
        print(f"    null edges: {len(edges)} non-zero median returns")
        print(f"    REGIME_STABLE edges: {len([r for r in null_summary[lb].assets.values() if r.overall_verdict == 'REGIME_STABLE'])} / {len(null_summary[lb].assets)}")
        print(f"    REGIME_STABLE_LOSS: {len(losses)} assets")
        print(f"    CONSISTENT_WITH_NOISE: {len(noise)} assets")
    
    # Cost robustness analysis
    print("\n=== 6. Cost robustness analysis ===")
    cost_robustness = {}
    for lb in LOOKBACKS:
        print(f"\n  Lookback {lb}:")
        
        momentum_edges = []
        null_edges = []
        
        for ticker in tickers.keys():
            if momentum_results[lb][ticker]["verdict"] == "REGIME_STABLE" and all(m > 0.001 for m in momentum_results[lb][ticker]["medians"]):
                momentum_edges.append(ticker)
            
            if null_results[lb][ticker]["verdict"] == "REGIME_STABLE" and all(m > 0.001 for m in null_results[lb][ticker]["medians"]):
                null_edges.append(ticker)
        
        # Compute comparative statistics
        momentum_medians = []
        null_medians = []
        
        for ticker in tickers.keys():
            momentum_medians.extend(momentum_results[lb][ticker]["medians"])
            null_medians.extend(null_results[lb][ticker]["medians"])
        
        momentum_mean = np.mean(momentum_medians) if momentum_medians else 0.0
        null_mean = np.mean(null_medians) if null_medians else 0.0
        momentum_std = np.std(momentum_medians, ddof=1) if momentum_medians else 0.0
        null_std = np.std(null_medians, ddof=1) if null_medians else 0.0
        
        # Statistical significance test (assuming independent samples)
        if len(momentum_medians) > 1 and len(null_medians) > 1:
            se_diff = np.sqrt(momentum_std**2 / len(momentum_medians) + null_std**2 / len(null_medians))
            t_stat = (momentum_mean - null_mean) / se_diff if se_diff > 0 else 0.0
            # Conservative threshold for significance given realistic costs
            is_significant = abs(t_stat) > 2.0  # 95% confidence roughly
        else:
            is_significant = False
        
        cost_robustness[lb] = {
            "momentum_verdict_counts": {
                "REGIME_STABLE": len([r for r in momentum_results[lb].values() if r["verdict"] == "REGIME_STABLE"]),
                "REGIME_STABLE_LOSS": len([r for r in momentum_results[lb].values() if r["verdict"] == "REGIME_STABLE_LOSS"]),
                "CONSISTENT_WITH_NOISE": len([r for r in momentum_results[lb].values() if r["verdict"] == "CONSISTENT_WITH_NOISE"]),
            },
            "null_verdict_counts": {
                "REGIME_STABLE": len([r for r in null_results[lb].values() if r["verdict"] == "REGIME_STABLE"]),
                "REGIME_STABLE_LOSS": len([r for r in null_results[lb].values() if r["verdict"] == "REGIME_STABLE_LOSS"]),
                "CONSISTENT_WITH_NOISE": len([r for r in null_results[lb].values() if r["verdict"] == "CONSISTENT_WITH_NOISE"]),
            },
            "momentum_mean_median": momentum_mean,
            "null_mean_median": null_mean,
            "momentum_std_median": momentum_std,
            "null_std_median": null_std,
            "significant_difference": bool(is_significant),
            "t_statistic": float(t_stat) if 't_stat' in locals() else 0.0,
            "momentum_tickers_with_edges": momentum_edges,
            "null_tickers_with_edges": null_edges,
        }
        
        print(f"    momentum mean median: {momentum_mean:+.4f}")
        print(f"    null mean median: {null_mean:+.4f}")
        print(f"    significant difference (p < 0.05): {is_significant}")
        print(f"    momentum REGIME_STABLE edges: {len(momentum_edges)} / {len(tickers)} assets")
        print(f"    null REGIME_STABLE edges: {len(null_edges)} / {len(tickers)} assets")
    
    # Determinism check
    print("\n=== 7. Determinism check ===")
    deterministic = True
    for lb in LOOKBACKS:
        # Rerun momentum and compare
        results2 = compute_fold_medians(tickers, lb, REALISTIC_COSTS)
        
        # Compare results
        for ticker in tickers.keys():
            r1 = momentum_results[lb][ticker]
            r2 = results2[ticker]
            
            # Compare key fields
            if (r1["verdict"] != r2["verdict"] or
                r1["medians"] != r2["medians"] or
                r1["null_medians"] != r2["null_medians"]):
                deterministic = False
                print(f"  NON-DETERMINISTIC: lookback {lb}, ticker {ticker}")
    
    print(f"  determinism: {'PASS' if deterministic else 'FAIL'}")
    if not deterministic:
        print("  WARNING: Non-deterministic results detected!")
    
    # Create final artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_costs": {
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
            "target_exposure": REALISTIC_COSTS.target_exposure,
        },
        "cost_robustness": cost_robustness,
        "deterministic": deterministic,
        "notes": "Cost robustness check using realistic transaction costs (commission_per_share=0.003, slippage_cents=5.0) vs matched null benchmark. Research/simulation only.",
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print("\n=== 8. Artifact written ===")
    print(f"  artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost robustness check with realistic transaction costs: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())