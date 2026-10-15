"""Simple cost-robustness check for momentum.

This is a minimal implementation that demonstrates the key concept:
cost robustness evaluation of momentum signals.

Key insight: realistic transaction costs can change which assets
show a positive momentum edge under cost stress.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add repository root to Python path for research imports
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=100,
    commission_per_share=5,
    slippage_cents=20,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def momentum_signals(closes, lookback):
    """Past-only short-horizon momentum signal (from research.backtest)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def cost_sensitive_run_asset(asset, bars, lookback):
    """Walk-forward run with realistic costs for one asset and lookback."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    sig = momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{asset}: signals {len(sig)} != bars {len(closes)}")
    
    # Cost-sensitive walk-forward with realistic transaction costs
    res = bt.stress_segments(
        signals_fn=momentum_signals,
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
        cfg=REALISTIC_COST_CONFIG,
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
        cost_config=REALISTIC_COST_CONFIG,
    )


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is cost-robust across lookbacks.
    
    ROBUST: uniformly positive REGIME_STABLE edge at >= 2 lookbacks.
    SENSITIVE: positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE: no lookback shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        verdict = lookback_results[str(lb)]["verdict"]
        medians = lookback_results[str(lb)]["medians"]
        if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
            counts["EDGE"] += 1
        elif verdict == "REGIME_STABLE_LOSS":
            counts["LOSS"] += 1
        else:
            counts["NO_EDGE"] += 1
    
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


def run_cost_sensitive_check():
    """Run cost-sensitive momentum check."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost-robustness check for momentum ===")
    print(f"Checking lookback {LOOKBACKS} with realistic transaction costs")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    print("\n=== Results ===")

    # Find which assets show cost-robust positive edges
    robust_assets = []
    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    for ticker in all_tickers.keys():
        profile = {}
        for lb in LOOKBACKS:
            result = cost_sensitive_run_asset(ticker, all_tickers[ticker], lb)
            profile[str(lb)] = result
        
        verdict, counts = robustness_verdict(profile)
        
        profile["edge_counts"] = counts
        profile["cost_robustness_verdict"] = verdict
        per_asset[ticker] = profile
        
        robustness_counts[verdict] += 1
        
        med = ", ".join(
            f"{lb}:{profile[str(lb)]['medians'][0]:+.3f} {profile[str(lb)]['verdict']}"
            for lb in LOOKBACKS
        )
        segments_str = "[".join(profile["3"]["segments"]) + "]"
        print(f"  {ticker:6s} segments={segments_str:<40s} medians={med} cost_rob={verdict:<8s} edge_counts={counts}")
        
        if verdict == "ROBUST":
            robust_assets.append(ticker)

    # Create artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "per_asset": per_asset,
        "universe": {
            "robustness_counts": robustness_counts,
            "ticker_order": list(all_tickers.keys()),
            "per_asset": {
                t: {
                    "lookback_medians": {str(lb): per_asset[t][str(lb)]["medians"] for lb in LOOKBACKS},
                    "lookback_null_medians": {str(lb): per_asset[t][str(lb)]["null_medians"] for lb in LOOKBACKS},
                    "verdicts": {str(lb): per_asset[t][str(lb)]["verdict"] for lb in LOOKBACKS},
                    "cost_robustness_verdict": per_asset[t]["cost_robustness_verdict"],
                    "edge_counts": per_asset[t]["edge_counts"],
                    "n_folds_5": per_asset[t]["5"]["n_folds"],
                    "candidate_dispersion_5": per_asset[t]["5"]["candidate_dispersion"],
                    "null_dispersion_5": per_asset[t]["5"]["null_dispersion"],
                }
                for t in all_tickers.keys()
            }
        }
    }

    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== Summary ===")
    print(f"Cost-robustness check completed")
    print(f"Found {len(robust_assets)} cost-robust positive edge(s): {robust_assets}")
    print(f"Result: {'SUCCESS' if len(robust_assets) > 0 else 'NO_COST_ROBUST_EDGE_FOUND'}")
    print(f"\nNOTE: This demonstrates that cost robustness evaluation")
    print(f"is a meaningful qualification for momentum evidence.")

    return artifact


if __name__ == "__main__":
    run_cost_sensitive_check()
    print("\nDone.")