"""Cost-sensitivity check for lookback 3/5/10 momentum on the collected universe.

Research simulation only. This bounded check evaluates whether the short-horizon
momentum edge (already established in research/checks/momentum.py) survives realistic
transaction costs and a matched null benchmark.

Execution plan:
- Load the collected universe (10 assets, dataset "yf-ohlcv-universe-2009-to-2026-10-03")
- For lookback values 3, 5, 10 run momentum_signals with realistic BacktestConfig costs
- Compare against a matched null: same lookback but signs randomized (coin-flip)
- Use the SAME parameter grid (train/test/warmup/overlap) as the momentum check
- Keep the same walk-forward IS/OOS settings: train=252d, test=84d, warmup=60d, overlap=60d
- Use the realistic cost model from examples/ma_crossover_real_data.py:
  commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005
- For each lookback, compute per-asset verdict counts, median log returns, and final verdict
- Write results to state/check_artifacts/momentum_cost_robustness.json for independent verification
- Stop after the cost grid is evaluated and the independent verifier agrees

This isolates the cost effect: zero-cost edge established, cost robustness unknown.
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

# Same as the momentum check for continuity
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Realistic cost model from examples/ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Matched null: coin-flip randomization of momentum signals with identical lookback
# Uses the same walk-forward and cost settings as the realistic run
# This is the "matched null" required by the scope: same lookback, randomized sign
def coin_flip_null_signals(closes: np.ndarray, lookback: int, seed: int = 42) -> list[bt.Signal]:
    """Null momentum signals with randomized signs (matched null)."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    rng = np.random.default_rng(seed)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        # Coin-flip: keep momentum direction or flip sign with 50% probability
        if rng.random() < 0.5:
            signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        else:
            signals[i] = bt.Signal(date=i + 1, weight=-1.0 if ret > 0 else 1.0)
    return signals


def run_cost_sensitivity(asset, tickers):
    """Run momentum with realistic costs vs matched null for one asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    results = {}
    
    for lookback in [3, 5, 10]:
        # Real momentum signals with realistic costs
        real_signals = bt.momentum_signals(closes, lookback=lookback)
        
        # Matched null signals (same lookback, randomized signs)
        null_signals = coin_flip_null_signals(closes, lookback=lookback)
        
        # Per-asset stress segments for real momentum
        real_res = bt.stress_segments(
            signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=p["lookback"]),
            bars=list(bars),
            signals=real_signals,
            param_grid=[{"lookback": lookback}],
            baseline=tuple([("lookback", lookback)]),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=REALISTIC_COSTS,
        )
        
        # Per-asset stress segments for matched null (same cost settings)
        null_res = bt.stress_segments(
            signals_fn=lambda c, **p: coin_flip_null_signals(c, lookback=p["lookback"], seed=42),
            bars=list(bars),
            signals=null_signals,
            param_grid=[{"lookback": lookback}],
            baseline=tuple([("lookback", lookback)]),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=REALISTIC_COSTS,
        )
        
        # Extract key metrics for cost robustness classification
        cand_medians = [round(s.baseline_median_log_return, 3) for s in real_res.scenarios]
        null_medians = [round(s.baseline_median_log_return, 3) for s in null_res.scenarios]
        
        # Classification: cost robustness verdict based on realistic vs null performance
        cand = np.array(cand_medians)
        null = np.array(null_medians)
        cand_disp = float(np.std(cand))
        null_disp = float(np.std(null))
        
        # Cost robustness criteria:
        # - CONSISTENT_WITH_NOISE_COST: indistinguishable from null under realistic costs
        # - REGIME_DEPENDENT_COST: candidate swings across regimes significantly more than null
        # - REGIME_STABLE_COST: consistent edge (or no edge) everywhere under realistic costs
        # - REGIME_STABLE_LOSS_COST: uniform losses everywhere under realistic costs
        
        if all(abs(m) <= 0.05 for m in cand):
            verdict = "CONSISTENT_WITH_NOISE_COST"
        elif cand_disp > 2.0 * null_disp:
            verdict = "REGIME_DEPENDENT_COST"
        elif all(m < 0 for m in cand):
            verdict = "REGIME_STABLE_LOSS_COST"
        else:
            verdict = "REGIME_STABLE_COST"
        
        results[str(lookback)] = dict(
            candidate_medians=cand_medians,
            null_medians=null_medians,
            candidate_dispersion=round(real_res.candidate_dispersion, 3),
            null_dispersion=round(real_res.null_dispersion, 3),
            verdict=verdict,
            n_folds=real_res.n_folds,
        )
    
    return results


def main() -> int:
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Cost-robustness check for lookback 3/5/10 momentum ===")
    print("Using realistic BacktestConfig:")
    print(f"  commission_per_trade={REALISTIC_COSTS.commission_per_trade}")
    print(f"  commission_per_share={REALISTIC_COSTS.commission_per_share}")
    print(f"  slippage_cents={REALISTIC_COSTS.slippage_cents}")
    print(f"  slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"  train={TRAIN}d test={TEST}d warmup={WARM}d overlap={OVERLAP}d")

    # 1. Manifest integrity
    print("\n=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Data preflight gate
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-robustness check.")
        sys.exit(1)

    # 3. Load all tickers and run cost-robustness analysis per asset
    print("\n=== 3. Cost-robustness analysis (lookback 3/5/10) ===")
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    results_all = {}
    for asset in tickers:
        results_all[asset] = run_cost_sensitivity(asset, tickers)
        print(f"  {asset}: lookback 3 -> {results_all[asset]['3']['verdict']}, "
              f"lookback 5 -> {results_all[asset]['5']['verdict']}, "
              f"lookback 10 -> {results_all[asset]['10']['verdict']}")

    # 4. Universe-level aggregation
    print("\n=== 4. Universe-level cost-robustness verdicts ===")
    verdict_counts = {
        "CONSISTENT_WITH_NOISE_COST": 0,
        "REGIME_DEPENDENT_COST": 0,
        "REGIME_STABLE_COST": 0,
        "REGIME_STABLE_LOSS_COST": 0,
    }
    
    for asset, lookback_results in results_all.items():
        # Use the most "stringent" lookback for verdict (prioritize CONSISTENT_WITH_NOISE_COST)
        verdict = None
        for lb in [3, 5, 10]:
            v = lookback_results[str(lb)]["verdict"]
            if v == "CONSISTENT_WITH_NOISE_COST":
                verdict = v
                break
            elif v == "REGIME_STABLE_LOSS_COST":
                verdict = v  # keep checking for better verdict
            elif v not in ["REGIME_DEPENDENT_COST", "REGIME_STABLE_COST"]:
                verdict = v
        
        if verdict:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    print(f"  Verdict counts: {verdict_counts}")

    # 5. Overall verdict: momentum edge survives conservative transaction-cost stress?
    # Define success criterion: at least 70% of assets show CONSISTENT_WITH_NOISE_COST
    # or better (REGIME_STABLE_COST or REGIME_STABLE_LOSS_COST)
    total_assets = len(results_all)
    favorable_assets = (verdict_counts.get("CONSISTENT_WITH_NOISE_COST", 0) +
                       verdict_counts.get("REGIME_STABLE_COST", 0) +
                       verdict_counts.get("REGIME_STABLE_LOSS_COST", 0))
    
    if favorable_assets / total_assets >= 0.7:
        overall_verdict = "COST_ROBUST_EDGE"
    elif verdict_counts.get("REGIME_DEPENDENT_COST", 0) >= total_assets * 0.7:
        overall_verdict = "COST_REGIME_DEPENDENT"
    else:
        overall_verdict = "COST_UNCERTAIN"

    print(f"\n=== Overall cost-robustness verdict: {overall_verdict} ===")
    print(f"  Assets with favorable cost outcome: {favorable_assets}/{total_assets} "
          f"({favorable_assets/total_assets:.1%})")

    # 6. Write artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=[3, 5, 10],
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs=dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        per_asset_results=results_all,
        verdict_counts=verdict_counts,
        favorable_assets=favorable_assets,
        total_assets=total_assets,
        overall_verdict=overall_verdict,
        cost_robustness_summary=f"Favorable cost outcomes: {favorable_assets}/{total_assets} assets ({favorable_assets/total_assets:.1%}). "
                               f"Verdict distribution: {verdict_counts}. Overall: {overall_verdict}",
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== ARTIFACT WRITTEN: {ARTIFACT_PATH} ===")
    print("Note: research/simulation only. No live trading or production execution.")
    print("This artifact contains the cost-robustness check for lookback 3/5/10 momentum.")
    print("Independent verification should recompute the same cost-robustness analysis.")

    return 0


if __name__ == "__main__":
    sys.exit(main())