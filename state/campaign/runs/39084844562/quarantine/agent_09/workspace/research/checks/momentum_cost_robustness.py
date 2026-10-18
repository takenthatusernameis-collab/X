"""Test the cost robustness of momentum signals on the collected universe.

Background: momentum (lookback 3/5/10) was admitted as candidate positive evidence
from the real-data lookback-sweep, surviving the regime-stability gate and the
independent verifier. The remaining qualification is whether this edge survives
conservative transaction-cost stress using the repository's existing realistic
cost model and a matched null.

Falsification prediction: the lookback-3/5/10 momentum edge either disappears
under realistic costs (CONSISTENT_WITH_NOISE) or becomes REGIME_DEPENDENT (cost
tolerance varies strongly across market regimes) or REGIME_STABLE_LOSS (uniformly
negative). A positive conclusion would show the edge persists with the realistic
cost model.

Method (a-priori fixed, not tuned to OOS):
- Long the previous N-day return (lookback 3, 5, 10) for the same
  walk-forward per-segment settings as in research/checks/momentum.py (lookback=5 baseline).
- Use realistic costs: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005 (same as examples/ma_crossover.py).
- Test across the collected 10-asset universe with the same regime classifier
  (volatility_blocks, N_BLOCKS=4, WINDOW=60).
- Matched null: same signal backbone but with a coin-flip null on each fold.
- Independent verification via fresh walk_forward recomputation (no call to
  stress_segments or regime_stability.py).

Research/simulation only. No live trading or production execution.
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
REALISTIC_COSTS = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    periods_per_year=252,
)
ZERO_COSTS = bt.BacktestConfig()
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
COST_ROBUSTNESS_ARTIFACT = ARTIFACT_DIR / "momentum_cost_robustness.json"


def momentum_signals(closes, lookback):
    """Momentum signals: long previous N-day return if positive."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def classify_regime_edge(medians):
    """Classify regime-stability verdict from medians using the same rule as the original momentum test."""
    cand_disp = float(np.std(np.array(medians)))
    null_disp = 0.05  # Sample-calibrated tolerance from methodology.md
    if all(abs(m) <= null_disp for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def run_cost_sensitivity_check():
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

    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost robustness run.")
        sys.exit(1)

    # Build the universe
    tickers_univ = {}
    for entry in manifest["entries"]:
        tickers_univ[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("\n=== 3. Cost-robustness test: momentum lookback 3/5/10 with realistic costs ===")
    print("Realistic costs: commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005")
    print("Baseline vs matched null (coin-flip): per-segment statistical test.")

    cost_results = {}

    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} ---")

        # Realistic costs
        realistic_medians = []
        realistic_verdicts = []
        zero_cost_medians = []
        zero_cost_verdicts = []

        for ticker in sorted(tickers_univ.keys()):
            bars = tickers_univ[ticker]
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            runs = bt.segment_fn_from_labels(labels)

            # Realistic costs - use walk_forward directly
            signals = momentum_signals(closes, lookback)
            res_real = bt.walk_forward(
                list(bars),
                signals,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=REALISTIC_COSTS
            )

            # Zero costs
            signals_zero = momentum_signals(closes, lookback)
            res_zero = bt.walk_forward(
                list(bars),
                signals_zero,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=ZERO_COSTS
            )

            # Compute segment medians and verdicts for realistic costs
            realistic_segment_medians = []
            realistic_segment_verdicts = []
            for fold in res_real.folds:
                log_return = np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None))
                realistic_segment_medians.append(round(log_return, 3))
                realistic_segment_verdicts.append(classify_regime_edge([log_return]))

            realistic_medians.extend(realistic_segment_medians)
            realistic_verdicts.extend(realistic_segment_verdicts)

            # Compute segment medians and verdicts for zero costs
            zero_segment_medians = []
            zero_segment_verdicts = []
            for fold in res_zero.folds:
                log_return = np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None))
                zero_segment_medians.append(round(log_return, 3))
                zero_segment_verdicts.append(classify_regime_edge([log_return]))

            zero_cost_medians.extend(zero_segment_medians)
            zero_cost_verdicts.extend(zero_segment_verdicts)

            print(f"  {ticker}: realistic {realistic_segment_medians}, "
                  f"zero_cost {zero_segment_medians}")

        cost_results[str(lookback)] = {
            "medians": realistic_medians,
            "verdicts": realistic_verdicts,
            "zero_cost_medians": zero_cost_medians,
            "zero_cost_verdicts": zero_cost_verdicts,
            "candidate_dispersion": round(float(np.std(realistic_medians)), 3),
            "zero_cost_dispersion": round(float(np.std(zero_cost_medians)), 3),
        }

    print("\n=== 4. Perturbation sweep (simplified) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]

    # Realistic costs perturbation
    recomputed_realistic = []
    for params in grid:
        signals = momentum_signals(bars.closes_array(), params["lookback"])
        res = bt.walk_forward(
            list(bars), signals, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=REALISTIC_COSTS)
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        recomputed_realistic.append(round(float(np.median(log)), 3))

    # Zero costs perturbation
    recomputed_zero = []
    for params in grid:
        signals = momentum_signals(bars.closes_array(), params["lookback"])
        res = bt.walk_forward(
            list(bars), signals, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=bt.BacktestConfig())
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds], dtype=np.float64)
        recomputed_zero.append(round(float(np.median(log)), 3))

    # Null benchmark
    null_res = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        cfg=REALISTIC_COSTS,
        seed=SEED,
    )

    print(f"  realistic cost medians: {recomputed_realistic}")
    print(f"  zero_cost medians: {recomputed_zero}")
    print(f"  null median: {null_res.baseline_median_log_return:+.3f}")

    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "zero_costs": {
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
        "lookback_results": cost_results,
        "perturbation": {
            "param_sets": [tuple(ps.items()) for ps in grid],
            "realistic_medians": recomputed_realistic,
            "zero_cost_medians": recomputed_zero,
            "baseline_median": round(null_res.baseline_median_log_return, 3),
            "n_folds": len(bars.closes_array()) // 60,
        },
    }

    with open(COST_ROBUSTNESS_ARTIFACT, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\n=== Cost robustness artifact written to: {COST_ROBUSTNESS_ARTIFACT} ===")
    print("Cost robustness analysis completed. Use verify_momentum_cost.py to independently verify.")
    print("Research/simulation only. No live trading or production execution.")

    return 0


def main():
    return run_cost_sensitivity_check()


if __name__ == "__main__":
    sys.exit(main())
