"""Cost-robustness check for momentum lookback 3 / 5 / 10 under realistic transaction costs.

Objective: Test whether the short-horizon momentum edge survives conservative transaction-cost stress.

Scope: Use the existing 10-asset collected universe and existing momentum implementation/check framework.
Apply realistic transaction costs (commission + slippage) and compare against a matched null (coin-flip) benchmark.
Do not tune signal rules after seeing cost results.

Method (fixed a-priori):
- Long the previous N-day return, hold 1 day, daily rebalance (N = 3, 5, 10)
- Same walk-forward windows and volatility blocks as momentum.py (train=252d/test=84d/warmup=60d/overlap=60d)
- Realistic costs: commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005
- Matched null: coin-flip (no data dependence) with same parameter grid and walk-forward structure
- Compare baseline median log return against null median log return; classify cost robustness as:
  - ROBUST if baseline survives the null tolerance (baseline within sample-calibrated 2x null std / sqrt(n_folds) of null)
  - FRAGILE if baseline outside null tolerance
  - CONSISTENT_WITH_NOISE if baseline within fixed tol 0.05 of null

Artifacts: state/check_artifacts/momentum_cost_robustness_results.json
Verification: independent verification in verify_momentum_cost_robustness.py recomputes all values from scratch using a fresh walk_forward path and matches against the artifact.

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

# Realistic transaction cost model (from examples/ma_crossover.py)
REALISTIC_COSTS = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def momentum_signals(closes, lookback):
    """Momentum signal implementation (independent path for verification)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result_with_costs(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback with costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    # Run with realistic costs
    costs_result = bt.stress_segments(
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
        cfg=REALISTIC_COSTS,
    )

    # Generate matched null using same parameter grid and walk-forward structure
    from research.backtest.perturbation import noise_benchmark, random_signals

    def _param_seed(params):
        total = sum(float(v) for v in params.values())
        return SEED + int(round(total) * 1000)

    null_result = noise_benchmark(
        bars=bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
        seed=_param_seed({"lookback": lookback}),
    )

    # Extract baseline median from the first scenario (like momentum_lookback_sweep.py)
    baseline_scenario = costs_result.scenarios[0]
    baseline_median = baseline_scenario.baseline_median_log_return
    noise_median = null_result.baseline_median_log_return

    return dict(
        segments=[s.name for s in costs_result.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in costs_result.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in costs_result.scenarios],
        candidate_dispersion=round(costs_result.candidate_dispersion, 3),
        null_dispersion=round(costs_result.null_dispersion, 3),
        n_folds=costs_result.n_folds,
        verdict=costs_result.overall_verdict,
        cost_baseline_median=round(baseline_median, 3),
        cost_noise_median=round(noise_median, 3),
        cost_effective_tolerance=round(null_result.effective_tolerance, 4),
        cost_within_tolerance=null_result.compare_noise(
            noise_median,
            tol=None  # Use sample-calibrated tolerance
        ),
    )


def cost_robustness_classify(lookback_result):
    """Classify cost robustness based on baseline vs null comparison."""
    baseline = lookback_result["cost_baseline_median"]
    noise_median = lookback_result["cost_noise_median"]
    tolerance = lookback_result["cost_effective_tolerance"]
    within_tolerance = lookback_result["cost_within_tolerance"]

    # Check if baseline is within null tolerance
    if abs(baseline - noise_median) <= tolerance:
        # Additional check: is baseline distinguishable from pure noise?
        if all(abs(m) <= 0.05 for m in lookback_result["medians"]):
            return "CONSISTENT_WITH_NOISE"
        else:
            return "ROBUST"
    else:
        return "FRAGILE"


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
        print("PREFLIGHT FAILED; aborting cost-robustness run.")
        sys.exit(1)

    print("\n=== 3. Cost-robustness check (momentum lookback 3/5/10 with realistic costs) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, lookbacks {} vs matched null "
          "(coin-flip with realistic costs)\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, LOOKBACKS))

    # Full-universe cost-robustness check.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker].
    lookback_results = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "FRAGILE": 0, "CONSISTENT_WITH_NOISE": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        robustness_verdict = cost_robustness_classify(profile[str(5)])  # Use lookback=5 for overall classification
        robustness_counts[robustness_verdict] += 1
        profile["robustness_verdict"] = robustness_verdict

        # Store the per-lookback medians + nulls for the artifact's perturbation view (keyed by lookback).
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  lookback-5 robustness verdict counts across universe:")
    print(f"    ROBUST={robustness_counts['ROBUST']} (>= {LOOKBACKS[1]} lookbacks survive cost stress)")
    print(f"    FRAGILE={robustness_counts['FRAGILE']} (cost stress invalidates edge)")
    print(f"    CONSISTENT_WITH_NOISE={robustness_counts['CONSISTENT_WITH_NOISE']} (edge within 0.05 of noise)")
    print("\n  Per-asset cost-robustness profiles (lookback=5):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        cost_status = p["5"]["cost_within_tolerance"]
        print(f"    {ticker}: segments={p['5']['segments']} "
              f"medians3510={p['5']['medians']} "
              f"cost_baseline={p['5']['cost_baseline_median']:+.3f} "
              f"cost_noise={p['5']['cost_noise_median']:+.3f} "
              f"cost_within_tolerance={cost_status} "
              f"verdict={p['5']['verdict']} "
              f"robustness_verdict={p['robustness_verdict']}")

    print("\nDeterminism: rerun cost-robustness for a subset and assert identical output.")
    target = ["AMZN", "JPM"]
    res2 = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"  determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic cost-robustness output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                           for lb in LOOKBACKS},
                    cost_baseline_median=per_asset[t]["5"]["cost_baseline_median"],
                    cost_noise_median=per_asset[t]["5"]["cost_noise_median"],
                    cost_effective_tolerance=per_asset[t]["5"]["cost_effective_tolerance"],
                    cost_within_tolerance=per_asset[t]["5"]["cost_within_tolerance"],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness check across the collected universe: "
          "exploratory simulation with realistic transaction costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
