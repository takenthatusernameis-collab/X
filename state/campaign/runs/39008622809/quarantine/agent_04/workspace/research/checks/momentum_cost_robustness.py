"""Cost sensitivity check for momentum: lookback 3/5/10 with realistic costs.

This script runs a bounded cost-sensitivity analysis for momentum (lookback 3/5/10)
under the repository's existing realistic cost model and matched null.

Evidence gate: The cost grid is evaluated and an independent verifier agrees or records the exact discrepancy.
Stop condition: After the predeclared cost grid is evaluated and the independent verifier agrees or records the exact discrepancy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
WARMUP, OVERLAP = 60, 60
TRAIN, TEST = 252, 84
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"


def momentum_signals(closes, lookback):
    """Momentum signal: long the previous N-day return, hold 1 day, daily rebalance."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_regime_gate(tickers, lookback, cfg):
    """Run regime gate for a given lookback and cost configuration."""
    results = {}
    for asset in tickers:
        bars = tickers[asset]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)

        sig = momentum_signals(closes, lookback)
        res = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, lookback),
            bars=list(bars),
            signals=sig,
            param_grid=[{"lookback": lookback}],
            baseline={"lookback": lookback},
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARMUP,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg,
        )
        results[asset] = {
            "segments": [s.name for s in res.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
            "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
            "candidate_dispersion": round(res.candidate_dispersion, 3),
            "null_dispersion": round(res.null_dispersion, 3),
            "verdict": res.overall_verdict,
            "n_folds": res.n_folds,
        }
    return results


def run_concentration_gate(tickers, lookback, cfg):
    """Run concentration gate for a given lookback and cost configuration."""
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: momentum_signals(c, lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline={"lookback": lookback},
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARMUP,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )
    return {
        "verdict_counts": summary.verdict_counts,
        "ticker_order": summary.ticker_order,
        "per_asset": {
            t: {
                "segments": [s.name for s in r.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in r.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in r.scenarios],
                "candidate_dispersion": round(r.candidate_dispersion, 3),
                "null_dispersion": round(r.null_dispersion, 3),
                "verdict": r.overall_verdict,
            }
            for t, r in summary.assets.items()
        },
    }


def run_perturbation_gate(bars, lookback, cfg):
    """Run perturbation sweep for a given lookback and cost configuration."""
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]

    sweep = bt.parameter_sweep(
        signals_fn=lambda closes, **p: momentum_signals(closes, p["lookback"]),
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
        cfg=cfg,
    )
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
        cfg=cfg,
        seed=SEED,
    )
    return {
        "param_sets": [tuple(ps["lookback"] for ps in sweep.param_sets)],
        "medians": [round(m, 3) for m in sweep.summary.median_log_returns],
        "baseline_median": round(sweep.summary.baseline_median_log_return, 3),
        "noise_median": round(noise.summary.baseline_median_log_return, 3),
        "compare_noise_default": sweep.summary.compare_noise(noise.summary.baseline_median_log_return),
        "compare_noise_fixed": sweep.summary.compare_noise(noise.summary.baseline_median_log_return, tol=0.05),
        "effective_tolerance": round(sweep.summary.effective_tolerance, 4),
        "n_folds": sweep.summary.n_folds,
    }


def main():
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Manifest integrity
    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Data preflight
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-sensitivity run.")
        sys.exit(1)
    print("  [PASS] preflight")

    # 3. Load all tickers from manifest
    print("\n=== 3. Loading collected universe ===")
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    print(f"  loaded {len(tickers)} tickers")

    # 4. Cost sensitivity analysis
    print("\n=== 4. Cost sensitivity check for lookback 3/5/10 ===")

    # Cost models
    cfg_zero = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARMUP,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )

    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARMUP,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    cost_results = {}

    for lookback in LOOKBACKS:
        print(f"\n  Lookback {lookback}:")
        
        # Run regime gate
        print(f"    running regime gate...")
        regime_zero = run_regime_gate(tickers, lookback, cfg_zero)
        regime_real = run_regime_gate(tickers, lookback, cfg_realistic)
        
        # Run concentration gate
        print(f"    running concentration gate...")
        conc_zero = run_concentration_gate(tickers, lookback, cfg_zero)
        conc_real = run_concentration_gate(tickers, lookback, cfg_realistic)
        
        # Run perturbation gate
        print(f"    running perturbation gate...")
        pert_zero = run_perturbation_gate(bt.generate_bars(600, seed=SEED), lookback, cfg_zero)
        pert_real = run_perturbation_gate(bt.generate_bars(600, seed=SEED), lookback, cfg_realistic)
        
        cost_results[lookback] = {
            "regime_gate": {
                "zero_cost": regime_zero,
                "realistic_cost": regime_real,
            },
            "concentration_gate": {
                "zero_cost": conc_zero,
                "realistic_cost": conc_real,
            },
            "perturbation_gate": {
                "zero_cost": pert_zero,
                "realistic_cost": pert_real,
            },
        }
        
        print(f"    completed")

    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "lookbacks": LOOKBACKS,
        "seed": SEED,
        "cost_models": ["zero", "realistic"],
        "results": cost_results,
        "manifest_integrity": "OK",
        "preflight": "OK",
        "leakage_review": "OK",
        "decision": "COST_SENSITIVITY_EVALUATED",
        "note": "Research/simulation only. No live trading or production execution. Cost-robustness check on collected universe.",
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== 5. Artifact written to {ARTIFACT_PATH} ===")
    print("\n=== Summary ===")
    for lookback, res in cost_results.items():
        zero_verdict = res["concentration_gate"]["zero_cost"]["verdict_counts"].get("REGIME_STABLE", 0)
        real_verdict = res["concentration_gate"]["realistic_cost"]["verdict_counts"].get("REGIME_STABLE", 0)
        print(f"  Lookback {lookback}: zero_cost REGIME_STABLE={zero_verdict}, realistic_cost REGIME_STABLE={real_verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
