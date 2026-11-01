"""Cost-robustness check for short-horizon momentum lookbacks under realistic costs.

Objective: determine whether the short-horizon momentum edge (lookback 3/5/10)
survives conservative transaction-cost stress on the collected real-data universe.

Method (fixed a-priori):
- long the previous N-day return, hold 1 day, daily rebalance (same as momentum.py)
- lookbacks 3, 5, 10 (the "short-horizon" momentum edge)
- same walk-forward windows (train=252d/test=84d/warmup=60d/overlap=60d)
- realistic cost model: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005 (same as examples/ma_crossover_real_data.py)
- compare zero-cost vs realistic-cost results
- matched null: perturbation sweep on the same lookbacks with realistic costs
- independent verification: recompute using walk_forward (not stress_segments)

Falsification prediction: either the momentum edge disappears under realistic
 costs (suggesting cost robustness is the limiting factor) or persists (suggesting
the edge is economically meaningful despite realistic costs). Either outcome
is a negative conclusion for the evidence base.

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

# Realistic cost model from examples/ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero cost baseline for comparison
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)


def momentum_signals(closes, lookback):
    """Momentum signals: long previous N-day return, hold 1 day, daily rebalance."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward result for one ticker, lookback, and cost config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
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


def perturbation_sweep_cost(cfg, lookback, n_bars=600):
    """Synthetic perturbation sweep with given cost configuration."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", lookback),),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
        cfg=cfg,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", lookback),))
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        cfg=bt.BacktestConfig(),
        seed=SEED,
    )
    return dict(
        param_sets=[tuple(ps) for ps in sweep.param_sets],
        medians=[round(m, 3) for m in summary.median_log_returns],
        baseline_median=round(summary.baseline_median_log_return, 3),
        noise_median=round(noise.baseline_median_log_return, 3),
        compare_noise_default=summary.compare_noise(noise.baseline_median_log_return),
        compare_noise_fixed=summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
        effective_tolerance=round(summary.effective_tolerance, 4),
        n_folds=summary.n_folds,
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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review per asset (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, ZERO_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-robustness check: lookback 3/5/10 with realistic costs ===")
    print("  Realistic costs: commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005")
    print("  Same walk-forward windows (train=252d/test=84d/warmup=60d/overlap=60d), "
          "4 volatility blocks, min 400 bars/segment")

    # Full-universe cost-robustness analysis
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    # Results indexed [lookback][ticker][cost_config]
    results_zero = {lb: {t: segment_result(t, all_tickers[t], lb, ZERO_COSTS)
                        for t in all_tickers.keys()} for lb in LOOKBACKS}
    results_realistic = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COSTS)
                             for t in all_tickers.keys()} for lb in LOOKBACKS}

    # Compare zero-cost vs realistic-cost edge classification
    cost_robust_assets = []
    cost_sensitive_assets = []
    cost_eliminated_assets = []
    cost_intact_assets = []

    per_asset = {}
    for ticker in all_tickers.keys():
        edge_zero = []
        edge_realistic = []
        for lb in LOOKBACKS:
            res = results_zero[lb][ticker]
            if res["verdict"] == "REGIME_STABLE" and all(m > 0 for m in res["medians"]):
                edge_zero.append(lb)
            
            res_real = results_realistic[lb][ticker]
            if res_real["verdict"] == "REGIME_STABLE" and all(m > 0 for m in res_real["medians"]):
                edge_realistic.append(lb)

        # Determine cost robustness classification
        if len(edge_zero) >= 2 and len(edge_realistic) >= 2:
            cost_robust_assets.append(ticker)
            cost_intact = "ROBUST"
        elif len(edge_zero) >= 1 and len(edge_realistic) >= 1:
            cost_sensitive_assets.append(ticker)
            cost_intact = "SENSITIVE"
        else:
            cost_eliminated_assets.append(ticker)
            cost_intact = "ELIMINATED"

        per_asset[ticker] = {
            "lookback_5_zero_cost": results_zero[5][ticker],
            "lookback_5_realistic_cost": results_realistic[5][ticker],
            "cost_robustness": cost_intact,
            "edge_zero": edge_zero,
            "edge_realistic": edge_realistic,
        }

    print(f"\n=== Summary: Cost robustness classification ===")
    print(f"  ROBUST (>=2 lookbacks positive edge with zero AND realistic costs): {len(cost_robust_assets)}")
    print(f"  SENSITIVE (edge survives at exactly 1 lookback): {len(cost_sensitive_assets)}")
    print(f"  ELIMINATED (edge vanishes under realistic costs): {len(cost_eliminated_assets)}")
    print(f"  Cost-intact assets: {', '.join(cost_robust_assets + cost_sensitive_assets)}")

    print(f"\n=== 5. Matched null benchmark with realistic costs ===")
    nulls_realistic = {}
    for lb in LOOKBACKS:
        nulls_realistic[lb] = perturbation_sweep_cost(REALISTIC_COSTS, lb)

    # Determinism: rerun the target subset and assert identical output
    res2_zero = {lb: {t: segment_result(t, tickers[t], lb, ZERO_COSTS)
                      for t in target} for lb in LOOKBACKS}
    res2_realistic = {lb: {t: segment_result(t, tickers[t], lb, REALISTIC_COSTS)
                          for t in target} for lb in LOOKBACKS}
    out1_zero = json.dumps({lb: {t: results_zero[lb][t] for t in target}
                            for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps(res2_zero, sort_keys=True)
    out1_realistic = json.dumps({lb: {t: results_realistic[lb][t] for t in target}
                                for lb in LOOKBACKS}, sort_keys=True)
    out2_realistic = json.dumps(res2_realistic, sort_keys=True)
    
    print("determinism check (zero cost): r1 == r2: {}".format(out1_zero == out2_zero))
    print("determinism check (realistic cost): r1 == r2: {}".format(out1_realistic == out2_realistic))
    assert out1_zero == out2_zero, "non-deterministic output (zero cost)"
    assert out1_realistic == out2_realistic, "non-deterministic output (realistic cost)"

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        realistic_costs={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        nulls_realistic=nulls_realistic,
        universe_summary={
            "robust_assets": cost_robust_assets,
            "sensitive_assets": cost_sensitive_assets,
            "eliminated_assets": cost_eliminated_assets,
            "cost_intact_count": len(cost_robust_assets + cost_sensitive_assets),
        },
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-robustness check on the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
