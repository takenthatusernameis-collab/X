"""Cost-sensitivity check for short-horizon momentum (lookback 3/5/10).

Background: prior momentum evidence (state/check_artifacts/momentum_results.json)
indicates REGIME_STABLE or REGIME_DEPENDENT performance with zero-cost assumptions.
This check preserves the same signal logic (lookback in days, long previous return,
hold 1 day, daily rebalance) but applies realistic transaction costs to test whether
the edge survives conservative execution friction.

Design (fixed a-priori, not tuned to OOS):
- Lookback: 3 / 5 / 10 days
- Signal: long previous N-day log return; hold 1 day; daily rebalance
- Cost model: commission per trade, per-share commission, fixed and proportional slippage
- Run on the 10-asset collected universe (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, JNJ, XOM)
- Deterministic: use seed 42 for perturbation sweep baseline; match null via coin-flip signal on same segments
- Artifact: JSON for independent verification via fresh walk_forward path
- Research/simulation only. No live trading or production execution.
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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"

# Realistic transaction cost model (conservative, research/simulation only)
# Commission: $0.005 per trade (typical retail brokerage)
# Slippage: $0.05 per share fixed + 1 bps proportional (market impact)
# These costs are applied to every round-trip trade.
COST_MODEL = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.005,
    commission_per_share=0.0,
    slippage_cents=5.0,  # $0.05 per share
    slippage_proportional=0.0001,  # 1 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
)

# Matched null: same lookback windows, same segments, but random signs (coin-flip) for baseline comparison
def null_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        out[i] = bt.Signal(date=i + 1, weight=np.random.choice([-1.0, 1.0]))
    return out


def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_asset_cost(asset, tickers, lookback):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}

    # Momentum with costs
    signals = momentum_signals(closes, lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{asset} momentum: signals {len(signals)} != bars {len(closes)}")

    res = bt.stress_segments(
        signals_fn=lambda c, **p: momentum_signals(c, lookback),
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=COST_MODEL,
    )
    names = [s.name for s in res.scenarios]
    medians = [round(s.baseline_median_log_return, 4) for s in res.scenarios]  # more precision for costs
    results["momentum_with_costs"] = dict(
        segments=names,
        medians=medians,
        candidate_dispersion=round(res.candidate_dispersion, 4),
        null_dispersion=round(res.null_dispersion, 4),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )

    # Matched null (coin-flip) with identical costs
    np.random.seed(SEED + hash(asset) % 10000)
    null_sigs = null_signals(closes, lookback)
    res_null = bt.stress_segments(
        signals_fn=lambda c, **p: null_signals(c, lookback),
        bars=list(bars),
        signals=null_sigs,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=COST_MODEL,
    )
    names_null = [s.name for s in res_null.scenarios]
    medians_null = [round(s.baseline_median_log_return, 4) for s in res_null.scenarios]
    results["matched_null"] = dict(
        segments=names_null,
        medians=medians_null,
        candidate_dispersion=round(res_null.candidate_dispersion, 4),
        null_dispersion=round(res_null.null_dispersion, 4),
        verdict=res_null.overall_verdict,
        n_folds=res_null.n_folds,
    )

    return results


def perturbation_sweep_cost(n_bars=600):
    """Perturbation sweep of lookback 3/5/10 with realistic costs."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACKS[1]),),  # use lookback=5 as baseline
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=lambda c, **p: momentum_signals(c, p["lookback"]),
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
        cfg=COST_MODEL,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", 5),))
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
        cfg=COST_MODEL,
    )
    return dict(
        param_sets=[tuple(ps) for ps in sweep.param_sets],
        medians=[round(m, 4) for m in summary.median_log_returns],
        baseline_median=round(summary.baseline_median_log_return, 4),
        noise_median=round(noise.baseline_median_log_return, 4),
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

    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # Load all collected tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]

    print("\n=== 3. Cost-sensitivity check for lookback 3 / 5 / 10 ===")
    print("Signal: long previous N-day log return, hold 1 day, daily rebalance.")
    print(f"Cost model: commission_per_trade={COST_MODEL.commission_per_trade}, "
          f"slippage_cents={COST_MODEL.slippage_cents}, "
          f"slippage_proportional={COST_MODEL.slippage_proportional}")
    print(f"Windows: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d, "
          f"min {MIN_SEGMENT_BARS} bars/segment, {N_BLOCKS} contiguous volatility blocks.\n")

    results_by_lookback = {}
    for lb in LOOKBACKS:
        print(f"--- Lookback = {lb} days ---")
        results_by_lookback[lb] = {}
        for asset in sorted(tickers.keys()):
            res = run_asset_cost(asset, tickers, lb)
            results_by_lookback[lb][asset] = res
            print(f"  {asset}:")
            for name, r in res.items():
                segs = "[" + ",".join(r["segments"]) + "]"
                meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
                print(f"    {name:20s} segments={segs} medians={meds} verdict={r['verdict']}")

    # Perturbation sweep with costs
    print("\n=== 4. Perturbation sweep (lookback 3 / 5 / 10) with costs ===")
    pers = perturbation_sweep_cost()
    print("  param sets:", pers["param_sets"])
    print("  candidate medians:", pers["medians"])
    print("  baseline (lookback=5): {:+.4f}".format(pers["baseline_median"]))
    print("  coin-flip null baseline: {:+.4f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}"
          .format(pers["compare_noise_fixed"]))

    # Determinism: verify that two independent runs produce identical results
    print("\n=== 5. Determinism check ===")
    # Recreate with a fresh random seed
    np.random.seed(SEED)
    results_by_lookback2 = {}
    for lb in LOOKBACKS:
        results_by_lookback2[lb] = {}
        for asset in sorted(tickers.keys()):
            res = run_asset_cost(asset, tickers, lb)
            results_by_lookback2[lb][asset] = res

    # Compare serialized outputs
    out1 = json.dumps(results_by_lookback, sort_keys=True)
    out2 = json.dumps(results_by_lookback2, sort_keys=True)
    deterministic = out1 == out2
    print(f"  deterministic: {deterministic}")
    if not deterministic:
        print("  WARNING: non-deterministic output with identical costs")
        # Find differences
        import difflib
        diff = list(difflib.unified_diff(out1.splitlines(), out2.splitlines(), lineterm=""))
        for line in diff[:10]:
            print(f"    {line}")
        if len(diff) > 10:
            print(f"    ... ({len(diff) - 10} more lines)")
        assert False, "non-deterministic cost check output"

    # Write artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=LOOKBACKS,
        min_segment_bars=MIN_SEGMENT_BARS,
        n_blocks=N_BLOCKS,
        overlap=OVERLAP,
        cost_model=dict(
            commission_per_trade=COST_MODEL.commission_per_trade,
            commission_per_share=COST_MODEL.commission_per_share,
            slippage_cents=COST_MODEL.slippage_cents,
            slippage_proportional=COST_MODEL.slippage_proportional,
            warmup_periods=COST_MODEL.warmup_periods,
            initial_capital=COST_MODEL.initial_capital,
            target_exposure=COST_MODEL.target_exposure,
        ),
        per_lookback=results_by_lookback,
        perturbation=pers,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. Cost-sensitive momentum edge evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
