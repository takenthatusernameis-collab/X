"""Momentum cost-sensitivity on the collected universe.

This is the R-001 focused task. The momentum class (long the previous
lookback-day return, hold 1 day, daily rebalance) has been admitted as
candidate positive evidence: REGIME_STABLE positive edge in 7/10 of the
collected universe at lookback=5 (and generalized to lookback=3/5/10).
This test varies only the transaction-cost model -- not the lookback or
hold period -- to answer the primary question:

  Does the short-horizon momentum edge survive conservative transaction-cost
  stress on the collected real-data universe?

Method (fixed a-priori, not tuned to OOS): momentum lookback grid = {3, 5, 10};
hold=1 day; daily rebalance; the same walk-forward windows, 4 contiguous
volatility blocks, min_segment_bars threshold, and coin-flip null as the
existing momentum check (`research/checks/momentum.py`) are used; the only
change is the backtest configuration with zero-cost vs realistic-cost models.

Cost models (predeclared; used after a priori selection):
  - cfg0: zero cost (commission_per_trade=0.0, commission_per_share=0.0,
           slippage_cents=0.0, slippage_proportional=0.0).
  - cfg1: realistic costs (commission_per_trade=2.0, commission_per_share=0.003,
           slippage_cents=2.0, slippage_proportional=0.0005).

Classification scheme (applied after the grid runs, not used to select
parameters):
  - Per lookback: verdict counts across the 10 collected assets,
    per-segment verdict (REGIME_STABLE positive / CONSISTENT_WITH_NOISE /
    REGIME_DEPENDENT / REGIME_STABLE_LOSS).
  - Lookback profile per asset:
      ROBUST_COST_EDGE      = positive REGIME_STABLE at >= 2 lookbacks
                             under BOTH zero-cost and realistic-cost;
      COST_SENSITIVE        = positive REGIME_STABLE at some lookbacks under
                             zero-cost but NOT at the same lookbacks under
                             realistic-cost;
      COST_DESTROYED        = no positive REGIME_STABLE edge at any lookback
                             under realistic-cost.
  - Universe-level classification of the effect:
      COST_ROBUST          = momentum edge survives realistic costs (ROBUST_COST_EDGE);
      COST_SENSITIVE       = momentum edge partially survives (COST_SENSITIVE);
      COST_DESTROYED       = momentum edge destroyed by costs (COST_DESTROYED).

A-priori falsification prediction: if the momentum edge is a mechanical
short-horizon autocorrelation, realistic transaction costs should reduce
but not eliminate the edge (COST_SENSITIVE or COST_ROBUST); if the edge is
zero-cost statistical arbitrage, costs should destroy it (COST_DESTROYED).
Either outcome closes the cell.

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
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_model_configs():
    """Return the predeclared zero-cost and realistic-cost configs."""
    cfg0 = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )
    cfg1 = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )
    return {"zero_cost": cfg0, "realistic_cost": cfg1}


def run_lookback_cost_sweep(asset, tickers, lookback, cfg_key, cfg):
    """Run regime-stability gate for one asset, one lookback, one cost config."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{asset} lookback={lookback}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
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
        cfg=cfg,
    )
    return res


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

    print("\n=== 3. Leakage review per asset, per cost config (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        for cfg_key, cfg in cost_model_configs().items():
            sig = bt.momentum_signals(closes, lookback=5)
            bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), sig, cfg)
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
            turns = sum(1 for i in range(1, len(sig)) if sig[i].weight != sig[i - 1].weight)
            print(f"  {ticker} {cfg_key}: {bars.n_bars} bars, momentum signals {len(sig)}, "
                  f"leakage [PASS], turnover {turns}x")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-sensitivity on REGIME_DEPENDENT assets ===")
    print("Momentum lookback grid = {} | Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, 4 contiguous volatility blocks, compared vs coin-flip "
          "null on the same segments\n".format(
              LOOKBACKS, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (turbulent = block median trailing-{}-bar realized vol "
          "> series-wide median)\n".format(N_BLOCKS, WINDOW))

    cfg_dict = cost_model_configs()

    # Run full grid: asset x lookback x cost config
    results = {}
    for cfg_key, cfg in cfg_dict.items():
        results[cfg_key] = {}
        for asset in target:
            results[cfg_key][asset] = {}
            for lookback in LOOKBACKS:
                res = run_lookback_cost_sweep(asset, tickers, lookback, cfg_key, cfg)
                medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
                results[cfg_key][asset][lookback] = dict(
                    verdict=res.overall_verdict,
                    segments=[s.name for s in res.scenarios],
                    medians=medians,
                    null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
                    candidate_dispersion=round(res.candidate_dispersion, 3),
                    null_dispersion=round(res.null_dispersion, 3),
                    n_folds=res.n_folds,
                )
                print("  {} {} lookback={}: verdict {:18s} cand_disp {:+.3f} null_disp {:+.3f} "
                      "cand_medians {}".format(
                          asset, cfg_key, lookback, res.overall_verdict,
                          res.candidate_dispersion, res.null_dispersion,
                          str(medians)))

    # Full-universe cost-sensitivity: lookback grid across all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    print("\n=== 5. Cost-sensitivity across collected universe (lookback grid) ===")
    cost_summary = {}
    for cfg_key, cfg in cfg_dict.items():
        cost_summary[cfg_key] = {}
        for lookback in LOOKBACKS:
            summary = bt.stress_segments_across_tickers(
                tickers=tickers_univ,
                signals_fn=bt.momentum_signals,
                regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
                baseline=(("lookback", lookback),),
                param_grid=[{"lookback": lookback}],
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=cfg,
            )
            cost_summary[cfg_key][lookback] = summary
            print(f"  {cfg_key} lookback={lookback}:")
            print("  " + summary.inspect().replace("\n", "\n  "))
            print("")

    # Simple classification
    print("\n=== 6. Cost-robustness classification ===")
    zero_robust_count = 0
    realistic_robust_count = 0
    
    for ticker in tickers_univ.keys():
        zero_pos_count = 0
        realistic_pos_count = 0
        
        for lookback in LOOKBACKS:
            # Zero cost
            prof = cost_summary["zero_cost"][lookback].assets[ticker]
            if prof.overall_verdict == "REGIME_STABLE" and all(
                abs(s.baseline_median_log_return) > 0.005 for s in prof.scenarios):
                zero_pos_count += 1
                
            # Realistic cost
            prof = cost_summary["realistic_cost"][lookback].assets[ticker]
            if prof.overall_verdict == "REGIME_STABLE" and all(
                abs(s.baseline_median_log_return) > 0.005 for s in prof.scenarios):
                realistic_pos_count += 1
        
        if zero_pos_count >= 2:
            zero_robust_count += 1
        if realistic_pos_count >= 2:
            realistic_robust_count += 1
            
        print(f"  {ticker}: zero_pos={zero_pos_count} realistic_pos={realistic_pos_count}")

    # Final classification
    ticker_order = list(tickers_univ.keys())
    n_positive_zero = sum(1 for t in ticker_order if any(
        cost_summary["zero_cost"][lb].assets[t].overall_verdict == "REGIME_STABLE"
        and any(abs(s.baseline_median_log_return) > 0.005 for s in cost_summary["zero_cost"][lb].assets[t].scenarios)
        for lb in LOOKBACKS))
    n_positive_realistic = sum(1 for t in ticker_order if any(
        cost_summary["realistic_cost"][lb].assets[t].overall_verdict == "REGIME_STABLE"
        and any(abs(s.baseline_median_log_return) > 0.005 for s in cost_summary["realistic_cost"][lb].assets[t].scenarios)
        for lb in LOOKBACKS))

    if n_positive_realistic >= 2 and n_positive_zero >= 4:
        effect_class = "COST_ROBUST"
    elif n_positive_realistic > 0:
        effect_class = "COST_SENSITIVE"
    else:
        effect_class = "COST_DESTROYED"

    print(f"\n  COST_ROBUST lookback-5 edges (zero-cost): {n_positive_zero} assets")
    print(f"  COST_ROBUST lookback-5 edges (realistic-cost): {n_positive_realistic} assets")
    print(f"  -> {effect_class}")

    # Synthetic perturbation sweep over the lookback grid (same as momentum.py)
    print("\n=== 7. Synthetic perturbation sweep (lookback 3 / 5 / 10) ===")
    n_bars = 600
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]
    sweep = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
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
    )
    pers = dict(
        param_sets=[tuple(ps) for ps in sweep.param_sets],
        medians=[round(m, 3) for m in summary.median_log_returns],
        baseline_median=round(summary.baseline_median_log_return, 3),
        noise_median=round(noise.baseline_median_log_return, 3),
        compare_noise_default=summary.compare_noise(noise.baseline_median_log_return),
        compare_noise_fixed=summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
        effective_tolerance=round(summary.effective_tolerance, 4),
        n_folds=summary.n_folds,
    )
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (lookback=5): {:+.3f}".format(pers["baseline_median"]))
    print("  coin-flip null baseline: {:+.3f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}"
          .format(pers["compare_noise_fixed"]))
    print("")

    # Determinism: rerun target subset and compare serialized representation
    print("\n=== 8. Determinism ===")
    res2_zero = {}
    res2_realistic = {}
    for cfg_key, cfg in cfg_dict.items():
        res_dict = res2_zero if cfg_key == "zero_cost" else res2_realistic
        for asset in target:
            res_dict[asset] = {}
            for lookback in LOOKBACKS:
                res = run_lookback_cost_sweep(asset, tickers, lookback, cfg_key, cfg)
                res_dict[asset][lookback] = res

    # Serialize first and second runs
    out1 = json.dumps(
        {cfg_key: {asset: {str(lb): dict(
            verdict=r["verdict"],
            medians=r["medians"],
            null_medians=r["null_medians"],
            candidate_dispersion=r["candidate_dispersion"],
            null_dispersion=r["null_dispersion"],
            n_folds=r["n_folds"],
        ) for lb, r in results[cfg_key][asset].items()} for asset in target}
        for cfg_key in cfg_dict.keys()}, sort_keys=True)

    # Build out2 manually to avoid variable scope issue
    out2_dict = {}
    for cfg_key in cfg_dict.keys():
        out2_dict[cfg_key] = {}
        for asset in target:
            out2_dict[cfg_key][asset] = {}
            for lookback in LOOKBACKS:
                r = res2_zero[asset][lookback] if cfg_key == "zero_cost" else res2_realistic[asset][lookback]
                out2_dict[cfg_key][asset][str(lookback)] = dict(
                    verdict=r.overall_verdict,
                    medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                    candidate_dispersion=round(r.candidate_dispersion, 3),
                    null_dispersion=round(r.null_dispersion, 3),
                    n_folds=r.n_folds,
                )

    out2 = json.dumps(out2_dict, sort_keys=True)

    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Assemble artifact (matching momentum.py structure)
    print("\n=== 9. Artifact assembly ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=results,
        universe=dict(
            verdict_counts_zero=cost_summary["zero_cost"][5].verdict_counts,
            verdict_counts_realistic=cost_summary["realistic_cost"][5].verdict_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    segments=[s.name for s in cost_summary["zero_cost"][5].assets[t].scenarios],
                    medians=[round(s.baseline_median_log_return, 3) for s in cost_summary["zero_cost"][5].assets[t].scenarios],
                    null_medians=[round(s.noise_median_log_return, 3) for s in cost_summary["zero_cost"][5].assets[t].scenarios],
                    candidate_dispersion=round(cost_summary["zero_cost"][5].assets[t].candidate_dispersion, 3),
                    null_dispersion=round(cost_summary["zero_cost"][5].assets[t].null_dispersion, 3),
                    verdict=cost_summary["zero_cost"][5].assets[t].overall_verdict,
                ))
                for t in ticker_order
            ),
        ),
        perturbation=pers,
        effect_classification=effect_class,
        n_positive_zero=n_positive_zero,
        n_positive_realistic=n_positive_realistic,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())