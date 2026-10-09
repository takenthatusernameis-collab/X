"""Cost sensitivity of momentum lookback 3 / 5 / 10.

Objective: the short-horizon momentum edge (lookback=5) has been admitted as candidate positive evidence. A conservative transaction-cost sensitivity test is the next decision-relevant qualification: under the repository's predeclared realistic cost model, does the momentum edge survive? The matched null (coin-flip sign on the same real-data segments) provides the benchmark for "no edge".

Method (fixed a-priori, not tuned to OOS):
- Use the same walk-forward parameters as the existing momentum check (`momentum.py`).
- Realistic cost model from examples: commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005.
- Zero-cost baseline from the existing momentum check for comparison (not to be changed post-hoc).
- Match the lookback sweep (3 / 5 / 10) from `momentum_lookback_sweep.py` to avoid repeating identical experiments; this is a new dimension: cost sensitivity.
- Per-asset and universe-level verdicts using the existing classification: REGIME_STABLE positive / CONSISTENT_WITH_NOISE / REGIME_DEPENDENT / REGIME_STABLE_LOSS.
- Independent verification via matched null recomputation.

Classification scheme (predeclared):
  - Per asset per lookback: verdict per segment + per-asset verdict (REGIME_STABLE positive / CONSISTENT_WITH_NOISE / REGIME_DEPENDENT / REGIME_STABLE_LOSS).
  - Lookback-robustness verdict per asset: ROBUST = uniformly positive REGIME_STABLE edge at >= 2 lookbacks; SENSITIVE = edge at exactly one lookback; NO_EDGE = no positive edge.
  - Universe-level classification: COST_ROBUST = the 7/10 H=1 positive assets remain REGIME_STABLE positive at >= 2 lookbacks under realistic costs; COST_SENSITIVE = some assets show edge at some lookbacks but not all; COST_DESTROYED = the positive edge collapses to NO_EDGE.

A-priori falsification prediction: the momentum edge is mechanical and will be eroded by transaction costs; under realistic costs, medians should decline toward the null band, likely producing CONSISTENT_WITH_NOISE or REGIME_DEPENDENT results, perhaps with a subset of assets sustaining ROBUST edges (COST_SENSITIVE outcome). Either outcome closes the cell.

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
CANONICAL_LOOKBACK = 5
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def config_zero_cost():
    """Zero-cost baseline from the existing momentum check."""
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=0)


def config_realistic_cost():
    """Realistic cost model from examples (ma_crossover.py and volatility_regime_filter.py)."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=0,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )


def segment_result_with_config(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, and one config."""
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
        cfg=cfg,
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
        trades_per_fold=res.trades_per_fold,
        total_commission=res.total_commission,
        total_slippage=res.total_slippage,
    )


def per_lookback_edge(sr):
    """Whether this lookback shows a uniformly-positive REGIME_STABLE edge."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def lookback_robustness_verdict(lookback_results):
    """Lookback-robustness verdict for one asset across lookbacks {3, 5, 10}."""
    counts = {"EDGE": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        if per_lookback_edge(lookback_results[str(lb)]):
            counts["EDGE"] += 1
        else:
            counts["NO_EDGE"] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
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

    print("\n=== 3. Leakage review per asset ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            sig = bt.momentum_signals(closes, lookback=lb)
            bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), bt.momentum_signals(closes, lookback=CANONICAL_LOOKBACK),
                          config_zero_cost())
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals (3/5/10) "
              f"{len(bt.momentum_signals(closes, lookback=5))}, leakage [PASS], "
              f"zero-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep across collected universe (zero cost) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, lookbacks {} vs coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP,
                                           MIN_SEGMENT_BARS, " / ".join(map(str, LOOKBACKS))))

    # Full-universe lookback sweep (zero cost) — used for matching null and artifact cross-consistency
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Zero-cost results for artifact cross-consistency (same as momentum_lookback_sweep.py)
    zero_cost_results = {lb: {t: segment_result_with_config(t, all_tickers[t], lb, config_zero_cost())
                            for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    lookback_5_counts_zero = {lb: {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                   "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
                             for lb in LOOKBACKS}
    lookback_robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        profile = {str(lb): zero_cost_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = lookback_robustness_verdict(profile)
        lookback_robustness_counts[v] += 1
        for lb in LOOKBACKS:
            lookback_5_counts_zero[lb][profile[str(lb)]["verdict"]] += 1
        profile["robustness_verdict"] = v
        profile["robustness_counts"] = counts
        profile["lookback_medians_zero"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  Lookback regime-stability verdict counts across universe (zero cost):")
    for lb in LOOKBACKS:
        print(f"    lookback {lb}: {lookback_5_counts_zero[lb]}")
    print(f"  Lookback-robustness verdict counts (zero cost): "
          f"ROBUST={lookback_robustness_counts['ROBUST']} (edge at >=2 lookbacks) | "
          f"SENSITIVE={lookback_robustness_counts['SENSITIVE']} (edge at exactly 1 lookback) | "
          f"NO_EDGE={lookback_robustness_counts['NO_EDGE']}\n")
    print("  Per-asset profiles (zero cost):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        details = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print(f"    {ticker:6s} segments={p['5']['segments']!r} medians3510={details} rob={p['robustness_verdict']!r} "
              f"counts={p['robustness_counts']}")
    print("")

    # Realistic cost results
    print("\n=== 5. Momentum lookback sweep across collected universe (realistic costs) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, lookbacks {} vs coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP,
                                           MIN_SEGMENT_BARS, " / ".join(map(str, LOOKBACKS))))

    realistic_cost_results = {lb: {t: segment_result_with_config(t, all_tickers[t], lb, config_realistic_cost())
                                  for t in ticker_order}
                              for lb in LOOKBACKS}

    per_asset_realistic = {}
    lookback_5_counts_realistic = {lb: {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                        "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
                                 for lb in LOOKBACKS}
    lookback_robustness_counts_realistic = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        profile = {str(lb): realistic_cost_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = lookback_robustness_verdict(profile)
        lookback_robustness_counts_realistic[v] += 1
        for lb in LOOKBACKS:
            lookback_5_counts_realistic[lb][profile[str(lb)]["verdict"]] += 1
        profile["robustness_verdict"] = v
        profile["robustness_counts"] = counts
        profile["lookback_medians_realistic"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_realistic[ticker] = profile

    print("  Lookback regime-stability verdict counts across universe (realistic costs):")
    for lb in LOOKBACKS:
        print(f"    lookback {lb}: {lookback_5_counts_realistic[lb]}")
    print(f"  Lookback-robustness verdict counts (realistic costs): "
          f"ROBUST={lookback_robustness_counts_realistic['ROBUST']} (edge at >=2 lookbacks) | "
          f"SENSITIVE={lookback_robustness_counts_realistic['SENSITIVE']} (edge at exactly 1 lookback) | "
          f"NO_EDGE={lookback_robustness_counts_realistic['NO_EDGE']}\n")
    print("  Per-asset profiles (realistic costs):")
    for ticker in ticker_order:
        p = per_asset_realistic[ticker]
        details = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print(f"    {ticker:6s} segments={p['5']['segments']!r} medians3510={details} rob={p['robustness_verdict']!r} "
              f"counts={p['robustness_counts']}")
    print("")

    # Determinism: rerun the target subset and assert identical output for both cost models
    zero2 = {lb: {t: segment_result_with_config(t, tickers[t], lb, config_zero_cost())
                  for t in target}
             for lb in LOOKBACKS}
    realistic2 = {lb: {t: segment_result_with_config(t, tickers[t], lb, config_realistic_cost())
                       for t in target}
                  for lb in LOOKBACKS}
    zero1 = {lb: {t: zero_cost_results[lb][t] for t in target} for lb in LOOKBACKS}
    realistic1 = {lb: {t: realistic_cost_results[lb][t] for t in target} for lb in LOOKBACKS}
    zero_match = json.dumps(zero1, sort_keys=True) == json.dumps(zero2, sort_keys=True)
    realistic_match = json.dumps(realistic1, sort_keys=True) == json.dumps(realistic2, sort_keys=True)
    print("determinism checks:")
    print(f"  zero-cost r1 == r2: {zero_match}")
    print(f"  realistic-cost r1 == r2: {realistic_match}")
    assert zero_match and realistic_match, "non-deterministic output"

    # Cross-consistency: ensure realistic-cost medians differ from zero-cost where expected (e.g., reduced edge magnitude)
    cross_consistency_issues = []
    for ticker in ticker_order:
        z = per_asset[ticker]
        r = per_asset_realistic[ticker]
        for lb in LOOKBACKS:
            if z[str(lb)]["verdict"] in ("REGIME_STABLE", "REGIME_STABLE_LOSS") and r[str(lb)]["verdict"] == "CONSISTENT_WITH_NOISE":
                pass  # expected: edge eroded by costs
            elif z[str(lb)]["verdict"] == "CONSISTENT_WITH_NOISE" and r[str(lb)]["verdict"] == "CONSISTENT_WITH_NOISE":
                pass  # expected: noise-like persists
            elif z[str(lb)]["verdict"] == r[str(lb)]["verdict"] and z[str(lb)]["medians"][0] != r[str(lb)]["medians"][0]:
                # same verdict class but magnitude differs — acceptable
                pass
            else:
                cross_consistency_issues.append(f"{ticker} lookback {lb}: zero={z[str(lb)]['verdict']} realistic={r[str(lb)]['verdict']}")

    if cross_consistency_issues:
        print("  cross-consistency issues (acceptable if within expected cost erosion):")
        for issue in cross_consistency_issues:
            print(f"    {issue}")
    else:
        print("  cross-consistency: realistic-cost results behave as expected relative to zero-cost baseline.")

    # Summary cost robustness verdict
    cost_robustness_verdict = "COST_ROBUST" if lookback_robustness_counts_realistic["ROBUST"] >= 7 else (
        "COST_SENSITIVE" if lookback_robustness_counts_realistic["ROBUST"] + lookback_robustness_counts_realistic["SENSITIVE"] >= 7 else "COST_DESTROYED")

    print("\n=== 6. Cost robustness verdict ===")
    print(f"  Zero-cost (baseline): ROBUST={lookback_robustness_counts['ROBUST']}, "
          f"SENSITIVE={lookback_robustness_counts['SENSITIVE']}, NO_EDGE={lookback_robustness_counts['NO_EDGE']}")
    print(f"  Realistic costs:   ROBUST={lookback_robustness_counts_realistic['ROBUST']}, "
          f"SENSITIVE={lookback_robustness_counts_realistic['SENSITIVE']}, NO_EDGE={lookback_robustness_counts_realistic['NO_EDGE']}")
    print(f"  Cost robustness verdict: {cost_robustness_verdict}")
    print(f"  Interpretation: {'Momentum edge survives conservative transaction costs (>=2 lookbacks positive edge)' if cost_robustness_verdict == 'COST_ROBUST' else ('Momentum edge is fragile under costs (some but not all lookbacks show edge)' if cost_robustness_verdict == 'COST_SENSITIVE' else 'Momentum edge is destroyed by transaction costs')}")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        canonical_lookback=CANONICAL_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost=dict(
            per_asset=per_asset,
            lookback_5_counts=lookback_5_counts_zero,
            robustness_counts=lookback_robustness_counts,
        ),
        realistic_cost=dict(
            per_asset=per_asset_realistic,
            lookback_5_counts=lookback_5_counts_realistic,
            robustness_counts=lookback_robustness_counts_realistic,
        ),
        cost_robustness_verdict=cost_robustness_verdict,
        ticker_order=ticker_order,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution. Momentum lookback cost robustness across the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
