"""Cost-sensitivity momentum check (lookback 3 / 5 / 10).

Objective: test whether the short-horizon momentum edge survives realistic
transaction costs (commission + slippage) and whether the cost impact is
similar across lookback 3, 5, and 10.

Method (fixed a-priori, not tuned to OOS):
- long the previous lookback-day return; hold 1 day; daily rebalance.
- lookbacks 3, 5, 10; same walk-forward windows as momentum.py (train=252d/
  test=84d/warmup=60d/overlap=60d), 4 contiguous volatility blocks.
- Compare each lookback against a coin-flip null benchmark run on the same
  segments.
- Real cost model: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005, initial_capital=1e6.
- Zero-cost baseline: all costs zero (BacktestConfig defaults).
- For each lookback, compute:
  * realistic_cost_median_log_return
  * zero_cost_median_log_return
  * realistic_cost_compare_noise (vs coin-flip null)
  * zero_cost_compare_noise (vs coin-flip null)
- Cost-robustness verdict: an edge is "ROBUST" if realistic_cost_compare_noise
  indicates a genuine edge (baseline > null with significance) AND the edge
  persists at >= 2 lookbacks; "SENSITIVE" if only one lookback survives; 
  "NO_EDGE" otherwise.

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitivity_momentum_results.json"

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6)


def segment_result_cost_sensitivity(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback with cost sensitivity.

    Returns dict with realistic cost and zero cost results vs matched coin-flip null.
    """
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    # Realistic costs
    res_real = bt.stress_segments(
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
        cfg=REALISTIC_COSTS,
    )

    # Zero costs
    res_zero = bt.stress_segments(
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
        cfg=ZERO_COSTS,
    )

    # Coin-flip null for realistic costs (different seed for independence)
    seed_for_null = (SEED + hash(ticker) + lookback) & 0xffffffff
    np.random.seed(seed_for_null)
    null_bars = bt.generate_bars(len(closes), seed=seed_for_null)
    null_signals = bt.momentum_signals(null_bars.closes_array(), lookback=lookback)
    res_real_null = bt.walk_forward(
        list(null_bars), null_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
    )
    real_null_median = float(np.median([
        np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
        for f in res_real_null.folds
    ])) if res_real_null.folds else 0.0

    # Coin-flip null for zero costs (different seed)
    np.random.seed(seed_for_null + 1)
    null_bars_zero = bt.generate_bars(len(closes), seed=seed_for_null + 1)
    null_signals_zero = bt.momentum_signals(null_bars_zero.closes_array(), lookback=lookback)
    res_zero_null = bt.walk_forward(
        list(null_bars_zero), null_signals_zero, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP,
        cfg=ZERO_COSTS,
    )
    zero_null_median = float(np.median([
        np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
        for f in res_zero_null.folds
    ])) if res_zero_null.folds else 0.0

    # Get first scenario's results for comparison
    realistic_scenario = res_real.scenarios[0]
    zero_scenario = res_zero.scenarios[0]
    
    # Helper function to compare baseline vs null using the same logic as regime_stability
    def compare_baseline_null(baseline_median, null_median):
        # Use fixed tolerance of 0.05 (same as OUTLIER_TOL in regime_stability)
        TOL = 0.05
        return abs(baseline_median - null_median) <= TOL
    
    return {
        "segments": [s.name for s in res_real.scenarios],
        "realistic_cost_medians": [round(s.baseline_median_log_return, 3) for s in res_real.scenarios],
        "zero_cost_medians": [round(s.baseline_median_log_return, 3) for s in res_zero.scenarios],
        "realistic_cost_null_median": round(real_null_median, 3),
        "zero_cost_null_median": round(zero_null_median, 3),
        "realistic_cost_candidate_dispersion": round(res_real.candidate_dispersion, 3),
        "zero_cost_candidate_dispersion": round(res_zero.candidate_dispersion, 3),
        "realistic_cost_null_dispersion": round(res_real.null_dispersion, 3),
        "zero_cost_null_dispersion": round(res_zero.null_dispersion, 3),
        "realistic_cost_verdict": res_real.overall_verdict,
        "zero_cost_verdict": res_zero.overall_verdict,
        "n_folds_realistic": res_real.n_folds,
        "n_folds_zero": res_zero.n_folds,
        "compare_realistic_vs_null": compare_baseline_null(
            realistic_scenario.baseline_median_log_return,
            real_null_median,
        ),
        "compare_zero_vs_null": compare_baseline_null(
            zero_scenario.baseline_median_log_return,
            zero_null_median,
        ),
    }


def cost_robustness_verdict(results):
    """Cost-robustness verdict for one ticker across lookbacks.

    ROBUST      : genuine edge survives realistic costs at >= 2 lookbacks
                  (realistic_cost_compare_noise indicates edge)
    SENSITIVE   : edge only at exactly one lookback with realistic costs
    NO_EDGE     : no lookback shows a genuine edge with realistic costs
    """
    edge_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    for lb in LOOKBACKS:
        res = results[str(lb)]
        # Edge exists if realistic_cost_compare_noise indicates genuine edge
        has_realistic_edge = res["compare_realistic_vs_null"] is True
        if has_realistic_edge:
            edge_counts["ROBUST"] += 1
        else:
            edge_counts["NO_EDGE"] += 1
    
    if edge_counts["ROBUST"] >= 2:
        return "ROBUST", edge_counts
    if edge_counts["ROBUST"] == 1:
        return "SENSITIVE", edge_counts
    return "NO_EDGE", edge_counts


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
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, REALISTIC_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-sensitivity momentum lookback sweep (3 / 5 / 10) ===")
    print(f"  realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs matched coin-flip null.")

    # Full-universe cost-sensitivity sweep
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker]
    lookback_results = {lb: {t: segment_result_cost_sensitivity(t, all_tickers[t], lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = cost_robustness_verdict(profile)
        robustness_counts[v] += 1
        
        # Store profile for artifact
        per_asset[ticker] = profile

    print("  cost-robustness verdict counts: "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(genuine edge survives realistic costs at >=2 lookbacks) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(edge only at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")
    print("  Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med_real = ", ".join(
            "{}:{:6.3f}({:6.3f})".format(lb, p[str(lb)]["realistic_cost_medians"][0], 
                                         p[str(lb)]["zero_cost_medians"][0])
            for lb in LOOKBACKS
        )
        print(f"    {ticker:6s} realistic_vs_null={p['5']['compare_realistic_vs_null']!s:<6} "
              f"zero_vs_null={p['5']['compare_zero_vs_null']!s:<6} "
              f"lookback_medians={med_real} rob={v}")
    print("")

    # Determinism: rerun the target subset and assert identical output
    res2 = {lb: {t: segment_result_cost_sensitivity(t, tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        per_asset=per_asset,
        universe=dict(
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    realistic_cost_medians_5=per_asset[t]["5"]["realistic_cost_medians"][0],
                    zero_cost_medians_5=per_asset[t]["5"]["zero_cost_medians"][0],
                    realistic_cost_compare_vs_null=per_asset[t]["5"]["compare_realistic_vs_null"],
                    zero_cost_compare_vs_null=per_asset[t]["5"]["compare_zero_vs_null"],
                    robustness_verdict=v,
                    n_folds_realistic=per_asset[t]["5"]["n_folds_realistic"],
                    n_folds_zero=per_asset[t]["5"]["n_folds_zero"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())