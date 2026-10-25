"""Bounded cost-sensitivity check for lookback 3/5/10 momentum.

Objective: Test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe.

Evidence gate: The candidate positive momentum evidence is short-horizon and
mechanical; cost robustness remains decision-relevant.

This bounded test uses the existing 10-asset collected universe, the existing
momentum implementation, and the repository's existing realistic cost model
and matched null.

Out-of-scope: do not tune signal rules after seeing cost results, expand the
universe, or introduce leverage.
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
COST_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def create_cost_configs(warmup):
    """Return zero-cost and realistic-cost BacktestConfig variants."""
    cfg0 = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
        warmup_periods=warmup,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )
    cfg1 = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
        warmup_periods=warmup,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )
    return {"zero_cost": cfg0, "realistic_costs": cfg1}


def segment_result(ticker, bars, lookback, cfg_key, cfg_map):
    """Walk-forward regime-stability result for one ticker, one lookback, one cost scenario."""
    cfg = cfg_map[cfg_key]
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
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    return dict(
        cfg_key=cfg_key,
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def lookback_verdict(sr):
    """Verdict for one lookback's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset, per cost scenario."""
    # lookback_results structure: {lookback_period: ticker_result_dict}
    # where ticker_result_dict has keys: cfg_key, segments, medians, null_medians, etc.
    result = {}
    
    # Group by cost scenario
    for cfg_key in ["zero_cost", "realistic_costs"]:
        counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            lb_key = str(lb)
            if lb_key in lookback_results:
                ticker_result = lookback_results[lb_key]
                if ticker_result.get("cfg_key") == cfg_key:
                    counts[lookback_verdict(ticker_result)] += 1
        
        result[cfg_key] = {
            "_edge_counts": counts,
            "_robustness_verdict": (
                "ROBUST" if counts["EDGE"] >= 2 else
                "SENSITIVE" if counts["EDGE"] == 1 else
                "NO_EDGE"
            )
        }
    
    return result


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
        print("PREFLIGHT FAILED; aborting cost-sensitivity run.")
        sys.exit(1)

    print("\n=== 3. Cost-sensitivity configuration ===")
    cost_configs = create_cost_configs(WARM)
    print("  Cost scenarios:")
    for cfg_key, cfg in cost_configs.items():
        print(f"    {cfg_key}: commission_trade=${cfg.commission_per_trade:.1f}, "
              f"commission_share=${cfg.commission_per_share:.4f}/share, "
              f"slippage_cents={cfg.slippage_cents}, slippage_proportional={cfg.slippage_proportional}")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe, per cost scenario ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost-sensitive lookback sweep.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [cost_scenario][lookback][ticker]
    cost_lookback_results = {}
    for cfg_key in cost_configs:
        cost_lookback_results[cfg_key] = {}
        for lb in LOOKBACKS:
            cost_lookback_results[cfg_key][str(lb)] = {}
            for ticker in ticker_order:
                cost_lookback_results[cfg_key][str(lb)][ticker] = segment_result(
                    ticker, all_tickers[ticker], lb, cfg_key, cost_configs)

    # Analysis per asset and per cost scenario.
    per_asset = {}
    for cfg_key in cost_lookback_results:
        per_asset[cfg_key] = {}
        for ticker in ticker_order:
            profile = {str(lb): cost_lookback_results[cfg_key][str(lb)][ticker] for lb in LOOKBACKS}
            robustness_result = robustness_verdict(profile)
            # Add robustness verdict to each ticker's profile data
            per_asset[cfg_key][ticker] = profile.copy()
            per_asset[cfg_key][ticker]["_edge_counts"] = robustness_result[cfg_key]["_edge_counts"]
            per_asset[cfg_key][ticker]["_robustness_verdict"] = robustness_result[cfg_key]["_robustness_verdict"]

    # Summary statistics per cost scenario.
    print("\n=== Cost Sensitivity Results Summary ===")
    for cfg_key in cost_lookback_results:
        print(f"\nCost scenario: {cfg_key}")
        print("  Per-asset lookback-robustness verdict counts:")
        robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        for ticker in ticker_order:
            verdict = per_asset[cfg_key][ticker]["_robustness_verdict"]
            robustness_counts[verdict] += 1
        print(f"    ROBUST (>=2 lookbacks positive edge): {robustness_counts['ROBUST']}")
        print(f"    SENSITIVE (edge at exactly one lookback): {robustness_counts['SENSITIVE']}")
        print(f"    NO_EDGE (no lookback positive edge): {robustness_counts['NO_EDGE']}")

    # Comparison between cost scenarios.
    print("\n=== Cost Sensitivity Analysis ===")
    zero_robust = sum(1 for t in ticker_order if per_asset["zero_cost"][t]["_robustness_verdict"] in ["ROBUST", "SENSITIVE"])
    realistic_robust = sum(1 for t in ticker_order if per_asset["realistic_costs"][t]["_robustness_verdict"] in ["ROBUST", "SENSITIVE"])
    
    print(f"  Assets with any momentum edge (zero cost): {zero_robust}/{len(ticker_order)}")
    print(f"  Assets with any momentum edge (realistic costs): {realistic_robust}/{len(ticker_order)}")
    print(f"  Edge survival rate: {realistic_robust/max(zero_robust, 1):.1%} ({realistic_robust}/{max(zero_robust, 1)})")

    # Determinism: rerun the target subset and assert identical output.
    print("\n=== Determinism check ===")
    target = ["AAPL", "JPM"]
    cfg_determ = {k: create_cost_configs(WARM)[k] for k in ["zero_cost", "realistic_costs"]}
    res2 = {}
    for cfg_key in cfg_determ:
        res2[cfg_key] = {}
        for lb in LOOKBACKS:
            res2[cfg_key][str(lb)] = {}
            for ticker in target:
                res2[cfg_key][str(lb)][ticker] = segment_result(
                    ticker, all_tickers[ticker], lb, cfg_key, cfg_determ)
    out1 = json.dumps(cost_lookback_results, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact that the independent verifier file recomputes
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_configs=dict(
            zero_cost=dict(commission_per_trade=cost_configs["zero_cost"].commission_per_trade,
                          commission_per_share=cost_configs["zero_cost"].commission_per_share,
                          slippage_cents=cost_configs["zero_cost"].slippage_cents,
                          slippage_proportional=cost_configs["zero_cost"].slippage_proportional),
            realistic_costs=dict(commission_per_trade=cost_configs["realistic_costs"].commission_per_trade,
                               commission_per_share=cost_configs["realistic_costs"].commission_per_share,
                               slippage_cents=cost_configs["realistic_costs"].slippage_cents,
                               slippage_proportional=cost_configs["realistic_costs"].slippage_proportional),
        ),
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_5_verdict=per_asset["zero_cost"][t]["5"]["verdict"],
                    lookback_5_medians=per_asset["zero_cost"][t]["5"]["medians"],
                    edge_counts_zero=per_asset["zero_cost"][t]["_edge_counts"],
                    robustness_verdict_zero=per_asset["zero_cost"][t]["_robustness_verdict"],
                    edge_counts_realistic=per_asset["realistic_costs"][t]["_edge_counts"],
                    robustness_verdict_realistic=per_asset["realistic_costs"][t]["_robustness_verdict"],
                    n_folds_5=per_asset["zero_cost"][t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(COST_ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(COST_ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())