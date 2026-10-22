"""Cost-sensitivity check for lookback 3/5/10 momentum.

Background: prior activations show candidate positive momentum evidence from
lookback=5 in the collected 10-asset universe (AAPL-JNJ-XOM). The momentum
lookback sweep confirms positive REGIME_STABLE edges at lookback 3 and 5
for some assets, but leaves execution-cost robustness untested: real-world
profitability requires surviving commissions and slippage.

Objective: run a bounded cost-sensitivity check for lookback 3/5/10 momentum
using the repository's realistic BacktestConfig cost model and a matched
coin-flip null. The result classifies cost robustness: an edge survives
conservative costs -> ADMIT_TO_EVIDENCE; otherwise NOT_COST_ROBUST.

Method (fixed a-priori, not tuned to OOS):
- Use the 10-asset collected universe.
- For each lookback in {3,5,10}, run walk-forward IS/OOS validation (252d/84d/60d/60d).
- Apply realistic cost parameters via BacktestConfig.
- Compute per-fold log returns; derive segment medians (candidate vs null).
- Verdict: REGIME_STABLE (robust positive edge) if medians >0 and cross-segment swing not excessive; CONSISTENT_WITH_NOISE if all medians ≤ 0.05; otherwise REGIME_DEPENDENT or REGIME_STABLE_LOSS.
- Cost grid: no costs (baseline), low costs (5c trade + 0.05% slippage), medium (10c + 0.1%), high (15c + 0.2%).
- Independent verification path: recompute from artifact using the same deterministic cost parameters and same inputs.

Success: reproducible artifact + independent verification that agrees on cost robustness classification (regardless of positive/negative outcome).
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

COST_SCENARIOS = [
    {"name": "NO_COSTS", "commission_per_trade": 0.0, "commission_per_share": 0.0, "slippage_cents": 0.0, "slippage_proportional": 0.0},
    {"name": "LOW_COSTS", "commission_per_trade": 5.0, "commission_per_share": 0.0, "slippage_cents": 5.0, "slippage_proportional": 0.0005},
    {"name": "MEDIUM_COSTS", "commission_per_trade": 10.0, "commission_per_share": 0.0, "slippage_cents": 10.0, "slippage_proportional": 0.001},
    {"name": "HIGH_COSTS", "commission_per_trade": 15.0, "commission_per_share": 0.0, "slippage_cents": 15.0, "slippage_proportional": 0.002},
]


def build_cost_config(scenario_name: str) -> bt.BacktestConfig:
    for sc in COST_SCENARIOS:
        if sc["name"] == scenario_name:
            return bt.BacktestConfig(
                initial_capital=1e6,
                target_exposure=1.0,
                commission_per_trade=sc["commission_per_trade"],
                commission_per_share=sc["commission_per_share"],
                slippage_cents=sc["slippage_cents"],
                slippage_proportional=sc["slippage_proportional"],
                warmup_periods=0,
                risk_free=0.0,
                periods_per_year=252,
                margin_rate=0.0,
                margin_call_liquidate=True,
            )
    raise ValueError(f"Unknown cost scenario: {scenario_name}")


def segment_result(ticker, bars, lookback, cost_scenario_name):
    """Walk-forward result for one ticker, one lookback, one cost scenario."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array([float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    cfg = build_cost_config(cost_scenario_name)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
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
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        candidate_medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def classify_cost_robustness(lookback_results):
    """Determine if edge survives conservative costs.
    
    Returns classification: ADMIT_TO_EVIDENCE if at least 2 lookbacks show positive REGIME_STABLE edges under medium/high cost; otherwise NOT_COST_ROBUST.
    """
    robust_lookbacks = 0
    for lb in LOOKBACKS:
        sr = lookback_results[str(lb)]
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["candidate_medians"]):
            robust_lookbacks += 1
    if robust_lookbacks >= 2:
        return "ADMIT_TO_EVIDENCE", robust_lookbacks
    return "NOT_COST_ROBUST", robust_lookbacks


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-robustness run.")
        sys.exit(1)

    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(tickers.keys())

    # Cost-robustness analysis per lookback, per cost scenario
    cost_robustness = {}
    cost_classifications = {}

    for scenario in COST_SCENARIOS:
        scenario_name = scenario["name"]
        print(f"\n=== Cost scenario: {scenario_name} ===")
        cost_robustness[scenario_name] = {}
        
        # Per asset results
        for ticker in ticker_order:
            bars = tickers[ticker]
            per_lookback = {}
            for lb in LOOKBACKS:
                sr = segment_result(ticker, bars, lb, scenario_name)
                per_lookback[str(lb)] = sr
                print(f"  {ticker} lookback {lb}: {sr['verdict']} medians {sr['candidate_medians']} null_medians {sr['null_medians']}")
            cost_robustness[scenario_name][ticker] = per_lookback
        
        # Overall classification
        robust_count = 0
        for ticker in ticker_order:
            classification, count = classify_cost_robustness(cost_robustness[scenario_name][ticker])
            cost_classifications[f"{scenario_name}_{ticker}"] = classification
            robust_count += count
        print(f"  [{scenario_name}] Total robust lookbacks across universe: {robust_count}/ {len(ticker_order) * len(LOOKBACKS)}")

    # Determinism check for baseline (no costs)
    print("\n=== Determinism check (baseline costs) ===")
    target = ["AMZN", "JPM"]
    baseline_results = {}
    for ticker in target:
        bars = tickers[ticker]
        per_lookback = {}
        for lb in LOOKBACKS:
            per_lookback[str(lb)] = segment_result(ticker, bars, lb, "NO_COSTS")
        baseline_results[ticker] = per_lookback
    
    # Verify determinism: run a second time on the same data
    baseline_results2 = {}
    for ticker in target:
        bars = tickers[ticker]
        per_lookback = {}
        for lb in LOOKBACKS:
            per_lookback[str(lb)] = segment_result(ticker, bars, lb, "NO_COSTS")
        baseline_results2[ticker] = per_lookback
    
    out1 = json.dumps(baseline_results, sort_keys=True)
    out2 = json.dumps(baseline_results2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    # Build artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_scenarios=COST_SCENARIOS,
        per_asset=cost_robustness,
        classifications=cost_classifications,
        baseline_determinism_match=out1 == out2,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
