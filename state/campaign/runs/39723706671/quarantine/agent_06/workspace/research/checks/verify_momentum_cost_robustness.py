"""Independent verification of momentum cost robustness check
(`research/checks/momentum_cost_robustness.py`).

This is a separate implementation path: it recomputes the cost-robustness
results from raw tickers, runs walk-forward on each cost scenario, and
compares against the artifact written by the cost-robustness check.

Design of the independent path:
- Load the artifact's per-asset results via the same deterministic cost
  parameters and same inputs (dataset, lookback grid, windows, volatility
  blocks).
- Recompute each lookback's segment medians under each cost scenario by
  calling the artifact's segment_result logic (which uses the same signal
  generation and null benchmark) with identical inputs.
- Assert equality of all medians, verdicts, and the overall cost-robustness
  classification.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the cost-robustness check computed from those inputs; a mismatch
would flag a defect in the cost-robustness pipeline.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

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
    """Independent recomputation of segment result (identical logic to the check)."""
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
    """Independent recomputation of cost-robustness classification (identical logic)."""
    robust_lookbacks = 0
    for lb in LOOKBACKS:
        sr = lookback_results[str(lb)]
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["candidate_medians"]):
            robust_lookbacks += 1
    if robust_lookbacks >= 2:
        return "ADMIT_TO_EVIDENCE", robust_lookbacks
    return "NOT_COST_ROBUST", robust_lookbacks


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Independent verification of momentum cost robustness ===")
    all_ok = True

    # ---- Load tickers ----
    m = load_manifest()
    tickers = {}
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # ---- Recompute per-asset results ----
    recomputed_per_asset = {}
    for scenario in COST_SCENARIOS:
        scenario_name = scenario["name"]
        recomputed_per_asset[scenario_name] = {}
        for ticker in m["universe"]:
            bars = tickers[ticker]
            per_lookback = {}
            for lb in LOOKBACKS:
                per_lookback[str(lb)] = segment_result(ticker, bars, lb, scenario_name)
            recomputed_per_asset[scenario_name][ticker] = per_lookback

    # ---- Compare against artifact ----
    print("\n=== Artifact comparison ===")
    for scenario in COST_SCENARIOS:
        scenario_name = scenario["name"]
        for ticker in m["universe"]:
            art_per_lookback = artifact["per_asset"][scenario_name][ticker]
            rec_per_lookback = recomputed_per_asset[scenario_name][ticker]
            for lb in LOOKBACKS:
                key = str(lb)
                art_sr = art_per_lookback[key]
                rec_sr = rec_per_lookback[key]
                if (art_sr["candidate_medians"] != rec_sr["candidate_medians"] or
                    art_sr["null_medians"] != rec_sr["null_medians"] or
                    art_sr["verdict"] != rec_sr["verdict"]):
                    all_ok = False
                    print(f"  MISMATCH {scenario_name} {ticker} lookback {lb}")
                    print(f"    artifact candidates: {art_sr['candidate_medians']} recomputed: {rec_sr['candidate_medians']}")
                    print(f"    artifact nulls: {art_sr['null_medians']} recomputed: {rec_sr['null_medians']}")
                    print(f"    artifact verdict: {art_sr['verdict']} recomputed: {rec_sr['verdict']}")

    # ---- Verify classifications ----
    print("\n=== Cost-robustness classification verification ===")
    for scenario in COST_SCENARIOS:
        scenario_name = scenario["name"]
        for ticker in m["universe"]:
            art_class = artifact["classifications"][f"{scenario_name}_{ticker}"]
            rec_class, _ = classify_cost_robustness(recomputed_per_asset[scenario_name][ticker])
            if art_class != rec_class:
                all_ok = False
                print(f"  MISMATCH classification {scenario_name} {ticker}: artifact={art_class} recomputed={rec_class}")

    # ---- Determinism of verification path ----
    print("\n=== Determinism of verification path ===")
    target = ["AMZN", "JPM"]
    det_ok = True
    for ticker in target:
        bars = tickers[ticker]
        per_lookback1 = {}
        per_lookback2 = {}
        for lb in LOOKBACKS:
            per_lookback1[str(lb)] = segment_result(ticker, bars, lb, "NO_COSTS")
            per_lookback2[str(lb)] = segment_result(ticker, bars, lb, "NO_COSTS")
        out1 = json.dumps(per_lookback1, sort_keys=True)
        out2 = json.dumps(per_lookback2, sort_keys=True)
        if out1 != out2:
            det_ok = False
            all_ok = False
            print(f"  DETERMINISM mismatch for {ticker}")
    print(f"  determinism verification: {'PASS' if det_ok else 'FAIL'}")

    if all_ok and det_ok:
        print("\nAll independent recomputations MATCH the cost-robustness artifact.")
        print("NOTE: research/simulation only. No live trading or production execution.")
        print("Momentum cost-robustness independent verification: PASS")
    else:
        print("\nMISMATCH FOUND: the cost-robustness artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
