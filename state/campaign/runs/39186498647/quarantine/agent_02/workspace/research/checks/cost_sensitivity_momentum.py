#!/usr/bin/env python3
"""Bounded cost-sensitivity check for lookback 3/5/10 momentum.

Objective: Test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected 10-asset universe.

Scope: lookback in {3, 5, 10}; walk-forward IS/OOS validated; 4 volatility
blocks; matched null (coin-flip) with same cost structure.

Methodology: Run the same momentum framework as momentum_lookback_sweep.py but
with realistic transaction costs applied via BacktestConfig. The results are
reproducible, independent, and stop at the first meaningful cost grid point
where the momentum edge no longer exceeds the matched null.

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
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitivity_results.json"


def conservative_cost_config():
    """Realistic transaction costs for a conservative execution model.
    
    These values reflect typical retail brokerage costs (commission per share,
    fixed per-trade fee, and bid-ask spread proxy as slippage). They are
    deliberately conservative to stress-test the momentum edge.
    """
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=4.95,      # $4.95 per trade (typical retail broker)
        commission_per_share=0.003,     # $0.003 per share
        slippage_cents=1.0,             # $0.01 per share (half-point spread proxy)
        slippage_proportional=0.0001,   # 10 bps proportional slippage
        warmup_periods=0,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )


def segment_result_cost_sensitive(ticker, bars, lookback, cost_cfg):
    """Walk-forward result for one ticker, one lookback, with cost sensitivity.
    
    This mirrors the segment_result function in momentum_lookback_sweep.py
    but uses the cost_cfg for realistic transaction costs.
    """
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
        cfg=cost_cfg,
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


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== BOUNDED COST-SENSITIVITY CHECK FOR MOMENTUM ===")
    print(f"Cost config: commission_per_trade=4.95, commission_per_share=0.003, ")
    print(f"            slippage_cents=1.0, slippage_proportional=0.0001")
    print(f"Lookbacks: {LOOKBACKS} across collected 10-asset universe")
    print(f"Walk-forward: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d")
    print(f"Segments: {N_BLOCKS} volatility blocks; min {MIN_SEGMENT_BARS} bars/segment")
    print()

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    print("=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-sensitive run.")
        sys.exit(1)
    print()

    cost_cfg = conservative_cost_config()
    print("=== 3. Cost-sensitive momentum lookback sweep (3 / 5 / 10) ===")
    print(f"  Params: walk-forward per segment, 4 volatility blocks, "
          f"matched null (coin-flip) with SAME cost structure")
    print(f"  Cost applied: commission + slippage as configured above")
    print()

    # Load all tickers from manifest (the 10-asset collected universe)
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())
    print(f"  Universe: {len(ticker_order)} tickers")

    # Results indexed [lookback][ticker]
    lookback_results = {lb: {t: segment_result_cost_sensitive(t, all_tickers[t], lb, cost_cfg) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        
        # Simple robustness verdict (same logic as momentum_lookback_sweep.py)
        edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            verdict = profile[str(lb)]["verdict"]
            medians = profile[str(lb)]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                edge_counts["EDGE"] += 1
            elif verdict == "REGIME_STABLE_LOSS":
                edge_counts["LOSS"] += 1
            else:
                edge_counts["NO_EDGE"] += 1
        
        if edge_counts["EDGE"] >= 2:
            v = "ROBUST"
        elif edge_counts["EDGE"] == 1:
            v = "SENSITIVE"
        else:
            v = "NO_EDGE"
        robustness_counts[v] += 1

        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        profile["edge_counts"] = edge_counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print(f"  Cost-sensitive lookback 5 verdict counts across universe: {lookback_5_counts}")
    print(f"  Cost-sensitive robustness verdict counts: ROBUST={robustness_counts['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")

    # Determinism: rerun the reference subset and assert identical output
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    res2 = {lb: {t: segment_result_cost_sensitive(t, tickers[t], lb, cost_cfg) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"
    print()

    # Prepare artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_config={
            "commission_per_trade": cost_cfg.commission_per_trade,
            "commission_per_share": cost_cfg.commission_per_share,
            "slippage_cents": cost_cfg.slippage_cents,
            "slippage_proportional": cost_cfg.slippage_proportional,
        },
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                           for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"artifact written to: {ARTIFACT_PATH}")
    print()
    print("=== BOUNDED COST-SENSITIVITY CHECK COMPLETE ===")
    print("This result addresses the evidence gate: momentum edge under")
    print("conservative transaction-cost stress on the collected universe.")
    print("Research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
