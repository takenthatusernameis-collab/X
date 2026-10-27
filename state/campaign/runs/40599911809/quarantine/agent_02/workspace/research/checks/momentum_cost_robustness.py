#!/usr/bin/env python3
"""Momentum cost-robustness check for lookback 3/5/10.

This is the bounded cost-sensitivity check required by the campaign contract:
"Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the
repository's existing realistic cost model and matched null."

The realistic cost model (copied from examples/ma_crossover.py):
  commission_per_trade = 2.0
  commission_per_share = 0.003
  slippage_cents = 2.0
  slippage_proportional = 0.0005

Method (fixed a-priori, not tuned to OOS):
- long the previous lookback-day return; hold 1 day; daily rebalance
- walk-forward IS/OOS validation (train=252d, test=84d, warmup=60d, overlap=60d)
- for each lookback in {3,5,10}, compare performance vs coin-flip null
- use the existing regime-stability verdict logic
- ensure results are deterministic and independently verifiable

Research / simulation only. No live trading or production execution.
This check closes the uncertainty: does the short-horizon momentum edge survive
conservative transaction-cost stress on the collected real-data universe?
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
REALISTIC_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"


def segment_result_cost_robustness(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, and one cost config."""
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
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Momentum Cost Robustness Check (lookback 3/5/10) ===")
    print("Dataset: {} | Universe: {} tickers".format(
        manifest["dataset_id"], len(manifest["universe"])))
    print("Walk-forward: train={}d / test={}d / warmup={}d / overlap={}d".format(
        TRAIN, TEST, WARM, OVERLAP))
    print("Realistic cost config: commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005")
    print()

    # 1. Manifest integrity
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")

    # 2. Data preflight
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # 3. Compute results with realistic costs for each lookback
    print("\n=== 3. Cost-robustness results (lookback 3/5/10) ===")
    print("  lookback 5 (reference): see existing momentum_results.json")
    print("  lookback 3 and 10: cost sensitivity comparison to zero-cost baseline")

    # Load tickers once
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # Results organized by lookback, then ticker
    results_realistic = {}
    results_zero = {}

    for lb in LOOKBACKS:
        print(f"\n  Lookback {lb}:")
        res_realistic = {}
        res_zero = {}
        for ticker in tickers:
            r1 = segment_result_cost_robustness(ticker, tickers[ticker], lb, REALISTIC_CFG)
            r2 = segment_result_cost_robustness(ticker, tickers[ticker], lb, ZERO_CFG)
            res_realistic[ticker] = r1
            res_zero[ticker] = r2
            # Show summary: real vs zero cost, and the cost impact
            cost_impact = r1["medians"][0] - r2["medians"][0]
            verdict_match = "SAME" if r1["verdict"] == r2["verdict"] else "DIFF"
            print(f"    {ticker}: real={r1['medians'][0]:+.3f} zero={r2['medians'][0]:+.3f} "
                  f"impact={cost_impact:+.3f} verdict={verdict_match}")
        results_realistic[lb] = res_realistic
        results_zero[lb] = res_zero

    # 4. Determinism test (rerun a subset)
    print("\n=== 4. Determinism check (rerun AMZN/JPM for lookback 5) ===")
    target = ["AMZN", "JPM"]
    res2_realistic = {lb: {t: segment_result_cost_robustness(t, tickers[t], lb, REALISTIC_CFG)
                          for t in target}
                     for lb in LOOKBACKS}
    res2_zero = {lb: {t: segment_result_cost_robustness(t, tickers[t], lb, ZERO_CFG)
                     for t in target}
                for lb in LOOKBACKS}

    # Compare with first run results for target assets
    deterministic_ok = True
    for lb in LOOKBACKS:
        out1_real = json.dumps({t: results_realistic[lb][t] for t in target}, sort_keys=True)
        out2_real = json.dumps(res2_realistic[lb], sort_keys=True)
        out1_zero = json.dumps({t: results_zero[lb][t] for t in target}, sort_keys=True)
        out2_zero = json.dumps(res2_zero[lb], sort_keys=True)
        if out1_real != out2_real:
            deterministic_ok = False
            print(f"    REALISTIC: Lookback {lb} non-deterministic (AMZN/JPM)")
        if out1_zero != out2_zero:
            deterministic_ok = False
            print(f"    ZERO COST: Lookback {lb} non-deterministic (AMZN/JPM)")

    print(f"  Determinism check: {'PASS' if deterministic_ok else 'FAIL'}")
    if not deterministic_ok:
        print("  ERROR: non-deterministic output detected")
        sys.exit(1)

    # 5. Write the artifact
    print("\n=== 5. Writing artifact to {} ===".format(ARTIFACT_PATH))
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_cost=dict(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
        zero_cost=dict(
            commission_per_trade=0.0,
            commission_per_share=0.0,
            slippage_cents=0.0,
            slippage_proportional=0.0,
        ),
        per_asset=dict(
            realistic={lb: results_realistic[lb] for lb in LOOKBACKS},
            zero_cost={lb: results_zero[lb] for lb in LOOKBACKS},
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("  artifact written to: {}".format(ARTIFACT_PATH))

    # 6. Summary statistics
    print("\n=== 6. Cost robustness summary ===")
    cost_sensitive_assets = []
    cost_robust_assets = []

    for ticker in tickers:
        for lb in LOOKBACKS:
            realistic_verdict = results_realistic[lb][ticker]["verdict"]
            zero_cost_verdict = results_zero[lb][ticker]["verdict"]
            realistic_median = results_realistic[lb][ticker]["medians"][0]
            zero_cost_median = results_zero[lb][ticker]["medians"][0]
            cost_impact = realistic_median - zero_cost_median
            verdict_changed = realistic_verdict != zero_cost_verdict

            if verdict_changed:
                cost_sensitive_assets.append((ticker, lb, realistic_verdict, zero_cost_verdict))
                print(f"  COST SENSITIVE: {ticker} lookback {lb}: zero={zero_cost_verdict} real={realistic_verdict} "
                      f"median impact={cost_impact:+.3f}")
            else:
                cost_robust_assets.append((ticker, lb, realistic_verdict, cost_impact))
                if cost_impact < -0.01:  # material cost drag
                    print(f"  ROBUST BUT COST-DRAGGED: {ticker} lookback {lb}: {realistic_verdict} "
                          f"median impact={cost_impact:+.3f}")
                elif abs(cost_impact) < 0.01:
                    print(f"  COST ROBUST: {ticker} lookback {lb}: {realistic_verdict} "
                          f"median impact={cost_impact:+.3f}")

    print(f"\n  Summary: {len(cost_sensitive_assets)} cost-sensitive assets, "
          f"{len(cost_robust_assets)} cost-robust assets")
    print("  Research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
