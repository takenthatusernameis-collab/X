"""Cost robustness check for short-horizon momentum (lookback 3 / 5 / 10) on the collected universe.

Objective: the momentum cell was admitted as candidate positive evidence with a REGIME_STABLE edge in 7/10 assets. The remaining information gap is whether this edge survives conservative transaction-cost stress on real data.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; walk-forward per segment train=252d/test=84d/warmup=60d/overlap=60d; for each lookback, run both zero-cost and realistic-cost models; compare against a matched coin-flip null benchmark run on the same segments. The realistic cost model is based on the enterprise's standard configuration:
    - commission_per_trade: 2.0 (cents)
    - commission_per_share: 0.003
    - slippage_cents: 2.0
    - slippage_proportional: 0.0005

Either outcome closes the momentum cell: a cost-robust edge is one where the uniformly-positive REGIME_STABLE pattern at lookback=5 (the admitted candidate) also appears at >=2 lookbacks under realistic costs; a cost-sensitive result shows the edge vanishes under transaction costs.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

# Realistic cost model from enterprise standard
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=0)


def cost_sensitivity_check(ticker, bars, lookback):
    """Run momentum signal with both zero-cost and realistic costs, compare to matched null."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Zero-cost momentum
    sig_zero = bt.momentum_signals(closes, lookback=lookback)
    if len(sig_zero) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig_zero)} != bars {len(closes)}")
    
    # Realistic-cost momentum
    sig_real = bt.momentum_signals(closes, lookback=lookback)
    
    # Run both versions using stress_segments with different cfg
    res_zero = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig_zero,
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
    
    res_real = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig_real,
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
    
    return {
        "lookback": lookback,
        "zero_cost": {
            "segments": [s.name for s in res_zero.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res_zero.scenarios],
            "null_medians": [round(s.noise_median_log_return, 3) for s in res_zero.scenarios],
            "candidate_dispersion": round(res_zero.candidate_dispersion, 3),
            "null_dispersion": round(res_zero.null_dispersion, 3),
            "verdict": res_zero.overall_verdict,
        },
        "realistic_cost": {
            "segments": [s.name for s in res_real.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res_real.scenarios],
            "null_medians": [round(s.noise_median_log_return, 3) for s in res_real.scenarios],
            "candidate_dispersion": round(res_real.candidate_dispersion, 3),
            "null_dispersion": round(res_real.null_dispersion, 3),
            "verdict": res_real.overall_verdict,
        },
    }


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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost-robustness check (lookback 3 / 5 / 10) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a matched "
          "coin-flip null on the same segments, both zero-cost and "
          "realistic-cost models\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost robustness check for all lookbacks
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    cost_results = {}
    for ticker in ticker_order:
        ticker_results = {}
        for lookback in LOOKBACKS:
            result = cost_sensitivity_check(ticker, all_tickers[ticker], lookback)
            ticker_results[str(lookback)] = result
        cost_results[ticker] = ticker_results

    # Analyze cost robustness
    zero_cost_robust = {lb: 0 for lb in LOOKBACKS}
    realistic_cost_robust = {lb: 0 for lb in LOOKBACKS}
    zero_cost_loss = {lb: 0 for lb in LOOKBACKS}
    realistic_cost_loss = {lb: 0 for lb in LOOKBACKS}

    for ticker in ticker_order:
        for lookback in LOOKBACKS:
            result = cost_results[ticker][str(lookback)]
            zero_result = result["zero_cost"]
            real_result = result["realistic_cost"]
            
            # Count robust (REGIME_STABLE with positive medians)
            zero_robust = zero_result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in zero_result["medians"])
            real_robust = real_result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in real_result["medians"])
            
            # Count losses (REGIME_STABLE_LOSS)
            zero_loss = zero_result["verdict"] == "REGIME_STABLE_LOSS"
            real_loss = real_result["verdict"] == "REGIME_STABLE_LOSS"
            
            if zero_robust:
                zero_cost_robust[lookback] += 1
            if real_robust:
                realistic_cost_robust[lookback] += 1
            if zero_loss:
                zero_cost_loss[lookback] += 1
            if real_loss:
                realistic_cost_loss[lookback] += 1

    print("\n=== 4. Cost-robustness summary ===")
    print("  Zero-cost model:")
    for lb in LOOKBACKS:
        robust = zero_cost_robust[lb]
        loss = zero_cost_loss[lb]
        print(f"    lookback {lb}: {robust}/10 REGIME_STABLE edges (positive), {loss}/10 REGIME_STABLE_LOSS")
    
    print("  Realistic-cost model:")
    for lb in LOOKBACKS:
        robust = realistic_cost_robust[lb]
        loss = realistic_cost_loss[lb]
        print(f"    lookback {lb}: {robust}/10 REGIME_STABLE edges (positive), {loss}/10 REGIME_STABLE_LOSS")

    # Determine overall verdict
    positive_lookbacks_zero = [lb for lb in LOOKBACKS if zero_cost_robust[lb] >= 2]
    positive_lookbacks_real = [lb for lb in LOOKBACKS if realistic_cost_robust[lb] >= 2]
    
    print(f"\n  Positive lookback counts (zero-cost): {len(positive_lookbacks_zero)}/{len(LOOKBACKS)}")
    print(f"  Positive lookback counts (realistic-cost): {len(positive_lookbacks_real)}/{len(LOOKBACKS)}")
    
    if len(positive_lookbacks_real) >= 2:
        verdict = "REGIME_STABLE"  # Cost-robust edge survives
        print(f"\n  verdict: {verdict} (cost-robust edge survives)")
    elif len(positive_lookbacks_zero) >= 2 and len(positive_lookbacks_real) == 0:
        verdict = "REGIME_STABLE_LOSS"  # Edge vanishes under costs
        print(f"\n  verdict: {verdict} (cost-sensitive, edge vanishes)")
    else:
        verdict = "CONSISTENT_WITH_NOISE"  # No robust edge
        print(f"\n  verdict: {verdict} (no robust edge under realistic costs)")

    # Determinism check: rerun target subset (AMZN, JPM)
    print("\n=== 5. Determinism check (AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    rerun_results = {}
    for ticker in target:
        rerun_results[ticker] = {}
        for lookback in LOOKBACKS:
            result = cost_sensitivity_check(ticker, all_tickers[ticker], lookback)
            rerun_results[ticker][str(lookback)] = result

    # Compare first run vs rerun
    all_match = True
    for ticker in target:
        for lookback in LOOKBACKS:
            first = cost_results[ticker][str(lookback)]
            rerun = rerun_results[ticker][str(lookback)]
            
            # Check zero-cost match
            zero_match = (
                first["zero_cost"]["verdict"] == rerun["zero_cost"]["verdict"] and
                first["zero_cost"]["medians"] == rerun["zero_cost"]["medians"] and
                first["zero_cost"]["candidate_dispersion"] == rerun["zero_cost"]["candidate_dispersion"] and
                first["zero_cost"]["null_dispersion"] == rerun["zero_cost"]["null_dispersion"]
            )
            
            # Check realistic-cost match
            real_match = (
                first["realistic_cost"]["verdict"] == rerun["realistic_cost"]["verdict"] and
                first["realistic_cost"]["medians"] == rerun["realistic_cost"]["medians"] and
                first["realistic_cost"]["candidate_dispersion"] == rerun["realistic_cost"]["candidate_dispersion"] and
                first["realistic_cost"]["null_dispersion"] == rerun["realistic_cost"]["null_dispersion"]
            )
            
            if not (zero_match and real_match):
                all_match = False
                print(f"  MISMATCH: {ticker} lookback {lookback} (zero-cost match: {zero_match}, realistic-cost match: {real_match})")
    
    if all_match:
        print("  deterministic: r1 == r2: True")
    else:
        print("  deterministic: r1 == r2: False")
        sys.exit(1)

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=cost_results,
        universe=dict(
            verdict_counts={
                "REGIME_STABLE": len(positive_lookbacks_real) >= 2,
                "REGIME_STABLE_LOSS": len(positive_lookbacks_zero) >= 2 and len(positive_lookbacks_real) == 0,
                "CONSISTENT_WITH_NOISE": len(positive_lookbacks_real) < 2 and not (len(positive_lookbacks_zero) >= 2 and len(positive_lookbacks_real) == 0),
            },
            ticker_order=ticker_order,
            positive_lookbacks_zero=positive_lookbacks_zero,
            positive_lookbacks_real=positive_lookbacks_real,
            verdict=verdict,
        ),
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness check on the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
