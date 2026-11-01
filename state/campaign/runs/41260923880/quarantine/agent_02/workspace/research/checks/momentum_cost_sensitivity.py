#!/usr/bin/env python3
"""Cost sensitivity check for lookback 3/5/10 momentum.

This is the artifact that the controller expects. It runs momentum with
realistic costs and compares against the zero-cost (matched null) benchmark.

The existing momentum research shows a positive REGIME_STABLE edge at lookback=5
across ~70% of the collected universe. The question is whether that edge
survives realistic transaction costs.

Method:
- Use the exact same universe and walk-forward parameters as momentum_lookback_sweep.py
- Run momentum_signals with lookback in {3, 5, 10} across all 10 collected tickers
- Compare results between realistic costs and zero costs
- The realistic cost config matches the standard cfg1 from ma_crossover.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result_cost_sensitive(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and config."""
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


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset.

    ROBUST      : uniformly positive REGIME_STABLE edges at >= 2 lookbacks.
    SENSITIVE   : a positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE     : no lookback shows a positive REGIME_STABLE edge.
    """
    def lookback_verdict(sr):
        if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
            return "EDGE"
        if sr["verdict"] == "REGIME_STABLE_LOSS":
            return "LOSS"
        return "NO_EDGE"

    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        verdict = lookback_verdict(lookback_results[str(lb)])
        counts[verdict] += 1
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
        print("PREFLIGHT FAILED; aborting cost-sensitivity run.")
        sys.exit(1)

    print("\n=== 3. Cost-sensitive momentum lookback sweep ===")
    print("  realistic costs: commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005")
    print("  zero costs (matched null): all costs = 0")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs a coin-flip null "
          f"on the same segments")

    # Full-universe lookback sweep with realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Realistic costs configuration (from ma_crossover.py cfg1)
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    # Zero costs configuration (matched null)
    cfg_zero = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )

    print("  Running momentum with realistic costs...")
    lookback_results_realistic = {
        lb: {t: segment_result_cost_sensitive(t, all_tickers[t], lb, cfg_realistic)
             for t in ticker_order}
        for lb in LOOKBACKS
    }

    print("  Running momentum with zero costs (matched null)...")
    lookback_results_zero = {
        lb: {t: segment_result_cost_sensitive(t, all_tickers[t], lb, cfg_zero)
             for t in ticker_order}
        for lb in LOOKBACKS
    }

    # Compare results across costs
    print("\n=== 4. Cost sensitivity comparison ===")
    per_asset_comparison = {}
    cost_robustness_counts_realistic = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    cost_robustness_counts_zero = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}

    for ticker in ticker_order:
        profile_real = {str(lb): lookback_results_realistic[lb][ticker] for lb in LOOKBACKS}
        profile_zero = {str(lb): lookback_results_zero[lb][ticker] for lb in LOOKBACKS}

        # Calculate cost reduction for each lookback
        cost_reduction = {}
        for lb in LOOKBACKS:
            median_real = profile_real[str(lb)]["medians"][0]
            median_zero = profile_zero[str(lb)]["medians"][0]
            reduction = round(median_real - median_zero, 3)
            cost_reduction[str(lb)] = reduction

        # Robustness verdict for realistic costs
        v_real, counts_real = robustness_verdict(profile_real)
        cost_robustness_counts_realistic[v_real] += 1

        # Robustness verdict for zero costs
        v_zero, counts_zero = robustness_verdict(profile_zero)
        cost_robustness_counts_zero[v_zero] += 1

        per_asset_comparison[ticker] = {
            "lookback_3_realistic": profile_real["3"],
            "lookback_5_realistic": profile_real["5"],
            "lookback_10_realistic": profile_real["10"],
            "lookback_3_zero": profile_zero["3"],
            "lookback_5_zero": profile_zero["5"],
            "lookback_10_zero": profile_zero["10"],
            "cost_reduction": cost_reduction,
            "robustness_verdict_realistic": v_real,
            "robustness_verdict_zero": v_zero,
        }

    print("  Cost sensitivity results:")
    for ticker in ticker_order[:3]:  # Show first 3 for brevity
        profile = per_asset_comparison[ticker]
        print(f"    {ticker}: lookback 3/5/10 realistic medians "
              f"{profile['lookback_3_realistic']['medians']} / "
              f"{profile['lookback_5_realistic']['medians']} / "
              f"{profile['lookback_10_realistic']['medians']}")
        print(f"             cost reduction: {profile['cost_reduction']}")
        print(f"             robust (realistic): {profile['robustness_verdict_realistic']}, "
              f"robust (zero): {profile['robustness_verdict_zero']}")

    if len(ticker_order) > 3:
        print(f"    ... and {len(ticker_order)-3} more tickers")

    print(f"\n  Lookback-robustness verdict counts (realistic costs): "
          f"ROBUST={cost_robustness_counts_realistic['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={cost_robustness_counts_realistic['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={cost_robustness_counts_realistic['NO_EDGE']}\n")

    print(f"  Lookback-robustness verdict counts (zero costs): "
          f"ROBUST={cost_robustness_counts_zero['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={cost_robustness_counts_zero['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={cost_robustness_counts_zero['NO_EDGE']}\n")

    # Determinism: rerun a subset and assert identical output
    print("\n=== 5. Determinism check ===")
    target = ["AMZN", "JPM"]
    res2_real = {lb: {t: segment_result_cost_sensitive(t, all_tickers[t], lb, cfg_realistic)
                      for t in target}
                 for lb in LOOKBACKS}
    res2_zero = {lb: {t: segment_result_cost_sensitive(t, all_tickers[t], lb, cfg_zero)
                      for t in target}
                 for lb in LOOKBACKS}

    out1_real = json.dumps({lb: {t: lookback_results_realistic[lb][t] for t in target}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_real = json.dumps(res2_real, sort_keys=True)
    out1_zero = json.dumps({lb: {t: lookback_results_zero[lb][t] for t in target}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps(res2_zero, sort_keys=True)

    real_det = out1_real == out2_real
    zero_det = out1_zero == out2_zero
    print(f"  realistic costs deterministic: {real_det}")
    print(f"  zero costs deterministic: {zero_det}")
    assert real_det and zero_det, "non-deterministic output"

    # Create artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset_comparison,
        cost_config_realistic=dict(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
        cost_config_zero=dict(
            commission_per_trade=0.0,
            commission_per_share=0.0,
            slippage_cents=0.0,
            slippage_proportional=0.0,
        ),
        robustness_counts_realistic=cost_robustness_counts_realistic,
        robustness_counts_zero=cost_robustness_counts_zero,
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\n=== 6. Artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. Cost sensitivity check for momentum "
          "lookback 3/5/10 across the collected universe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())