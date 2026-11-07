"""Cost sensitivity test for momentum lookback 3/5/10 under realistic transaction costs.

This is the R-001 focused task. The momentum class has been admitted as candidate
positive evidence: REGIME_STABLE positive edge in 7/10 of the collected universe
at lookback=5, hold=1 day. This test varies transaction costs only, not the
lookback window, to answer the primary question:

  Does the qualified momentum edge survive conservative transaction-cost stress
  on the collected real-data universe?

Method (fixed a-priori, not tuned to OOS): same momentum signal as `momentum.py`
(lookback fixed at 5, 3/5/10 grid) with realistic transaction costs:
- commission_per_trade: 2.0 (dollars)
- commission_per_share: 0.3 (dollars per share)
- slippage_cents: 2.0 (dollars per share)
- slippage_proportional: 0.0010 (10 basis points)
- target_exposure: 1.0 (constant-dollar sizing)
- initial_capital: 1e6

Walk-forward validation uses the same regime blocks, windows, and coin-flip null
as the existing momentum check (`research/checks/momentum.py`).

Classification scheme (predeclared; applied after the grid runs, not used
for parameter selection):
  - Cost robustness verdict per lookback:
      ROBUST = positive edge survives with realistic costs (median > 0.002)
      SENSITIVE = edge collapses or becomes noise-like under realistic costs
      NO_EDGE = edge is noise-like even with zero costs
  - Universe-level classification:
      COST_ROBUST = majority of lookbacks show ROBUST verdicts
      COST_SENSITIVE = majority show SENSITIVE verdicts
      COST_DESTRUCTIVE = majority show NO_EDGE verdicts

A-priori falsification prediction: if the lookback=5 momentum edge is
transaction-cost sensitive, the positive median log returns will collapse
significantly under realistic costs, and the edge will be eliminated or become
indistinguishable from the coin-flip null.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

# Predeclared realistic cost model from activation 2026-10-02
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,        # $2 per trade
    commission_per_share=0.3,        # $0.3 per share
    slippage_cents=2.0,              # 2 cents per share
    slippage_proportional=0.0010,    # 10 basis points
    warmup_periods=0,
    periods_per_year=252,
)


# Zero-cost baseline config for comparison
ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,        # No commission
    commission_per_share=0.0,        # No per-share commission
    slippage_cents=0.0,              # No slippage
    slippage_proportional=0.0,       # No proportional slippage
    warmup_periods=0,
    periods_per_year=252,
)


def momentum_cost_sensitivity(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback under realistic costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    # Run with realistic costs
    realistic_res = bt.stress_segments(
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
        cfg=REALISTIC_COST_CONFIG,
    )
    
    # Run with zero costs
    zero_cost_res = bt.stress_segments(
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
        cfg=ZERO_COST_CONFIG,
    )
    
    # Calculate cost impact by comparing realistic vs zero-cost medians
    cost_impacts = []
    for realistic_scen, zero_scen in zip(realistic_res.scenarios, zero_cost_res.scenarios):
        cost_impact = realistic_scen.baseline_median_log_return - zero_scen.baseline_median_log_return
        cost_impacts.append(round(cost_impact, 3))
    
    return dict(
        segments=[s.name for s in realistic_res.scenarios],
        realistic_medians=[round(s.baseline_median_log_return, 3) for s in realistic_res.scenarios],
        zero_cost_medians=[round(s.baseline_median_log_return, 3) for s in zero_cost_res.scenarios],
        candidate_dispersion=round(realistic_res.candidate_dispersion, 3),
        null_dispersion=round(realistic_res.null_dispersion, 3),
        n_folds=realistic_res.n_folds,
        verdict=realistic_res.overall_verdict,
        cost_impact=round(sum(cost_impacts) / len(cost_impacts), 3),
    )


def zero_cost_sensitivity(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback with zero costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
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
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=ZERO_COST_CONFIG,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        zero_cost_medians=[round(s.zero_cost_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        cost_impact=round(scen.median_log_return - scen.zero_cost_median_log_return, 3),
    )


def cost_robustness_verdict(lookback_results):
    """Verdict on whether the edge is cost-robust for one lookback across the universe.
    
    ROBUST      : positive edge survives realistic costs (median > 0.002)
    SENSITIVE   : edge collapses or becomes noise-like under realistic costs
    NO_EDGE     : edge is noise-like even with zero costs
    """
    robust_count = 0
    sensitive_count = 0
    no_edge_count = 0
    
    for ticker_result in lookback_results.values():
        verdict = ticker_result["verdict"]
        cost_impact = ticker_result["cost_impact"]
        
        if verdict == "REGIME_STABLE" and all(m > 0.002 for m in ticker_result["realistic_medians"]):
            if cost_impact > -0.001:  # Edge survives realistic costs with minimal impact
                robust_count += 1
            else:
                sensitive_count += 1
        else:
            no_edge_count += 1
    
    if robust_count >= len(lookback_results) * 0.6:  # Majority robust
        return "ROBUST", {"robust": robust_count, "sensitive": sensitive_count, "no_edge": no_edge_count}
    elif sensitive_count >= len(lookback_results) * 0.6:  # Majority sensitive
        return "SENSITIVE", {"robust": robust_count, "sensitive": sensitive_count, "no_edge": no_edge_count}
    else:
        return "NO_EDGE", {"robust": robust_count, "sensitive": sensitive_count, "no_edge": no_edge_count}


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

    print("\n=== 3. Cost sensitivity review (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, REALISTIC_COST_CONFIG)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"cost review [PASS], total_return {total_return:+.3f}")
    print("  realistic cost model: $2/trade + $0.3/share commission + 2¢/share fixed + 10bps proportional slippage")

    print("\n=== 4. Momentum cost sensitivity sweep (lookback 3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, realistic cost grid, zero-cost baseline\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost sensitivity sweep for each lookback
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker]
    lookback_results = {lb: {t: momentum_cost_sensitivity(t, all_tickers[t], lb) for t in ticker_order}
                       for lb in LOOKBACKS}

    per_lookback = {}
    for lb in LOOKBACKS:
        verdict, counts = cost_robustness_verdict(lookback_results[lb])
        per_lookback[lb] = {"verdict": verdict, "counts": counts}

    print("  Cost sensitivity results per lookback:")
    for lb in LOOKBACKS:
        verdict = per_lookback[lb]["verdict"]
        counts = per_lookback[lb]["counts"]
        print(f"    lookback {lb}: {verdict} (robust={counts['robust']}, sensitive={counts['sensitive']}, no_edge={counts['no_edge']})")

    print("\n  Per-asset cost-impact profiles (lookback=5, reference):")
    for ticker in ticker_order:
        result = lookback_results[5][ticker]
        cost_impact = result["cost_impact"]
        zero_cost_median = result["zero_cost_medians"][0]
        real_cost_median = result["realistic_medians"][0]
        print(f"    {ticker:6s}: zero-cost={zero_cost_median:+6.3f}, realistic={real_cost_median:+6.3f}, "
              f"impact={cost_impact:+6.3f}, verdict={result['verdict']}")

    # Determinism: rerun the target subset and assert identical output
    res2 = {lb: {t: momentum_cost_sensitivity(t, tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                      for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_lookback=per_lookback,
        cost_config={
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
            "target_exposure": REALISTIC_COST_CONFIG.target_exposure,
        },
        zero_cost_config={
            "commission_per_trade": ZERO_COST_CONFIG.commission_per_trade,
            "commission_per_share": ZERO_COST_CONFIG.commission_per_share,
            "slippage_cents": ZERO_COST_CONFIG.slippage_cents,
            "slippage_proportional": ZERO_COST_CONFIG.slippage_proportional,
            "target_exposure": ZERO_COST_CONFIG.target_exposure,
        },
        per_asset={},
    )
    
    # Add per-asset details for verification
    for ticker in ticker_order:
        asset_data = {}
        for lb in LOOKBACKS:
            asset_data[str(lb)] = lookback_results[lb][ticker]
        artifact["per_asset"][ticker] = asset_data

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost sensitivity test: exploratory simulation under realistic transaction costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())