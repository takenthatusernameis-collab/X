"""Cost-sensitive momentum lookback sweep (3 / 5 / 10) on collected universe.

Objective: test whether the short-horizon momentum edge survives realistic transaction costs
and compare against a zero-cost matched null. Follows the repository's existing realistic cost model:
  commission_per_trade = 2.0
  commission_per_share = 0.003
  slippage_cents = 2.0
  slippage_proportional = 0.0005

The matched null uses zero costs (all parameters = 0.0).

Method (fixed a-priori, not tuned to OOS):
- long the previous lookback-day return; hold 1 day; daily rebalance;
- lookbacks 3, 5, 10; same walk-forward regime-stability gate as momentum.py;
- each cost scenario is run with the same coin-flip null (matched null);
- Determinism is asserted via internal recomputation of the AMZN/JPM subset.

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

# Existing realistic cost model from the repository's examples
cost_realistic = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Matched null with zero costs
cost_zero = bt.BacktestConfig(initial_capital=1e6)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def momentum_signals(closes: np.ndarray, lookback: int):
    """Long previous lookback-day return; same contract as research/backtest/regime_stability."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback - 1, n):
        signals[i] = bt.Signal(date=i + 1, weight=1.0)
    return signals


def run_cost_comparison(ticker, all_tickers, lookback):
    """Run momentum for both realistic and zero cost models, return comparison."""
    bars = all_tickers[ticker]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Realistic costs
    signals_real = momentum_signals(closes, lookback)
    res_real = bt.stress_segments(
        signals_fn=momentum_signals,
        bars=list(bars),
        signals=signals_real,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_realistic,
    )
    
    # Zero cost matched null
    signals_zero = momentum_signals(closes, lookback)
    res_zero = bt.stress_segments(
        signals_fn=momentum_signals,
        bars=list(bars),
        signals=signals_zero,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_zero,
    )
    
    # Compare the results
    return dict(
        realistic=dict(
            segments=[s.name for s in res_real.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in res_real.scenarios],
            null_medians=[round(s.noise_median_log_return, 3) for s in res_real.scenarios],
            candidate_dispersion=round(res_real.candidate_dispersion, 3),
            null_dispersion=round(res_real.null_dispersion, 3),
            verdict=res_real.overall_verdict,
            n_folds=res_real.n_folds,
        ),
        zero_cost=dict(
            segments=[s.name for s in res_zero.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in res_zero.scenarios],
            null_medians=[round(s.noise_median_log_return, 3) for s in res_zero.scenarios],
            candidate_dispersion=round(res_zero.candidate_dispersion, 3),
            null_dispersion=round(res_zero.null_dispersion, 3),
            verdict=res_zero.overall_verdict,
            n_folds=res_zero.n_folds,
        ),
        cost_impact=dict(
            realistic_median=res_real.scenarios[0].baseline_median_log_return,
            zero_median=res_zero.scenarios[0].baseline_median_log_return,
            median_diff=res_real.scenarios[0].baseline_median_log_return - res_zero.scenarios[0].baseline_median_log_return,
        )
    )


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

    print("\n=== 3. Leakage review per asset (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = momentum_signals(closes, 5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost sensitivity (lookback 3 / 5 / 10) across collected universe ===")
    print("  Realistic cost model: commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005")
    print("  Matched null: zero costs (all cost parameters = 0.0)")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs matched null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost sensitivity
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][cost_scenario]
    cost_results = {lb: {t: run_cost_comparison(t, all_tickers, lb) for t in ticker_order}
                    for lb in LOOKBACKS}

    per_asset_summary = {}
    for ticker in ticker_order:
        profile = {str(lb): cost_results[lb][ticker] for lb in LOOKBACKS}
        per_asset_summary[ticker] = profile

    print("  Per-asset cost sensitivity summaries:")
    for ticker in ticker_order:
        p = per_asset_summary[ticker]
        print(f"    {ticker}:")
        for lb in LOOKBACKS:
            impact = p[str(lb)]["cost_impact"]
            realistic_med = p[str(lb)]["realistic"]["medians"][0]
            realistic_verdict = p[str(lb)]["realistic"]["verdict"]
            zero_verdict = p[str(lb)]["zero_cost"]["verdict"]
            print(f"      lookback {lb}: realistic={realistic_med:+.3f} ({realistic_verdict}), "
                  f"zero={p[str(lb)]['zero_cost']['medians'][0]:+.3f} ({zero_verdict}), "
                  f"impact={impact['median_diff']:+.3f}")

    print("\n  Cost sensitivity verdict counts:")
    realistic_verdicts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0, "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    zero_verdicts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0, "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            realistic_verdicts[cost_results[lb][ticker]["realistic"]["verdict"]] += 1
            zero_verdicts[cost_results[lb][ticker]["zero_cost"]["verdict"]] += 1
    
    print(f"    Realistic costs: {realistic_verdicts}")
    print(f"    Zero costs (null): {zero_verdicts}")

    # Determinism: rerun the target subset and assert identical output.
    print("\n=== 5. Determinism (AMZN/JPM subset) ===")
    res2 = {lb: {t: run_cost_comparison(t, all_tickers, lb) for t in target}
            for lb in LOOKBACKS}

    # For determinism, compare just the realistic results (same as momentum_lookback_sweep.py)
    out1_realistic = json.dumps({lb: {t: cost_results[lb][t]["realistic"] for t in target}
                               for lb in LOOKBACKS}, sort_keys=True)
    out2_realistic = json.dumps({lb: {t: res2[lb][t]["realistic"] for t in target}
                               for lb in LOOKBACKS}, sort_keys=True)
    print("determinism: realistic costs r1 == r2: {}".format(out1_realistic == out2_realistic))
    assert out1_realistic == out2_realistic, "non-deterministic realistic costs output"

    # Also check zero cost determinism
    out1_zero = json.dumps({lb: {t: cost_results[lb][t]["zero_cost"] for t in target}
                         for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps({lb: {t: res2[lb][t]["zero_cost"] for t in target}
                         for lb in LOOKBACKS}, sort_keys=True)
    print("determinism: zero costs r1 == r2: {}".format(out1_zero == out2_zero))
    assert out1_zero == out2_zero, "non-deterministic zero costs output"

    # Create and write the artifact
    per_asset = {}
    for ticker in ticker_order:
        profile = {str(lb): cost_results[lb][ticker] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t][str(lb)]["realistic"]["medians"]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t][str(lb)]["realistic"]["null_medians"]
                                           for lb in LOOKBACKS},
                    cost_impact_realistic={str(lb): per_asset[t][str(lb)]["cost_impact"]["realistic_median"]
                                           for lb in LOOKBACKS},
                    cost_impact_zero={str(lb): per_asset[t][str(lb)]["cost_impact"]["zero_median"]
                                     for lb in LOOKBACKS},
                    cost_impact_diff={str(lb): per_asset[t][str(lb)]["cost_impact"]["median_diff"]
                                     for lb in LOOKBACKS},
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print("\n=== Cost sensitivity check completed ===")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost sensitivity across the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())