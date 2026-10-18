"""Momentum lookback sweep with cost sensitivity analysis.

Tests the momentum edge at lookback=3,5,10 with realistic transaction costs
vs. zero-cost baseline, under the repository's existing realistic cost model.

Background: the short-horizon return-reversal class was exhaustively explored
on the collected adjusted-close universe and rejected. Momentum is the direct
competing hypothesis to reversal: if shorting the previous N-day return loses
uniformly, the momentum variant should make the opposite mistake. This test
measures whether the short-horizon momentum edge survives conservative
transaction-cost stress.

Falsification prediction: if the momentum edge disappears under realistic
costs (or concentrates at a single lookback), this cell closes with no
positive evidence.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3,5,10; same walk-forward
windows and volatility blocks as momentum.py. Each lookback is run twice:
- zero cost: BacktestConfig(commission_per_trade=0, commission_per_share=0,
  slippage_cents=0, slippage_proportional=0)
- realistic costs: BacktestConfig(commission_per_trade=2.0,
  commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)


def segment_result(ticker, bars, lookback, costs):
    """Walk-forward regime-stability result for one ticker, lookback, and cost config."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        signals=bt.momentum_signals(closes, lookback=lookback),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=costs,
    )
    return res


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

    print("\n=== 3. Leakage review per asset (lookback=5 zero cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, ZERO_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep for zero costs (baseline)
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [cost_config][lookback][ticker][segment_index]
    results = {"zero_cost": {}, "realistic_costs": {}}

    print("\n=== 5. Running zero-cost baseline ===")
    for lb in LOOKBACKS:
        print(f"  lookback {lb}:")
        results["zero_cost"][str(lb)] = {}
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            res = segment_result(ticker, bars, lb, ZERO_COSTS)
            results["zero_cost"][str(lb)][ticker] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
                "verdict": res.overall_verdict,
                "n_folds": res.n_folds,
            }
            # Print a few examples for verification
            if ticker in ["AMZN", "JPM"]:
                segs = results["zero_cost"][str(lb)][ticker]["segments"]
                meds = results["zero_cost"][str(lb)][ticker]["medians"]
                print(f"    {ticker}: segments={len(segs)} medians={meds}")

    print("\n=== 6. Running realistic-cost sweep ===")
    for lb in LOOKBACKS:
        print(f"  lookback {lb}:")
        results["realistic_costs"][str(lb)] = {}
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            res = segment_result(ticker, bars, lb, REALISTIC_COSTS)
            results["realistic_costs"][str(lb)][ticker] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
                "verdict": res.overall_verdict,
                "n_folds": res.n_folds,
            }
            if ticker in ["AMZN", "JPM"]:
                segs = results["realistic_costs"][str(lb)][ticker]["segments"]
                meds = results["realistic_costs"][str(lb)][ticker]["medians"]
                print(f"    {ticker}: segments={len(segs)} medians={meds}")

    print("\n=== 7. Cost impact summary ===")
    print("  Comparing zero-cost vs realistic-cost medians (first segment only):")
    for lb in LOOKBACKS:
        print(f"  Lookback {lb}:")
        for ticker in ["AMZN", "JPM"]:
            zero_med = results["zero_cost"][str(lb)][ticker]["medians"][0]
            real_med = results["realistic_costs"][str(lb)][ticker]["medians"][0]
            diff = real_med - zero_med
            print(f"    {ticker}: zero_cost={zero_med:+.3f}, realistic={real_med:+.3f}, diff={diff:+.3f}")

    print("\n=== 8. Determinism: verify zero-cost results ===")
    res2 = {}
    for lb in LOOKBACKS:
        res2[str(lb)] = {}
        for ticker in target:
            bars = tickers[ticker]
            res = segment_result(ticker, bars, lb, ZERO_COSTS)
            res2[str(lb)][ticker] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
                "verdict": res.overall_verdict,
                "n_folds": res.n_folds,
            }

    # Compare first run vs second run for determinism
    deterministic = True
    for lb in LOOKBACKS:
        for ticker in target:
            orig = results["zero_cost"][str(lb)][ticker]
            second = res2[str(lb)][ticker]
            if (orig["segments"] != second["segments"] or
                orig["medians"] != second["medians"] or
                orig["null_medians"] != second["null_medians"] or
                orig["verdict"] != second["verdict"]):
                deterministic = False
                print(f"  DETERMINISM FAIL: {ticker} lookback={lb}")
                break

    if deterministic:
        print("  zero-cost results are deterministic ✓")
    else:
        print("  zero-cost results are NON-DETERMINISTIC ✗")
        sys.exit(1)

    print("\n=== 9. Writing cost-sensitivity artifact ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs_config={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        zero_costs_config={
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
        per_asset=dict(
            (t, dict(
                zero_cost=dict(
                    (str(lb), results["zero_cost"][str(lb)][t])
                    for lb in LOOKBACKS
                ),
                realistic_costs=dict(
                    (str(lb), results["realistic_costs"][str(lb)][t])
                    for lb in LOOKBACKS
                ),
            ))
            for t in ticker_order
        ),
        target_assets=dict(
            deterministic=res2,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"  artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity analysis across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())