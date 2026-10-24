"""Cost-sensitivity check for lookback 3/5/10 momentum on the collected universe.

Objective: test whether the momentum edge survives realistic transaction costs.
The existing momentum evidence shows candidate positive edges at lookback=5 across many assets; this test asks whether those edges persist under realistic execution costs.

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
 daily rebalance) is walk-forward validated (train=252d/test=84d/warmup=60d/
 overlap=60d) inside each of the 4 contiguous volatility blocks, for lookback in
 {3, 5, 10}. Each scenario is run under:
   - zero cost (cfg0)
   - realistic costs (cfg1)
   - matched null under realistic costs (coin-flip signal benchmark)
The null must be drawn from the collected data itself; we use random_signals
with the same seed derived from the lookback and ticker to maintain determinism.

Method (fixed a-priori, not tuned to OOS): 
- use existing BacktestConfig objects for zero cost and realistic costs
- realistic costs are cfg1 from examples/ma_crossover.py:
    commission_per_trade=2.0
    commission_per_share=0.003
    slippage_cents=2.0
    slippage_proportional=0.0005
    initial_capital=1e6
- matched null: random_signals with deterministic seed = hash(ticker + str(lookback)) % 2**32
- same walk-forward parameters and volatility blocks as momentum_lookback_sweep.py
- determinism: internal independent recomputation of AMZN/JPM subset

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import hashlib
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

# Cost configs from examples/ma_crossover.py
cfg_zero = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
cfg_realistic = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def deterministic_null_seed(ticker: str, lookback: int) -> int:
    """Deterministic seed for null signals based on ticker and lookback."""
    # Use hash to ensure different but deterministic seeds for each asset/lookback
    combined = f"{ticker}-{lookback}"
    return int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16) % 2**31


def null_signals_for_lookback(closes, lookback, **params):
    """Wrapper for random_signals that accepts lookback parameter but ignores it."""
    # Create deterministic seed based on lookback to ensure different nulls for different lookbacks
    null_seed = deterministic_null_seed("dummy", lookback)
    return bt.random_signals(len(closes), seed=null_seed)


def segment_result_cost_sensitivity(ticker, bars, lookback):
    """Walk-forward regime-stability result with cost sensitivity analysis."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    # Zero cost run
    res_zero = bt.stress_segments(
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
        cfg=cfg_zero,
    )

    # Realistic cost run
    res_realistic = bt.stress_segments(
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
        cfg=cfg_realistic,
    )

    # Matched null under realistic costs
    # Use deterministic null signals that don't depend on lookback parameter
    # Create deterministic seed based on lookback to ensure different nulls for different lookbacks
    # Create a wrapper that accepts **params
    def null_wrapper(closes, **params):
        null_seed = deterministic_null_seed(ticker, lookback)
        return bt.random_signals(len(closes), seed=null_seed)
    
    res_null = bt.stress_segments(
        signals_fn=null_wrapper,
        bars=bars,
        signals=null_wrapper(closes),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg_realistic,
    )

    # Get scenario results
    scen_zero = res_zero.scenarios[0]
    scen_realistic = res_realistic.scenarios[0]
    scen_null = res_null.scenarios[0]

    return dict(
        lookback=lookback,
        zero_cost=dict(
            segments=[scen_zero.name],
            medians=[round(scen_zero.baseline_median_log_return, 3)],
            null_medians=[round(scen_zero.noise_median_log_return, 3)],
            candidate_dispersion=round(res_zero.candidate_dispersion, 3),
            null_dispersion=round(res_zero.null_dispersion, 3),
            verdict=res_zero.overall_verdict,
            n_folds=res_zero.n_folds,
        ),
        realistic_cost=dict(
            segments=[scen_realistic.name],
            medians=[round(scen_realistic.baseline_median_log_return, 3)],
            null_medians=[round(scen_realistic.noise_median_log_return, 3)],
            candidate_dispersion=round(res_realistic.candidate_dispersion, 3),
            null_dispersion=round(res_realistic.null_dispersion, 3),
            verdict=res_realistic.overall_verdict,
            n_folds=res_realistic.n_folds,
            cost_reduction_ratio=round(
                (scen_zero.baseline_median_log_return - scen_realistic.baseline_median_log_return) /
                scen_zero.baseline_median_log_return if scen_zero.baseline_median_log_return != 0 else 0.0,
                4,
            ) if scen_zero.baseline_median_log_return != 0 else None,
        ),
        matched_null=dict(
            segments=[scen_null.name],
            medians=[round(scen_null.baseline_median_log_return, 3)],
            null_medians=[round(scen_null.noise_median_log_return, 3)],
            candidate_dispersion=round(res_null.candidate_dispersion, 3),
            null_dispersion=round(res_null.null_dispersion, 3),
            verdict=res_null.overall_verdict,
            n_folds=res_null.n_folds,
        ),
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
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg_zero)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum cost-sensitivity check (lookback 3 / 5 / 10) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe cost-sensitivity.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [ticker][lookback]
    cost_results = {}
    for ticker in ticker_order:
        cost_results[ticker] = {}
        for lb in LOOKBACKS:
            print(f"  {ticker} lookback={lb} ...")
            cost_results[ticker][lb] = segment_result_cost_sensitivity(ticker, all_tickers[ticker], lb)

    print("  per-asset summaries:")
    for ticker in ticker_order:
        print(f"    {ticker}:")
        for lb in LOOKBACKS:
            res = cost_results[ticker][lb]
            zero_verdict = res["zero_cost"]["verdict"]
            realistic_verdict = res["realistic_cost"]["verdict"]
            null_verdict = res["matched_null"]["verdict"]
            zero_pos = any(m > 0 for m in res["zero_cost"]["medians"])
            realistic_pos = any(m > 0 for m in res["realistic_cost"]["medians"])
            null_pos = any(m > 0 for m in res["matched_null"]["medians"])
            cost_ratio = res["realistic_cost"].get("cost_reduction_ratio", None)
            print(f"      lookback={lb}: zero_cost verdict={zero_verdict[:20]:<20} "
                  f"realistic_cost verdict={realistic_verdict[:20]:<20} "
                  f"null verdict={null_verdict[:20]:<20} "
                  f"positive edge zero={zero_pos} realistic={realistic_pos} null={null_pos} "
                  f"cost_reduction={cost_ratio if cost_ratio is not None else 'N/A'}")

    # Determinism: rerun the target subset and assert identical output.
    print("\n=== 5. Determinism check ===")
    res2 = {}
    for ticker in target:
        res2[ticker] = {}
        for lb in LOOKBACKS:
            print(f"  {ticker} lookback={lb} ...")
            res2[ticker][lb] = segment_result_cost_sensitivity(ticker, tickers[ticker], lb)

    # Create artifacts
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_configs=dict(
            zero=dict(
                initial_capital=cfg_zero.initial_capital,
                commission_per_trade=cfg_zero.commission_per_trade,
                commission_per_share=cfg_zero.commission_per_share,
                slippage_cents=cfg_zero.slippage_cents,
                slippage_proportional=cfg_zero.slippage_proportional,
                warmup_periods=cfg_zero.warmup_periods,
            ),
            realistic=dict(
                initial_capital=cfg_realistic.initial_capital,
                commission_per_trade=cfg_realistic.commission_per_trade,
                commission_per_share=cfg_realistic.commission_per_share,
                slippage_cents=cfg_realistic.slippage_cents,
                slippage_proportional=cfg_realistic.slippage_proportional,
                warmup_periods=cfg_realistic.warmup_periods,
            ),
        ),
        per_asset=cost_results,
        reference_assets=target,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check across the collected "
          "universe: exploratory simulation.")

    # Check determinism for reference assets
    deterministic = True
    for ticker in target:
        for lb in LOOKBACKS:
            orig = cost_results[ticker][lb]
            rerun = res2[ticker][lb]
            if json.dumps(orig, sort_keys=True) != json.dumps(rerun, sort_keys=True):
                deterministic = False
                print(f"  DETERMINISM FAIL: {ticker} lookback={lb}")

    if deterministic:
        print("determinism: reference subset matches - PASS")
    else:
        print("determinism: reference subset matches - FAIL")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())