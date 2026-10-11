"""
Simplified cost-sensitivity check for lookback 3/5/10 momentum under realistic costs.

This is a minimal implementation that follows the exact pattern of momentum_lookback_sweep.py
but applies realistic transaction costs to both the momentum signal and matched null.

The realistic costs are: commission_per_share=0.003, slippage_cents=5.0,
slippage_proportional=0.001 (from tests/test_engine.py).
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

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.003,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

NO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)


def segment_result(ticker, bars, lookback, costs):
    """Walk-forward regime-stability result for one ticker and one lookback."""
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
        cfg=costs,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
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

    print("\n=== 3. Cost-sensitivity check for lookback 3 / 5 / 10 under realistic costs ===")
    print(f"  Realistic costs: commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Run with realistic costs
    print("Running momentum lookback sweep (3, 5, 10) with realistic costs...")
    lookback_results = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COSTS)
                             for t in ticker_order} for lb in LOOKBACKS}

    # Run with no costs for comparison
    print("Running momentum lookback sweep (3, 5, 10) with no costs...")
    lookback_results_no_costs = {lb: {t: segment_result(t, all_tickers[t], lb, NO_COSTS)
                                       for t in ticker_order} for lb in LOOKBACKS}

    # Build simple artifact
    per_asset = {}
    for ticker in ticker_order:
        profile = {
            "lookback_medians_real": {str(lb): lookback_results[lb][ticker]["medians"]
                                      for lb in LOOKBACKS},
            "lookback_nulls_real": {str(lb): lookback_results[lb][ticker]["null_medians"]
                                    for lb in LOOKBACKS},
            "lookback_verdicts_real": {str(lb): lookback_results[lb][ticker]["verdict"]
                                       for lb in LOOKBACKS},
            "lookback_medians_no_costs": {str(lb): lookback_results_no_costs[lb][ticker]["medians"]
                                          for lb in LOOKBACKS},
            "lookback_nulls_no_costs": {str(lb): lookback_results_no_costs[lb][ticker]["null_medians"]
                                        for lb in LOOKBACKS},
            "lookback_verdicts_no_costs": {str(lb): lookback_results_no_costs[lb][ticker]["verdict"]
                                           for lb in LOOKBACKS},
        }
        per_asset[ticker] = profile

    print(f"\nCost-sensitivity analysis complete for {len(ticker_order)} assets.")
    print("This shows how realistic transaction costs affect the momentum edge.")

    artifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=list(LOOKBACKS),
        realistic_costs=dict(
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        per_asset=per_asset,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
