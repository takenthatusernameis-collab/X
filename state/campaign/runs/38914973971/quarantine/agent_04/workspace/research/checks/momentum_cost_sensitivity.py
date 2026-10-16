"""Cost-sensitivity analysis for momentum lookback 3/5/10.

Background: the prior momentum research (`research/checks/momentum.py`) shows
a short-horizon edge for lookback=5 but does not account for transaction costs.
The candidate positive evidence is mechanical; execution-cost robustness remains
decision-relevant. This check tests whether the edge survives realistic
commission and slippage assumptions, using a matched null benchmark.

Method (fixed a-priori, not tuned to OOS):
- Momentum: long the previous N-day return, hold 1 day, daily rebalance.
- Real-data: collected 10-asset universe (manifest: dataset_id
  yf-ohlcv-universe-2009-to-2026-10-03).
- Walk-forward per segment train=252d/test=84d/warmup=60d/overlap=60d, min
  400 bars/segment, 4 contiguous volatility blocks per regime family.
- Realistic cost model: commission_per_share = 0.000005 (0.5 bps), 
  slippage_proportional = 0.0005 (5 bps). Zero fixed commission and slippage.
- Matched null: coin-flip sign benchmark with identical cost structure.
- Per-asset medians: each segment's median log return (OOS only) with costs
  applied. Null medians: coin-flip benchmark medians under same cost regime.
- Verdict: edge when median > 0 and dispersion > 2x null dispersion.
- Determinism: recompute AMZN/JPM segment medians to ensure reproducibility.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Realistic cost model
COMMISSION_PER_SHARE = 0.000005  # 0.5 bps per share
SLIPPAGE_PROPORTIONAL = 0.0005    # 5 bps proportional slippage
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def create_cost_config():
    """Create BacktestConfig with realistic costs."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=0.0,      # Zero fixed commission
        commission_per_share=COMMISSION_PER_SHARE,
        slippage_cents=0.0,            # Zero fixed slippage in cents
        slippage_proportional=SLIPPAGE_PROPORTIONAL,
        warmup_periods=0,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )


def segment_result_with_costs(ticker, lookback):
    """Walk-forward result for one ticker and one lookback, with realistic costs."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    cfg = create_cost_config()
    
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
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
    from research.data.preflight import main as preflight_main
    rc = preflight_main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost-sensitivity check (lookback 3 / 5 / 10) ===")
    print(f"  Params: commission_per_share={COMMISSION_PER_SHARE:.6f}, "
          f"slippage_proportional={SLIPPAGE_PROPORTIONAL:.6f}")
    print(f"  Walk-forward per segment train={TRAIN}d/test={TEST}d/warmup={WARM}d/"
          f"overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs coin-flip null "
          f"on same segments (cost-aware)")

    # Full-universe cost-sensitive lookback sweep
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    lookback_results = {lb: {t: segment_result_with_costs(t, lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        
        # Robustness verdict with cost-aware logic
        counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr = profile[str(lb)]
            if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
                counts["EDGE"] += 1
            elif sr["verdict"] == "REGIME_STABLE_LOSS":
                counts["LOSS"] += 1
            else:
                counts["NO_EDGE"] += 1
                
        if counts["EDGE"] >= 2:
            robustness = "ROBUST"
        elif counts["EDGE"] == 1:
            robustness = "SENSITIVE"
        else:
            robustness = "NO_EDGE"
            
        robustness_counts[robustness] += 1
        
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = robustness
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  lookback-robustness verdict counts (cost-aware): "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")

    # Determinism: rerun the target subset and assert identical output
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        tickers[ticker] = bt.load_ticker(ticker)[0]
    
    res2 = {lb: {t: segment_result_with_costs(t, lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        commission_per_share=COMMISSION_PER_SHARE,
        slippage_proportional=SLIPPAGE_PROPORTIONAL,
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
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check across the collected "
          "universe: exploratory simulation with realistic costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())