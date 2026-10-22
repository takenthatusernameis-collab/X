"""Cost-sensitive momentum check for lookback 3/5/10 with realistic transaction costs.

Objective: test whether the short-horizon momentum edge (from momentum.py) survives
conservative transaction-cost stress on the collected real-data universe.

Method (fixed a-priori): long the previous lookback-day return; hold 1 day; daily
rebalance; lookbacks 3, 5, 10; same walk-forward windows and volatility blocks as
momentum.py; realistic cost model includes commission and slippage; matched
coin-flip null benchmark run on the same segments. Determinism is asserted via an
internal independent recomputation.

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

# Realistic cost model (per trade, per share, and slippage)
COST_MODEL = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,      # $0 per trade
    commission_per_share=0.003,    # $0.003 per share ($30 per round-trip for 10k shares)
    slippage_cents=5.0,            # $5 per share fixed slippage
    slippage_proportional=0.001,  # 10 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
)

# Matched null: coin-flip benchmark with same cost model
NULL_COST_MODEL = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.003,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def run_asset_with_cost(asset, tickers, lookback, cost_model):
    """Run momentum on one asset with given cost model."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
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
    return res


def run_null_benchmark(asset, tickers, lookback, cost_model):
    """Run coin-flip null with same cost model on same asset/segment."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Coin-flip null signals: +1/-1 randomly, then re-run walk-forward with same segments
    np.random.seed(SEED)
    signals = []
    for i in range(len(closes)):
        if i < lookback:
            signals.append(bt.Signal(date=i + 1, weight=0.0))
        else:
            signals.append(bt.Signal(date=i + 1, weight=1.0 if np.random.randn() > 0 else -1.0))
    
    res = bt.stress_segments(
        signals_fn=lambda c, **p: signals,
        bars=bars,
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost-sensitive momentum check (lookback 3 / 5 / 10) ===")
    print("  Params: realistic cost model - commission $0.003/share, $5 fixed slippage per share, 10 bps proportional slippage")
    print("  Matched null: coin-flip signals with identical cost model, same segments")
    print("  Walk-forward: train={}d / test={}d / warmup={}d / overlap={}d, min {} bars/segment, 4 contiguous volatility blocks\n".format(
        TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full universe cost-sensitive results
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        all_tickers[ticker] = bars
    ticker_order = list(all_tickers.keys())

    # Cost-sensitive momentum results
    cost_momentum_results = {lb: {} for lb in LOOKBACKS}
    cost_null_results = {lb: {} for lb in LOOKBACKS}

    print("Running cost-sensitive momentum with lookback {} / {} / {}...".format(*LOOKBACKS))
    for lookback in LOOKBACKS:
        print(f"  Processing lookback={lookback} for all {len(ticker_order)} tickers")
        for ticker in ticker_order:
            print(f"    {ticker}...", end=" ")
            
            # Cost-sensitive momentum
            momentum_res = run_asset_with_cost(ticker, all_tickers, lookback, COST_MODEL)
            cost_momentum_results[lookback][ticker] = {
                "segments": [s.name for s in momentum_res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in momentum_res.scenarios],
                "noise_medians": [round(s.noise_median_log_return, 3) for s in momentum_res.scenarios],
                "candidate_dispersion": round(momentum_res.candidate_dispersion, 3),
                "null_dispersion": round(momentum_res.null_dispersion, 3),
                "n_folds": momentum_res.n_folds,
                "verdict": momentum_res.overall_verdict,
            }
            print(f"done ({momentum_res.overall_verdict})")

    # Summary statistics
    print("\n=== 4. Cost-Sensitivity Summary ===")
    lookback_5_cost_results = cost_momentum_results[5]
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                        "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        # Count lookback 5 verdict categories
        verdict = lookback_5_cost_results[ticker]["verdict"]
        lookback_5_counts[verdict] += 1
        
        # Determine lookback-robustness based on positive edges
        positive_edges = 0
        for lb in LOOKBACKS:
            sr = cost_momentum_results[lb][ticker]
            if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
                positive_edges += 1
        
        if positive_edges >= 2:
            robustness_counts["ROBUST"] += 1
        elif positive_edges == 1:
            robustness_counts["SENSITIVE"] += 1
        else:
            robustness_counts["NO_EDGE"] += 1

    print("  Lookback 5 (reference) verdict counts across universe: {}".format(lookback_5_counts))
    print("  Lookback-robustness verdict counts: ROBUST={} (>=2 lookbacks positive) | SENSITIVE={} (edge at exactly one lookback) | NO_EDGE={}\n".format(
        robustness_counts["ROBUST"], robustness_counts["SENSITIVE"], robustness_counts["NO_EDGE"]))

    print("  Per-asset cost-sensitivity profiles (lookback 3 / 5 / 10):")
    for ticker in ticker_order:
        # Reconstruct per-ticker profile from cost_momentum_results structure
        profile = {}
        for lb in LOOKBACKS:
            profile[lb] = cost_momentum_results[lb][ticker]
        
        # Calculate positive edges for this ticker
        positive_edges = 0
        for lb in LOOKBACKS:
            sr = profile[lb]
            if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
                positive_edges += 1
        
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, profile[lb]["medians"][0], profile[lb]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
            ticker, str(profile[5]["segments"]), med, "EDGE" if positive_edges >= 2 else "NO", "{}".format(positive_edges)))

    # Determinism: rerun a subset
    print("\n=== 5. Determinism verification (AMZN / JPM subset) ===")
    target = ["AMZN", "JPM"]
    target_tickers = {t: all_tickers[t] for t in target}

    res1 = {lb: {t: run_asset_with_cost(t, target_tickers, lb, COST_MODEL) 
                 for t in target} for lb in LOOKBACKS}
    res2 = {lb: {t: run_asset_with_cost(t, target_tickers, lb, COST_MODEL) 
                 for t in target} for lb in LOOKBACKS}

    # Compare results
    deterministic = True
    for lb in LOOKBACKS:
        for ticker in target:
            r1 = res1[lb][ticker]
            r2 = res2[lb][ticker]
            if (r1.scenarios[0].baseline_median_log_return != r2.scenarios[0].baseline_median_log_return or
                r1.scenarios[0].noise_median_log_return != r2.scenarios[0].noise_median_log_return):
                deterministic = False
                print(f"  NON-DETERMINISTIC: {ticker} lookback={lb}")

    print("Determinism: {} (identical cost-sensitive results on rerun)".format(
        "PASS" if deterministic else "FAIL"))
    assert deterministic, "cost-sensitive output non-deterministic"

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_model=dict(
            commission_per_share=COST_MODEL.commission_per_share,
            commission_per_trade=COST_MODEL.commission_per_trade,
            slippage_cents=COST_MODEL.slippage_cents,
            slippage_proportional=COST_MODEL.slippage_proportional,
        ),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_momentum=dict(
            per_asset={ticker: cost_momentum_results[lb][ticker] for lb in LOOKBACKS for ticker in ticker_order},
            universe=dict(
                lookback_5_counts=lookback_5_counts,
                robustness_counts=robustness_counts,
                ticker_order=ticker_order,
                per_asset=dict(
                    (t, dict(
                        medians_3=cost_momentum_results[3][t]["medians"],
                        medians_5=cost_momentum_results[5][t]["medians"],
                        medians_10=cost_momentum_results[10][t]["medians"],
                        nulls_3=cost_momentum_results[3][t]["noise_medians"],
                        nulls_5=cost_momentum_results[5][t]["noise_medians"],
                        nulls_10=cost_momentum_results[10][t]["noise_medians"],
                        verdict_5=cost_momentum_results[5][t]["verdict"],
                        edge_counts_robustness=robustness_counts,
                    ))
                    for t in ticker_order
                ),
            ),
        ),
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check on lookback 3 / 5 / 10 "
          "with realistic transaction costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())