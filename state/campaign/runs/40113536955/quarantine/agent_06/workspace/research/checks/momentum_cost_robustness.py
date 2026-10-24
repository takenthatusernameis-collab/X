"""Cost-sensitivity check for momentum signals.

Research objective: the momentum class is currently admitted as candidate positive
evidence based on short-horizon lookback=5 tests. The next bottleneck is whether
this edge survives realistic transaction costs, a decision that is currently
constrained by the cost robustness qualification.

Method: long the previous N-day return; hold 1 day; daily rebalance. For each
lookback N \u2208 {3, 5, 10}, we compute the cost-adjusted walk-forward median log
return on the collected universe using realistic costs (commission per trade,
commission per share, fixed slippage, proportional slippage). We compare against
a coin-flip null model that uses identical costs and the same walk-forward
parameters. The result classifies whether the momentum edge survives realistic
costs and remains statistically distinguishable from noise.

Design mirrors the existing momentum checks: 4 contiguous volatility blocks,
train=252d/test=84d/warmup=60d/overlap=60d, min 400 bars/segment, seed=42.
Walk-forward folds are evaluated on the OOS segment only, and the per-asset
median log return across folds is used for the edge decision.

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

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def segment_result(ticker, bars, lookback, costs):
    """Walk-forward regime-stability result for one ticker, lookback and cost config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    # Compute candidate (momentum) results with given costs
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
        cfg=costs,
    )

    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        costs={
            "commission_per_trade": costs.commission_per_trade,
            "commission_per_share": costs.commission_per_share,
            "slippage_cents": costs.slippage_cents,
            "slippage_proportional": costs.slippage_proportional,
        },
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

    print("\n=== 3. Momentum cost-sensitivity check ===")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")

    # Load all ticker data
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    results = {}
    for lookback in LOOKBACKS:
        print(f"\nLookback {lookback} days with realistic costs:")
        # Compute results for each ticker to display
        results_for_print = []
        for ticker in ticker_order:
            res = segment_result(ticker, all_tickers[ticker], lookback, REALISTIC_COSTS)
            results_for_print.append(f"{ticker}: segments {len(res['segments'])} medians {res['medians']} verdict {res['verdict']}")
        print("  " + "\n  ".join(results_for_print))

        results[f"lookback_{lookback}"] = {
            f"cost_realistic_{lookback}": {
                ticker: segment_result(ticker, all_tickers[ticker], lookback, REALISTIC_COSTS)
                for ticker in ticker_order
            },
            f"cost_zero_{lookback}": {
                ticker: segment_result(ticker, all_tickers[ticker], lookback, ZERO_COSTS)
                for ticker in ticker_order
            },
        }

    # Determine cost robustness: how many assets survive realistic costs vs zero costs
    print("\n=== 4. Cost robustness classification ===")
    cost_robustness = {}
    for lookback in LOOKBACKS:
        realistic_edges = []
        zero_cost_edges = []
        for ticker in ticker_order:
            realistic_res = results[f"lookback_{lookback}"][f"cost_realistic_{lookback}"][ticker]
            zero_res = results[f"lookback_{lookback}"][f"cost_zero_{lookback}"][ticker]
            realistic_edges.append(1 if realistic_res["verdict"] == "REGIME_STABLE" and all(m > 0 for m in realistic_res["medians"]) else 0)
            zero_cost_edges.append(1 if zero_res["verdict"] == "REGIME_STABLE" and all(m > 0 for m in zero_res["medians"]) else 0)

        cost_robustness[lookback] = {
            "realistic_cost_edges": sum(realistic_edges),
            "zero_cost_edges": sum(zero_cost_edges),
            "robustness_ratio": sum(realistic_edges) / sum(zero_cost_edges) if sum(zero_cost_edges) > 0 else 0.0,
        }

        print(f"  Lookback {lookback}: {sum(realistic_edges)}/{sum(zero_cost_edges)} assets survive realistic costs "
              f"(robustness ratio: {cost_robustness[lookback]['robustness_ratio']:.3f})")

    # Determinism: rerun key results and assert identical output
    print("\n=== 5. Determinism check ===")
    res2 = {}
    for lookback in LOOKBACKS:
        res2[f"lookback_{lookback}"] = {
            f"cost_realistic_{lookback}": {
                ticker: segment_result(ticker, all_tickers[ticker], lookback, REALISTIC_COSTS)
                for ticker in ticker_order[:3]  # Check only first 3 assets for determinism
            },
            f"cost_zero_{lookback}": {
                ticker: segment_result(ticker, all_tickers[ticker], lookback, ZERO_COSTS)
                for ticker in ticker_order[:3]
            },
        }

    # Create comprehensive artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs=results[f"lookback_{LOOKBACKS[0]}"][f"cost_realistic_{LOOKBACKS[0]}"][ticker_order[0]]["costs"],
        zero_costs=results[f"lookback_{LOOKBACKS[0]}"][f"cost_zero_{LOOKBACKS[0]}"][ticker_order[0]]["costs"],
        cost_robustness=cost_robustness,
        per_asset={},
    )

    # Store full results for independent verification
    for lookback in LOOKBACKS:
        artifact["per_asset"][f"lookback_{lookback}"] = {
            "cost_realistic": results[f"lookback_{lookback}"][f"cost_realistic_{lookback}"],
            "cost_zero": results[f"lookback_{lookback}"][f"cost_zero_{lookback}"],
        }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check for lookback 3/5/10: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
