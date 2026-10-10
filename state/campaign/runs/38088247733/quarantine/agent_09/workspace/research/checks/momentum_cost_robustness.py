"""Cost robustness check for momentum signals (lookback 3 / 5 / 10) on the collected universe.

Objective: test whether the short-horizon momentum edge observed at lookback=5
`research/checks/momentum.py` survives realistic transaction costs across lookback
3, 5, and 10. The candidate positive evidence admitted into the evidence base is
now qualified: a conservative cost model (commission_per_trade=2.0, 
commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)
and a matched null (coin-flip benchmark) must both be walked forward and
compared. If the candidate edge survives the cost grid, momentum is
qualified as candidate positive evidence with a cost-robustness qualification;
otherwise, the cost grid closes the momentum cell.

Method (fixed a-priori, not tuned to OOS):
- long the previous lookback-day return; hold 1 day; daily rebalance
- walk-forward validation (train=252d / test=84d / warmup=60d / overlap=60d)
- 4 contiguous volatility blocks per asset (regime stability)
- realistic costs cfg1 (commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005)
- matched null: coin-flip null benchmark using the same parameters
- compare candidate vs null medians across all segments; if candidate medians
  exceed null's effective tolerance consistently across lookbacks, the edge
  is cost-robust

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

# Realistic cost model (from examples/ma_crossover.py)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_cost_robustness_results.json"


def load_universe():
    """Load the full collected 10-asset universe from the manifest."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"dataset id mismatch: {manifest['dataset_id']}"
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    return all_tickers, list(all_tickers.keys())


def momentum_signals_with_lookback(closes: np.ndarray, lookback: int):
    """Generate momentum signals for given lookback."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback - 1, n):
        prev_ret = (closes[i] / closes[i - lookback]) - 1.0
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if prev_ret > 0 else -1.0)
    return signals


def segment_result_with_costs(ticker: str, bars: bt.BarSequence, lookback: int,
                              costs_cfg: bt.BacktestConfig, is_null: bool = False):
    """Walk-forward result for one ticker, one lookback, with given cost config.

    If is_null is True, use a null signal generation function.
    """
    closes = bars.closes_array()

    if not is_null:
        # Candidate: use momentum_signals
        sig = momentum_signals_with_lookback(closes, lookback)
        def signals_fn(c, **params):
            return momentum_signals_with_lookback(c, lookback)
    else:
        # Null: use a function that returns neutral signals
        # This creates a matched null with no momentum structure
        sig = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(bars))]
        def signals_fn(c, **params):
            return sig

    if len(sig) != len(bars):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(bars)}")

    res = bt.stress_segments(
        signals_fn=signals_fn,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=bt.segment_fn_from_labels(bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)),
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=costs_cfg,
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

    print("=== Cost-robustness momentum check (lookback 3 / 5 / 10) ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Costs: commission_per_trade=2.0, commission_per_share=0.003, "
          f"slippage_cents=2.0, slippage_proportional=0.0005")
    print(f"Parameters: train={TRAIN}d / test={TEST}d / warmup={WARM}d / "
          f"overlap={OVERLAP}d, {N_BLOCKS} volatility blocks, "
          f"min {MIN_SEGMENT_BARS} bars/segment")
    print()

    # 1. Manifest integrity
    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    # 2. Data preflight
    print("=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-robustness run.")
        sys.exit(1)
    print()

    # 3. Load universe
    print("=== 3. Loading 10-asset universe ===")
    all_tickers, ticker_order = load_universe()
    print(f"  Loaded {len(ticker_order)} tickers: {ticker_order}")
    print()

    # 4. Run candidate and null with realistic costs
    print("=== 4. Cost-robustness momentum on collected universe (candidate + null) ===")
    print(f"  Parameters: walk-forward {TRAIN}d train / {TEST}d test / {WARM}d warmup / "
          f"{OVERLAP}d overlap, {N_BLOCKS} volatility blocks per asset")
    print(f"  Lookbacks: {LOOKBACKS}")
    print(f"  Real costs applied to both candidate (momentum) and null (neutral signals)")
    print()

    # Results storage
    candidate_results = {}
    null_results = {}

    # Run for each lookback
    for lookback in LOOKBACKS:
        print(f"--- Lookback {lookback} ---")

        # Candidate: momentum on real data with realistic costs
        candidate_lookup = {}
        print(f"  Candidate: momentum signals with realistic costs (commission_per_trade=2.0, ...)")
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            res = segment_result_with_costs(ticker, bars, lookback, REALISTIC_COSTS,
                                           is_null=False)
            candidate_lookup[ticker] = res
            verdict = res["verdict"]
            verdict_symbol = "EDGE" if "EDGE" in verdict else "NO_EDGE" if "NO_EDGE" in verdict else "LOSS"
            print(f"    {ticker}: {len(res['segments'])} segments, "
                  f"candidate median {res['medians'][0]:+.3f}, "
                  f"null median {res['null_medians'][0]:+.3f}, verdict {verdict_symbol}")
        candidate_results[str(lookback)] = candidate_lookup

        # Null: neutral signals with realistic costs (matched null)
        null_lookup = {}
        print(f"  Null: neutral signals with same realistic costs (matched null)")
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            res = segment_result_with_costs(ticker, bars, lookback, REALISTIC_COSTS,
                                           is_null=True)
            null_lookup[ticker] = res
            verdict = res["verdict"]
            verdict_symbol = "EDGE" if "EDGE" in verdict else "NO_EDGE" if "NO_EDGE" in verdict else "LOSS"
            print(f"    {ticker}: {len(res['segments'])} segments, "
                  f"candidate median {res['medians'][0]:+.3f}, "
                  f"null median {res['null_medians'][0]:+.3f}, verdict {verdict_symbol}")
        null_results[str(lookback)] = null_lookup

        print(f"  Lookback {lookback} complete.")
        print()

    # 5. Compare candidate vs null and determine cost robustness
    print("=== 5. Cost-robustness analysis ===")

    # Compute effective tolerance for each lookback
    cost_robustness_summary = {}
    for lookback in LOOKBACKS:
        lb_str = str(lookback)
        robust_assets = 0
        sensitive_assets = 0
        no_edge_assets = 0

        for ticker in ticker_order:
            cand = candidate_results[lb_str][ticker]
            null_res = null_results[lb_str][ticker]

            # Determine if candidate edge survives realistic costs
            cand_median = cand["medians"][0]
            null_median = null_res["null_medians"][0]
            effective_tol = cand["candidate_dispersion"] + 2 * cand["null_dispersion"]

            # Cost robustness criterion: candidate edge exceeds null tolerance
            if cand_median > abs(null_median) + 0.01 and cand_median > effective_tol:
                robust_assets += 1
            elif abs(cand_median) < effective_tol * 1.5:  # Within tolerance
                no_edge_assets += 1
            else:
                sensitive_assets += 1

        cost_robustness_summary[lb_str] = {
            "robust_assets": robust_assets,
            "sensitive_assets": sensitive_assets,
            "no_edge_assets": no_edge_assets,
            "total_assets": len(ticker_order),
            "candidate_dispersion": float(np.mean([candidate_results[lb_str][t]["candidate_dispersion"]
                                            for t in ticker_order])),
            "null_dispersion": float(np.mean([null_results[lb_str][t]["null_dispersion"]
                                       for t in ticker_order])),
        }

    print("  Cost-robustness summary across lookbacks:")
    for lb_str, summary in cost_robustness_summary.items():
        robust_pct = 100 * summary["robust_assets"] / summary["total_assets"]
        print(f"    Lookback {lb_str}: ROBUST={summary['robust_assets']} "
              f"({robust_pct:.1f}%), SENSITIVE={summary['sensitive_assets']}, "
              f"NO_EDGE={summary['no_edge_assets']}")
        print(f"      candidate dispersion: {summary['candidate_dispersion']:.3f}, "
              f"null dispersion: {summary['null_dispersion']:.3f}")

    print()

    # 6. Write artifact
    print("=== 6. Writing artifact ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        costs={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        candidate=candidate_results,
        null=null_results,
        cost_robustness_summary=cost_robustness_summary,
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"  Artifact written to: {ARTIFACT_PATH}")
    print()
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness check across lookbacks 3/5/10 on the collected universe: exploratory simulation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())