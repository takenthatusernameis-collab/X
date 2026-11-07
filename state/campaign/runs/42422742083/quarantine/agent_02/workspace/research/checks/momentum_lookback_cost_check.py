from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
COST_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_cost_results.json"

def cost_config(zero_cost: bool = True) -> bt.BacktestConfig:
    """Return BacktestConfig matching the realistic cost model from examples."""
    if zero_cost:
        return bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
    else:
        return bt.BacktestConfig(
            initial_capital=1e6, warmup_periods=WARM,
            commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005,
        )

def segment_result(ticker, bars, lookback, cfg) -> dict:
    """Walk-forward regime-stability result for one ticker, lookback, and cost config."""
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

def lookback_verdict(sr: dict) -> str:
    """Verdict for one lookback's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"

def robustness_verdict(lookback_results: dict) -> tuple[str, dict]:
    """Verdict on whether the edge is lookback-robust for one asset."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts

def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

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
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost-sensitivity sweep (lookback 3/5/10, zero vs realistic) ===")
    print("  Parameters: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load all 10 tickersall_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    # Run for both cost configurationscost_results = {}
    for cost_label, cfg in [("zero_cost", cost_config(zero_cost=True)),
                          ("realistic_cost", cost_config(zero_cost=False))]:
        print(f"\n  Running with {cost_label}...\n")
        cost_results[cost_label] = {}

        # For each cost config, run lookback sweep    for lb in LOOKBACKS:
            lb_results = {}
            for ticker in manifest["universe"]:
                bars = all_tickers[ticker]
                res = segment_result(ticker, bars, lb, cfg)
                lb_results[ticker] = res
            cost_results[cost_label][str(lb)] = lb_results

    print("\n=== 4. Summary of cost-sensitivity analysis ===")
    print("  Universe-level comparison: lookback 3/5/10 edge presence with zero vs realistic costs")

    # For each asset, compare edge status between zero and realistic costscomparison = {}
    for ticker in manifest["universe"]:
        zero_ro = robustness_verdict(cost_results["zero_cost"][ticker])
        real_ro = robustness_verdict(cost_results["realistic_cost"][ticker])

        comparison[ticker] = {
            "zero_cost": {
                "robustness_verdict": zero_ro[0],
                "edge_counts": zero_ro[1],
            },
            "realistic_cost": {
                "robustness_verdict": real_ro[0],
                "edge_counts": real_ro[1],
            },
            "edge_change": zero_ro[0] != real_ro[0] and real_ro[0] in ("SENSITIVE", "NO_EDGE")
        }

    print("\n  Per-asset cost-sensitivity results:")
    for ticker in manifest["universe"]:
        c = comparison[ticker]
        print(f"    {ticker}: zero_cost={c['zero_cost']['robustness_verdict']} -> "
              f"realistic_cost={c['realistic_cost']['robustness_verdict']} "
              f"{'(edge lost)' if c['edge_change'] else ''}")

    # Robustness verdict: edge is robust if it appears at >= 2 lookbacks under realistic costsrobust_assets = {
        t: comparison[t]["realistic_cost"]["robustness_verdict"] == "ROBUST"
        for t in manifest["universe"]
    }
    robust_count = sum(robust_assets.values())

    print(f"\n  Summary: {robust_count} assets show lookback-robust edge under realistic costs\n")

    # Create artifactartifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        comparison=comparison,
        universe=dict(
            ticker_order=list(manifest["universe"]),
            robust_under_realistic=robust_assets,
            robust_count=robust_count,
            robustness_verdict="ROBUST" if robust_count >= 2 else ("SENSITIVE" if robust_count == 1 else "NO_EDGE"),
        ),
    )
    with open(COST_ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {COST_ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback cost-sensitivity check across the collected "
          "universe: exploratory simulation.")

    return 0

if __name__ == "__main__":
    sys.exit(main())