"""Cost-sensitivity check for momentum with lookback 3/5/10.

Objective: test whether the candidate positive momentum edge survives conservative
transaction-cost stress on the collected real-data universe.

Cost model: realistic trading costs as defined in examples/ma_crossover_real_data.py:
  commission_per_trade: 2.0
  commission_per_share: 0.003
  slippage_cents: 2.0
  slippage_proportional: 0.0005

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return;
hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward windows and
volatility blocks as `momentum.py` and `momentum_lookback_sweep.py`.
The realistic cost model is applied to every segment of the walk-forward.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sweep_results.json"
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)
COIN_FLIP_COST_CONFIG = bt.BacktestConfig(
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
    margin_call_liquidate=False,
)
def segment_result_with_cost(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback
    using realistic costs. The cost model is applied to each segment's walk-forward.
    """
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
        cfg=REALISTIC_COST_CONFIG,
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
def lookback_verdict(sr):
    """Robustness verdict for one lookback's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"
def robustness_verdict(lookback_results):
    """Verdict on whether the edge survives cost stress for one asset."""
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

    print("\n=== 3. Cost model definition ===")
    print(f"  commission_per_trade: {REALISTIC_COST_CONFIG.commission_per_trade}")
    print(f"  commission_per_share: {REALISTIC_COST_CONFIG.commission_per_share}")
    print(f"  slippage_cents: {REALISTIC_COST_CONFIG.slippage_cents}")
    print(f"  slippage_proportional: {REALISTIC_COST_CONFIG.slippage_proportional}")
    print("  realistic cost model applied to all momentum segments with lookback 3/5/10")

    print("\n=== 4. Momentum cost sensitivity across collected universe ===")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs a coin-flip null "
          f"on the same segments, realistic cost model applied\n")

    # Load all tickers from manifest
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]

    # Cost sensitivity results indexed [lookback][ticker][segment index].
    lookback_results = {}
    for lb in LOOKBACKS:
        ticker_results = {}
        for ticker, bars in all_tickers.items():
            result = segment_result_with_cost(ticker, bars, lb)
            ticker_results[ticker] = result
        lookback_results[lb] = ticker_results

    # Per-asset robustness analysis under cost stress
    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in all_tickers.keys():
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts[v] += 1
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  cost-robustness verdict counts: "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(survives costs at >=2 lookbacks) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(survives costs at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")
    print("  Per-asset cost-robustness profiles:")
    for ticker in all_tickers.keys():
        p = per_asset[ticker]
        p["lookback_medians"] = {str(lb): p[str(lb)]["medians"] for lb in LOOKBACKS}
        p["lookback_null_medians"] = {str(lb): p[str(lb)]["null_medians"] for lb in LOOKBACKS}
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Generate coin-flip null benchmarks for each lookback
    print("\n=== 5. Coin-flip null benchmarks under cost stress ===")
    null_benches = {}
    for lb in LOOKBACKS:
        np.random.seed(SEED)
        ticker_nulls = {}
        for ticker, bars in all_tickers.items():
            null_bench = bt.noise_benchmark(
                bars=bars,
                param_grid=[{"lookback": lb}],
                train_window=TRAIN,
                test_window=TEST,
                warmup=0,
                overlap_window=OVERLAP,
                periods_per_year=252,
                cfg=COIN_FLIP_COST_CONFIG,
                seed=SEED + hash(ticker) % 10000,
            )
            ticker_nulls[ticker] = null_bench
        null_benches[lb] = ticker_nulls

    print("  Null bench medians (realistic cost model):")
    for lb in LOOKBACKS:
        medians = []
        for ticker, bench in null_benches[lb].items():
            medians.append(bench.baseline_median_log_return)
        print(f"    lookback {lb}: {[f'{m:+.3f}' for m in medians]}")

    # Determinism check
    print("\n=== 6. Determinism check (AMZN/JPM subset) ===")
    target = ["AMZN", "JPM"]
    res2 = {lb: {t: segment_result_with_cost(t, all_tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact
    print("\n=== 7. Cost-sensitivity artifact written ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_model=dict(
            commission_per_trade=REALISTIC_COST_CONFIG.commission_per_trade,
            commission_per_share=REALISTIC_COST_CONFIG.commission_per_share,
            slippage_cents=REALISTIC_COST_CONFIG.slippage_cents,
            slippage_proportional=REALISTIC_COST_CONFIG.slippage_proportional,
        ),
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=list(all_tickers.keys()),
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
                for t in all_tickers.keys()
            ),
        ),
        cost_null_benches=dict(
            (str(lb), {t: bench.baseline_median_log_return
                      for t, bench in null_benches[lb].items()})
            for lb in LOOKBACKS
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity check across the collected universe: "
          "exploratory simulation.")

    return 0
if __name__ == "__main__":
    sys.exit(main())