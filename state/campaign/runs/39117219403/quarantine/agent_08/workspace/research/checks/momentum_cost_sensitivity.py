"""Cost sensitivity test for momentum lookback 3/5/10 on the collected universe.

Objective: test whether the qualified momentum edge (lookback=5, 7/10 assets
REGIME_STABLE) survives conservative transaction costs under the realistic cost
model from examples/ma_crossover_real_data.py:
- commission_per_trade=2.0
- commission_per_share=0.003
- slippage_cents=2.0
- slippage_proportional=0.0005

Use matched null (coin-flip) benchmark for each asset/segment combination.
The test runs the same walk-forward regime-stability gate used in
momentum_lookback_sweep.py but with realistic costs and matched null.

Research/simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Insert the project root directory into Python path so we can import research modules
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
REALISTIC_COST_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COST_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"


def segment_result_cost_sensitivity(ticker: str, bars, lookback: int):
    """Walk-forward regime-stability result for one ticker, one lookback, with realistic costs."""
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
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=REALISTIC_COST_CFG,
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
        zero_cost_result=None,  # placeholder, filled later if needed
    )


def robustness_verdict_cost(sr):
    """Verdict on whether the edge survives realistic costs for one lookback's scenario result.
    
    REGIME_STABLE means the edge survived costs (candidate > noise with statistical significance);
    CONSISTENT_WITH_NOISE means costs erased the edge (candidate indistinguishable from noise);
    REGIME_DEPENDENT means the edge is regime-dependent but survived costs;
    REGIME_STABLE_LOSS means the edge survived costs but was uniformly negative.
    """
    return sr["verdict"]


def robustness_verdict_aggregate(lookback_results):
    """Aggregate verdict across lookbacks for one asset.
    
    COST_ROBUST: uniformly positive REGIME_STABLE edge at >=2 lookbacks (edge survives costs).
    COST_SENSITIVE: a positive REGIME_STABLE edge at exactly one lookback only (fragile to costs).
    NO_COST_EDGE: no lookback shows a positive REGIME_STABLE edge (costs erased edge everywhere).
    """
    counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "NO_COST_EDGE": 0}
    for lb in LOOKBACKS:
        verdict = robustness_verdict_cost(lookback_results[str(lb)])
        if verdict == "REGIME_STABLE" and all(m > 0 for m in lookback_results[str(lb)]["medians"]):
            counts["COST_ROBUST"] += 1
        elif verdict == "REGIME_STABLE_LOSS":
            counts["COST_SENSITIVE"] += 1
        else:
            counts["NO_COST_EDGE"] += 1
    if counts["COST_ROBUST"] >= 2:
        return "COST_ROBUST", counts
    if counts["COST_ROBUST"] == 1:
        return "COST_SENSITIVE", counts
    return "NO_COST_EDGE", counts


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

    print("\n=== 3. Leakage review per asset (lookback=5 reference, realistic costs) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, REALISTIC_COST_CFG)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe with realistic costs ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a matched coin-flip null "
          "on the same segments with realistic costs\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with realistic costs.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index].
    lookback_results = {lb: {t: segment_result_cost_sensitivity(t, all_tickers[t], lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    cost_robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "NO_COST_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict_aggregate(profile)
        cost_robustness_counts[v] += 1
        l5 = profile["5"]
        lookback_5_counts[l5["verdict"]] += 1
        profile["cost_robustness_verdict"] = v
        profile["edge_counts"] = counts
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  cost-robustness verdict counts: "
          f"COST_ROBUST={cost_robustness_counts['COST_ROBUST']} "
          f"(edge survives costs at >=2 lookbacks) | "
          f"COST_SENSITIVE={cost_robustness_counts['COST_SENSITIVE']} "
          f"(edge survives costs but is single-lookback sensitive) | "
          f"NO_COST_EDGE={cost_robustness_counts['NO_COST_EDGE']}\n")
    print("  Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} cost_rob={:<12s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["cost_robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result_cost_sensitivity(t, tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            cost_robustness_counts=cost_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t][str(lb)]["medians"] for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t][str(lb)]["null_medians"] for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    cost_robustness_verdict=per_asset[t]["cost_robustness_verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity test across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())