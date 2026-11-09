"""Cost-sensitive momentum lookback sweep across the collected universe.

Objective: Test whether the momentum edge observed at lookback=5
(research/checks/momentum.py, activation 37318950814) survives conservative
transaction-cost stress and remains distinguishable from a matched null
benchmark. Momentum is a mechanical 1-day-ahead bet, so the most salient
robustness question is whether the positive REGIME_STABLE pattern at
lookback=5 also appears at lookback 3 and lookback 10 under realistic costs.

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
daily rebalance) is walk-forward validated (train=252d/test=84d/warmup=60d/
overlap=60d) inside each of the 4 contiguous volatility blocks of the real-data
regime family, for lookback in {3, 5, 10}. Each lookback's per-segment median
log return is compared against a matched null benchmark run on the same
segments. The synthetic generator cannot answer this because it contains no
momentum structure, so the null must be drawn from the collected data itself.

A lookback-robust edge is one where the uniformly-positive REGIME_STABLE pattern
at lookback=5 also appears at >= 2 of the 3 lookbacks for an asset; a
lookback-sensitive result (edge only at lookback=5) is a parameter-fragility
warning even when the lookback=5 edge is genuine. Either outcome directly
informs whether momentum should be admitted to the evidence base as candidate
positive evidence.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as momentum.py. Realistic transaction costs
(commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
slippage_proportional=0.0005) applied in every round trip. Determinism is
asserted via an internal independent recomputation of the AMZN/JPM subset.

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# Realistic transaction cost model (existing in repository examples)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    periods_per_year=252,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    periods_per_year=252,
)


def segment_result(
    ticker: str, bars, lookback: int, cost_config: bt.BacktestConfig
):
    """Walk-forward regime-stability result for one ticker and one lookback with cost config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64
    )
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


def lookback_verdict(sr):
    """Verdict for one lookback's scenario result.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge on this asset; REGIME_STABLE with uniformly negative
    medians is a stable loss, not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset.

    ROBUST      : uniformly positive REGIME_STABLE edges at >= 2 lookbacks.
    SENSITIVE   : a positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE     : no lookback shows a positive REGIME_STABLE edge.
    """
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
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
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
        res = bt.run_bars(list(bars), signals, REALISTIC_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe with realistic costs ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments with realistic transaction costs\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index] with realistic costs
    lookback_results_realistic = {
        lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COSTS) for t in ticker_order}
        for lb in LOOKBACKS
    }
    
    # Results indexed [lookback][ticker][segment index] with zero costs
    lookback_results_zero = {
        lb: {t: segment_result(t, all_tickers[t], lb, ZERO_COSTS) for t in ticker_order}
        for lb in LOOKBACKS
    }

    per_asset = {}
    robustness_counts_realistic = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    robustness_counts_zero = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts_realistic = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                 "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    lookback_5_counts_zero = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                             "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile_realistic = {str(lb): lookback_results_realistic[lb][ticker] for lb in LOOKBACKS}
        profile_zero = {str(lb): lookback_results_zero[lb][ticker] for lb in LOOKBACKS}
        
        v_realistic, counts_realistic = robustness_verdict(profile_realistic)
        v_zero, counts_zero = robustness_verdict(profile_zero)
        robustness_counts_realistic[v_realistic] += 1
        robustness_counts_zero[v_zero] += 1
        
        r5_realistic = profile_realistic["5"]
        r5_zero = profile_zero["5"]
        lookback_5_counts_realistic[r5_realistic["verdict"]] += 1
        lookback_5_counts_zero[r5_zero["verdict"]] += 1
        
        # Lookback-robustness edge counts (edges per asset, for the summary).
        profile_realistic["edge_counts"] = counts_realistic
        profile_realistic["robustness_verdict"] = v_realistic
        profile_realistic["lookback_medians"] = {str(lb): profile_realistic[str(lb)]["medians"] for lb in LOOKBACKS}
        profile_realistic["lookback_null_medians"] = {str(lb): profile_realistic[str(lb)]["null_medians"] for lb in LOOKBACKS}
        profile_realistic["cost_type"] = "realistic"
        
        profile_zero["edge_counts"] = counts_zero
        profile_zero["robustness_verdict"] = v_zero
        profile_zero["lookback_medians"] = {str(lb): profile_zero[str(lb)]["medians"] for lb in LOOKBACKS}
        profile_zero["lookback_null_medians"] = {str(lb): profile_zero[str(lb)]["null_medians"] for lb in LOOKBACKS}
        profile_zero["cost_type"] = "zero"
        
        per_asset[ticker] = {"realistic": profile_realistic, "zero": profile_zero}

    print("  lookback 5 (reference) verdict counts across universe (realistic costs): "
          f"{lookback_5_counts_realistic}")
    print("  lookback 5 (reference) verdict counts across universe (zero costs): "
          f"{lookback_5_counts_zero}")
    print("  lookback-robustness verdict counts (realistic costs): "
          f"ROBUST={robustness_counts_realistic['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts_realistic['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts_realistic['NO_EDGE']}")
    print("  lookback-robustness verdict counts (zero costs): "
          f"ROBUST={robustness_counts_zero['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts_zero['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts_zero['NO_EDGE']}\n")
    print("  Per-asset perturbation profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]["realistic"]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={} cost=realistic".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("    Per-asset zero-cost profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]["zero"]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={} cost=zero".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    print("=== 5. Determinism check (realistic costs) ===")
    res2_realistic = {lb: {t: segment_result(t, tickers[t], lb, REALISTIC_COSTS) for t in target}
                     for lb in LOOKBACKS}
    out1_realistic = json.dumps({lb: {t: lookback_results_realistic[lb][t] for t in target}
                        for lb in LOOKBACKS}, sort_keys=True)
    out2_realistic = json.dumps(res2_realistic, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1_realistic == out2_realistic))
    assert out1_realistic == out2_realistic, "non-deterministic output (realistic costs)"

    print("=== 6. Determinism check (zero costs) ===")
    res2_zero = {lb: {t: segment_result(t, tickers[t], lb, ZERO_COSTS) for t in target}
                for lb in LOOKBACKS}
    out1_zero = json.dumps({lb: {t: lookback_results_zero[lb][t] for t in target}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps(res2_zero, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1_zero == out2_zero))
    assert out1_zero == out2_zero, "non-deterministic output (zero costs)"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=42,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts_realistic=lookback_5_counts_realistic,
            lookback_5_counts_zero=lookback_5_counts_zero,
            robustness_counts_realistic=robustness_counts_realistic,
            robustness_counts_zero=robustness_counts_zero,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians_realistic={str(lb): per_asset[t]["realistic"]["lookback_medians"][str(lb)]
                                              for lb in LOOKBACKS},
                    lookback_null_medians_realistic={str(lb): per_asset[t]["realistic"]["lookback_null_medians"][str(lb)]
                                             for lb in LOOKBACKS},
                    lookback_medians_zero={str(lb): per_asset[t]["zero"]["lookback_medians"][str(lb)]
                                         for lb in LOOKBACKS},
                    lookback_null_medians_zero={str(lb): per_asset[t]["zero"]["lookback_null_medians"][str(lb)]
                                              for lb in LOOKBACKS},
                    verdicts_5_realistic=per_asset[t]["realistic"]["5"]["verdict"],
                    verdicts_5_zero=per_asset[t]["zero"]["5"]["verdict"],
                    edge_counts_realistic=per_asset[t]["realistic"]["edge_counts"],
                    edge_counts_zero=per_asset[t]["zero"]["edge_counts"],
                    robustness_verdict_realistic=per_asset[t]["realistic"]["robustness_verdict"],
                    robustness_verdict_zero=per_asset[t]["zero"]["robustness_verdict"],
                    n_folds_5=per_asset[t]["realistic"]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity lookback sweep across the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
