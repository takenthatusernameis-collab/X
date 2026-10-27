"""Real-data momentum lookback sweep across the collected universe with cost sensitivity.

Objective: test whether the momentum edge observed at lookback=5
(`research/checks/momentum.py`, activation 37318950814) is robust to the
lookback parameter AND survives conservative transaction-cost stress on the
collected real-data universe.

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
daily rebalance) is walk-forward validated (train=252d/test=84d/warmup=60d/
overlap=60d) inside each of the 4 contiguous volatility blocks of the real-data
regime family, for lookback in {3, 5, 10}. Each lookback's per-segment median
log return is compared against a coin-flip sign null benchmark run on the same
segments. The synthetic generator cannot answer this because it contains no
momentum structure, so the null must be drawn from the collected data itself.

Additionally, each lookback run is executed under two cost regimes:
- Baseline zero-cost: commission_per_trade=0, commission_per_share=0,
  slippage_cents=0, slippage_proportional=0
- Conservative-cost: commission_per_trade=0.01, commission_per_share=0.001,
  slippage_cents=1, slippage_proportional=0.005

A lookback-robust edge is one where the uniformly-positive REGIME_STABLE
pattern at lookback=5 also appears at >= 2 of the 3 lookbacks for an asset;
a lookback-sensitive result (edge only at lookback=5) is a parameter-fragility
warning even when the lookback=5 edge is genuine. Either outcome directly
informs whether momentum should be admitted to the evidence base as candidate
positive evidence under realistic costs.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as `momentum.py`. Costs: conservative transaction
costs as above. Determinism is asserted via an internal independent recomputation
of the AMZN/JPM subset under both cost regimes.

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


def segment_result_with_costs(ticker, bars, lookback, cost_config):
    """Walk-forward regime-stability result for one ticker and one lookback under given costs."""
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
        cfg=cost_config,
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
    """Robustness verdict for one lookback's scenario result.

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
    """Verdict on whether the edge is lookback-robust for one asset under given costs.

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
        print(f"  {entry['ticker']}: {"OK" if actual == entry["checksum_sha256"] else "MISMATCH"}")
    print(f"  dataset: {manifest["dataset_id"]} | collection: "
          f"{manifest["collection_date"]} | universe: {len(manifest["universe"])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost model definition ===")
    zero_cost = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
        warmup_periods=60,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )
    conservative_cost = bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=0.01,
        commission_per_share=0.001,
        slippage_cents=1.0,
        slippage_proportional=0.005,
        warmup_periods=60,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )
    print(f"  Zero-cost: commission_per_trade={zero_cost.commission_per_trade}, "
          f"commission_per_share={zero_cost.commission_per_share}, "
          f"slippage_cents={zero_cost.slippage_cents}, "
          f"slippage_proportional={zero_cost.slippage_proportional}")
    print(f"  Conservative-cost: commission_per_trade={conservative_cost.commission_per_trade}, "
          f"commission_per_share={conservative_cost.commission_per_share}, "
          f"slippage_cents={conservative_cost.slippage_cents}, "
          f"slippage_proportional={conservative_cost.slippage_proportional}")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments. Costs: baseline zero and conservative.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep under both cost regimes.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [cost][lookback][ticker][segment index].
    results = {
        "zero_cost": {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, zero_cost) 
                           for t in ticker_order} for lb in LOOKBACKS},
        "conservative_cost": {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, conservative_cost) 
                                   for t in ticker_order} for lb in LOOKBACKS},
    }

    # Per-asset robustness verdicts for both cost regimes.
    robustness_results = {
        "zero_cost": {},
        "conservative_cost": {},
    }
    for cost_regime in ["zero_cost", "conservative_cost"]:
        for ticker in ticker_order:
            profile = {str(lb): results[cost_regime][lb][ticker] for lb in LOOKBACKS}
            v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
            robustness_results[cost_regime][ticker] = {
                "edge_counts": counts,
                "robustness_verdict": v,
                "lookback_medians": {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS},
                "lookback_null_medians": {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS},
                "verdicts": {str(lb): profile[str(lb)]["verdict"] for lb in LOOKBACKS},
            }

    print("\n=== 5. Results summary ===")
    print("  Lookback 5 (reference) verdict counts across universe (zero-cost):")
    lookback_5_counts_zero = {v: 0 for v in ["REGIME_STABLE", "CONSISTENT_WITH_NOISE", "REGIME_DEPENDENT", "REGIME_STABLE_LOSS"]}
    for ticker in ticker_order:
        verdict = robustness_results["zero_cost"][ticker]["verdicts"]["5"]
        lookback_5_counts_zero[verdict] += 1
    print(f"    {lookback_5_counts_zero}")
    print("  Lookback-robustness verdict counts (zero-cost):")
    print(f"    ROBUST={sum(1 for v in robustness_results['zero_cost'].values() if v['robustness_verdict'] == 'ROBUST')} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={sum(1 for v in robustness_results['zero_cost'].values() if v['robustness_verdict'] == 'SENSITIVE')} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={sum(1 for v in robustness_results['zero_cost'].values() if v['robustness_verdict'] == 'NO_EDGE')}\n")

    print("  Lookback-robustness verdict counts (conservative-cost):")
    print(f"    ROBUST={sum(1 for v in robustness_results['conservative_cost'].values() if v['robustness_verdict'] == 'ROBUST')} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={sum(1 for v in robustness_results['conservative_cost'].values() if v['robustness_verdict'] == 'SENSITIVE')} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={sum(1 for v in robustness_results['conservative_cost'].values() if v['robustness_verdict'] == 'NO_EDGE')}\n")

    print("  Per-asset perturbation profiles:")
    for ticker in ticker_order:
        p = robustness_results["conservative_cost"][ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p["lookback_medians"][str(lb)][0], p["verdicts"][str(lb)])
            for lb in LOOKBACKS
        )
        print("    {:6s} medians3510={:<40s} rob_zero={:<8s} rob_cons={:<8s} "
              "edge_counts_zero={} edge_counts_cons={}".format(
            ticker, med, robustness_results["zero_cost"][ticker]["robustness_verdict"],
            robustness_results["conservative_cost"][ticker]["robustness_verdict"],
            robustness_results["zero_cost"][ticker]["edge_counts"],
            robustness_results["conservative_cost"][ticker]["edge_counts"]))
    print("")

    # Determinism: rerun the target subset and assert identical output under both cost regimes.
    res2_zero = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, zero_cost) 
                      for t in ["AMZN", "JPM"]} for lb in LOOKBACKS}
    res2_cons = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, conservative_cost) 
                      for t in ["AMZN", "JPM"]} for lb in LOOKBACKS}
    out1_zero = json.dumps({lb: {t: results["zero_cost"][lb][t] for t in ["AMZN", "JPM"]}
                           for lb in LOOKBACKS}, sort_keys=True)
    out1_cons = json.dumps({lb: {t: results["conservative_cost"][lb][t] for t in ["AMZN", "JPM"]}
                           for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps(res2_zero, sort_keys=True)
    out2_cons = json.dumps(res2_cons, sort_keys=True)
    print("determinism: zero-cost r1 == r2: {}".format(out1_zero == out2_zero))
    print("determinism: conservative-cost r1 == r2: {}".format(out1_cons == out2_cons))
    assert out1_zero == out2_zero, "non-deterministic zero-cost output"
    assert out1_cons == out2_cons, "non-deterministic conservative-cost output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost=dict(
            per_asset=robustness_results["zero_cost"],
            robustness_counts={
                v: sum(1 for asset in robustness_results["zero_cost"].values() if asset["robustness_verdict"] == v)
                for v in ["ROBUST", "SENSITIVE", "NO_EDGE"]
            },
        ),
        conservative_cost=dict(
            per_asset=robustness_results["conservative_cost"],
            robustness_counts={
                v: sum(1 for asset in robustness_results["conservative_cost"].values() if asset["robustness_verdict"] == v)
                for v in ["ROBUST", "SENSITIVE", "NO_EDGE"]
            },
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with cost sensitivity across the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())