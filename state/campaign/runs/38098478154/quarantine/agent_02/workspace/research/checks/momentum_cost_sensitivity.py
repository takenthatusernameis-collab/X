#!/usr/bin/env python3
"""
Cost-sensitivity check for lookback 3/5/10 momentum under realistic transaction costs.

This is the primary check (not a verification) that runs momentum signals with
realistic transaction costs (from tests/test_engine.py: commission_per_share=0.003,
slippage_cents=5.0, slippage_proportional=0.001) and compares the edge against
a matched null with the same costs.

Primary Question: Does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

Method: Long the previous N-day return; hold 1 day; daily rebalance. Walk-forward
IS/OOS per segment (regime-stable blocks), using realistic BacktestConfig.

A lookback-robust edge is one where the uniformly-positive REGIME_STABLE pattern
survives realistic costs across >= 2 of the 3 lookbacks for an asset; otherwise
it's SENSITIVE (only at one lookback) or NO_EDGE.

Output: state/check_artifacts/momentum_cost_robustness_results.json
"""
from __future__ import annotations

import sys
if sys.version_info < (3, 8):
    print("Python 3.8 or higher is required")
    sys.exit(1)

import json
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
    commission_per_share=0.003,      # 3 bps per share (realistic)
    slippage_cents=5.0,               # 5 cents per share (realistic)
    slippage_proportional=0.001,     # 10 bps proportional slippage (realistic)
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

MATCHED_NULL_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.003,      # Same costs for matched null
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


def momentum_signals_no_costs(closes, lookback):
    """Momentum signals with no transaction costs (reference)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


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
    print("  Realistic costs (from tests/test_engine.py):")
    print(f"    commission_per_share: {REALISTIC_COSTS.commission_per_share}")
    print(f"    slippage_cents: {REALISTIC_COSTS.slippage_cents}")
    print(f"    slippage_proportional: {REALISTIC_COSTS.slippage_proportional}")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs matched null "
          "with SAME costs.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index] with realistic costs
    lookback_results_real = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COSTS)
                                  for t in ticker_order} for lb in LOOKBACKS}

    # Results indexed [lookback][ticker][segment index] with NO costs (reference)
    lookback_results_no_costs = {lb: {t: segment_result(t, all_tickers[t], lb, NO_COSTS)
                                      for t in ticker_order} for lb in LOOKBACKS}

    # Build artifact
    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile_real = {str(lb): lookback_results_real[lb][ticker] for lb in LOOKBACKS}
        profile_no_costs = {str(lb): lookback_results_no_costs[lb][ticker] for lb in LOOKBACKS}

        # Determine cost-robustness verdict
        def edge_counts_for_real_costs(lookback_results):
            counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            for lb in LOOKBACKS:
                sr = lookback_results[str(lb)]
                if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
                    counts["EDGE"] += 1
                elif sr["verdict"] == "REGIME_STABLE_LOSS":
                    counts["LOSS"] += 1
            return counts

        counts_real = edge_counts_for_real_costs(profile_real)
        v_real = "ROBUST" if counts_real["EDGE"] >= 2 else (
            "SENSITIVE" if counts_real["EDGE"] == 1 else "NO_EDGE")
        robustness_counts[v_real] += 1

        r5_real = profile_real["5"]
        lookback_5_counts[r5_real["verdict"]] += 1

        # Combine real-cost and no-cost profiles for artifact
        profile = {
            str(lb): {
                **profile_real[str(lb)],
                "no_costs_medians": profile_no_costs[str(lb)]["medians"],
                "no_costs_nulls": profile_no_costs[str(lb)]["null_medians"],
                "no_costs_verdict": profile_no_costs[str(lb)]["verdict"],
            }
            for lb in LOOKBACKS
        }
        profile["edge_counts_real"] = counts_real
        profile["robustness_verdict_real"] = v_real
        profile["lookback_medians_real"] = {str(lb): profile_real[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians_real"] = {str(lb): profile_real[str(lb)]["null_medians"] for lb in LOOKBACKS}
        profile["lookback_verdicts_real"] = {str(lb): profile_real[str(lb)]["verdict"] for lb in LOOKBACKS}

        per_asset[ticker] = profile

    print("\n=== Cost-robustness summary (REALISTIC COSTS) ===")
    print("  lookback 5 (realistic costs) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  lookback-robustness verdict counts (realistic costs): "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(>=2 lookbacks positive edge under costs) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(edge under costs at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")

    print("  Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med_real = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["lookback_medians_real"][str(lb)][0], p[str(lb)]["lookback_verdicts_real"][str(lb)])
            for lb in LOOKBACKS
        )
        med_no_costs = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["no_costs_medians"][0], p[str(lb)]["no_costs_verdict"][0])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians_real={:<40s} medians_no_costs={:<40s} rob_real={:<8s} "
              "edge_counts_real={}".format(
            ticker, str(p["5"]["segments"]), med_real, med_no_costs, p["robustness_verdict_real"],
            p["edge_counts_real"]))
    print("")

    # Determinism: rerun the target subset with realistic costs
    print("\n=== Determinism check (realistic costs) ===")
    target = ["AMZN", "JPM"]
    res2_real = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COSTS)
                      for t in target}
                 for lb in LOOKBACKS}
    out1_real = json.dumps({lb: {t: lookback_results_real[lb][t] for t in target}
                             for lb in LOOKBACKS}, sort_keys=True)
    out2_real = json.dumps(res2_real, sort_keys=True)
    print("  deterministic: r1 == r2 (realistic costs): {}".format(out1_real == out2_real))
    assert out1_real == out2_real, "non-deterministic output with realistic costs"

    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs=dict(
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians_real={str(lb): per_asset[t][str(lb)]["lookback_medians_real"][str(lb)]
                                          for lb in LOOKBACKS},
                    lookback_null_medians_real={str(lb): per_asset[t][str(lb)]["lookback_null_medians_real"][str(lb)]
                                               for lb in LOOKBACKS},
                    lookback_verdicts_real={str(lb): per_asset[t][str(lb)]["lookback_verdicts_real"][str(lb)]
                                           for lb in LOOKBACKS},
                    no_costs_medians={str(lb): per_asset[t][str(lb)]["no_costs_medians"]
                                      for lb in LOOKBACKS},
                    no_costs_nulls={str(lb): per_asset[t][str(lb)]["no_costs_nulls"]
                                    for lb in LOOKBACKS},
                    no_costs_verdicts={str(lb): per_asset[t][str(lb)]["no_costs_verdict"]
                                      for lb in LOOKBACKS},
                    robustness_verdict_real=per_asset[t]["robustness_verdict_real"],
                    edge_counts_real=per_asset[t]["edge_counts_real"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with realistic costs: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
