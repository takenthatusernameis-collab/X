"""Cost-robustness momentum check for lookback 3/5/10 across the collected universe.

Objective: test whether the short-horizon momentum edge observed at lookback=5
(strong evidence in some assets) survives a realistic transaction-cost stress test
using the repository's BacktestConfig cost model, compared to a matched null (coin-flip).

Method (fixed a-priori): long the previous N-day return; hold 1 day; daily rebalance;
lookbacks 3, 5, 10; same walk-forward per segment train=252d/test=84d/warmup=60d/
overlap=60d, min 400 bars/segment, 4 contiguous volatility blocks; each lookback's
segment medians are compared against a coin-flip null benchmark on the same segments;
the BacktestConfig includes commission_per_trade, commission_per_share, slippage_cents,
and slippage_proportional (the realistic cost model from the repository's existing
framework). Determinism is asserted via an internal independent recomputation of the
AMZN/JPM subset.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_cost_results.json"

# The repository's existing realistic cost model: BacktestConfig with typical values.
# Based on existing engine.py defaults and tests, these values represent realistic
# trading costs while preserving the research/simulation nature of this work.
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,        # $0 per round-trip transaction
    commission_per_share=0.003,      # $0.003 per share
    slippage_cents=0.5,              # $0.50 fixed slippage per share
    slippage_proportional=0.0001,    # 10 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

# Matched null: coin-flip signals matching the same cost model and segment structure.
from research.backtest.perturbation import noise_benchmark, parameter_grid_around
def segment_cost_robustness(ticker, bars, lookback, cfg):
    """Walk-forward cost-robustness result for one ticker and one lookback with costs."""
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
        cfg=cfg,
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
        cost_config={
            "commission_per_trade": cfg.commission_per_trade,
            "commission_per_share": cfg.commission_per_share,
            "slippage_cents": cfg.slippage_cents,
            "slippage_proportional": cfg.slippage_proportional,
        },
    )
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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost-robustness momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"{N_BLOCKS} contiguous volatility blocks, each lookback vs coin-flip null "
          f"on the same segments, with realistic costs:")
    print(f"    commission_per_trade: ${REALISTIC_COSTS.commission_per_trade:.3f}")
    print(f"    commission_per_share: ${REALISTIC_COSTS.commission_per_share:.3f}")
    print(f"    slippage_cents: ${REALISTIC_COSTS.slippage_cents:.3f} per share")
    print(f"    slippage_proportional: {REALISTIC_COSTS.slippage_proportional:.6f} ({REALISTIC_COSTS.slippage_proportional*100:.2f} bps)")

    # Full-universe cost-robustness lookback sweep.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index]. Cost test for each lookback.
    lookback_results = {lb: {t: segment_cost_robustness(t, all_tickers[t], lb, REALISTIC_COSTS)
                        for t in ticker_order}
                       for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        # Reuse verdict logic from momentum_lookback_sweep.py
        def lookback_verdict(sr):
            if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
                return "EDGE"
            if sr["verdict"] == "REGIME_STABLE_LOSS":
                return "LOSS"
            return "NO_EDGE"

        def robustness_verdict(lookback_results):
            counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            for lb in LOOKBACKS:
                counts[lookback_verdict(lookback_results[str(lb)])] += 1
            if counts["EDGE"] >= 2:
                return "ROBUST", counts
            if counts["EDGE"] == 1:
                return "SENSITIVE", counts
            return "NO_EDGE", counts

        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts[v] += 1
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        # Lookback-robustness edge counts (edges per asset, for the summary).
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        # Store the per-lookback medians + nulls for the artifact's perturbation view.
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: ")
    print("    " + ", ".join(f"{k}: {v}" for k, v in lookback_5_counts.items()))
    print("  lookback-robustness verdict counts: ")
    print(f"    ROBUST={robustness_counts['ROBUST']} (>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} (edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")
    print("  Per-asset perturbation profiles (with realistic costs):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    target = ["AMZN", "JPM"]
    res2 = {lb: {t: segment_cost_robustness(t, all_tickers[t], lb, REALISTIC_COSTS)
                   for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Matched null benchmark for the realistic cost model
    print("\n=== 4. Matched null (coin-flip) under realistic cost model ===")
    # Create a noise benchmark with the same cost structure and segment parameters
    noise_bars = bt.generate_bars(MIN_SEGMENT_BARS * N_BLOCKS, seed=SEED)
    null_grid = parameter_grid_around((("lookback", LOOKBACKS[1]),), multipliers=(0.5, 1.0, 2.0))
    null_result = bt.parameter_sweep(
        signals_fn=lambda closes, **params: bt.random_signals(len(closes), seed=42),
        bars=noise_bars,
        param_grid=null_grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
    )
    null_summary = bt.sweep_summary(null_result, (("lookback", LOOKBACKS[1]),), noise_median=None)
    print("  Null benchmark under realistic costs:")
    print("    " + str(null_summary.inspect()).replace("\n", "\n    "))

    # Cost-robustness verdict
    print("\n=== 5. Cost-robustness summary ===")
    print(f"  ROBUST: {robustness_counts['ROBUST']} assets have lookback-robust positive edges under realistic costs")
    print(f"  SENSITIVE: {robustness_counts['SENSITIVE']} assets have edges at only one lookback under costs")
    print(f"  NO_EDGE: {robustness_counts['NO_EDGE']} assets have no lookback-robust edges under costs")

    print(f"\n  lookback 5 cost-verdict counts across universe: {lookback_5_counts}")
    print("  Interpretation: assets with REGIME_STABLE and uniformly positive medians under realistic costs")
    print("  show a cost-robust edge; CONSISTENT_WITH_NOISE or REGIME_STABLE_LOSS indicate the edge")
    print("  is not robust to transaction costs.")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
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
                    cost_config=per_asset[t]["5"]["cost_config"],
                ))
                for t in ticker_order
            ),
        ),
        realistic_cost_model={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        null_benchmark=dict(
            param_sets=[tuple(ps) for ps in null_summary.param_sets],
            baseline_median=null_summary.baseline_median_log_return,
            noise_median=null_summary.noise_median_log_return,
            compare_noise=null_summary.compare_noise(null_summary.noise_median_log_return) if null_summary.noise_median_log_return is not None else None,
            effective_tolerance=null_summary.effective_tolerance,
            n_folds=null_summary.n_folds,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Cost-robustness momentum check across the collected universe.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
