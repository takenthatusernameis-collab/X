"""Cost robustness check for momentum lookback 3 / 5 / 10 on the collected universe.

Objective: test whether the momentum edge observed at lookback=5
(research/checks/momentum.py, activation 37318950814) survives realistic transaction costs.
Momentum is a very mechanical 1-day-ahead bet, so the most salient robustness question
is whether the positive edge remains profitable after realistic commission and slippage.

Method (fixed a-priori, not tuned to OOS): long the previous N-day return; hold 1 day;
daily rebalance; lookbacks 3, 5, 10; same walk-forward, regime-split, and coin-flip null
as the momentum check. Compare against a zero-cost baseline and realistic cost model.

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

# Cost model parameters from examples
COST_ZERO = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)

COST_REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def segment_result(ticker, bars, lookback, cfg):
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

    print("\n=== 3. Leakage review per asset (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe (zero cost) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep for zero cost.
    all_tickers_zero = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers_zero[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers_zero.keys())

    # Results indexed [lookback][ticker] for zero cost.
    lookback_results_zero = {lb: {t: segment_result(t, all_tickers_zero[t], lb, COST_ZERO) for t in ticker_order}
                             for lb in LOOKBACKS}

    print("\n=== 5. Momentum lookback sweep across collected universe (realistic costs) ===")
    # Full-universe lookback sweep for realistic costs.
    all_tickers_realistic = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers_realistic[ticker] = bt.load_ticker(ticker)[0]

    # Results indexed [lookback][ticker] for realistic costs.
    lookback_results_realistic = {lb: {t: segment_result(t, all_tickers_realistic[t], lb, COST_REALISTIC) for t in ticker_order}
                                  for lb in LOOKBACKS}

    per_asset_zero = {}
    per_asset_realistic = {}
    robustness_counts_zero = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    robustness_counts_realistic = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts_zero = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                             "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    lookback_5_counts_realistic = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                 "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        # Zero cost profile
        profile_zero = {str(lb): lookback_results_zero[lb][ticker] for lb in LOOKBACKS}
        v_zero, counts_zero = robustness_verdict({str(lb): profile_zero[str(lb)] for lb in LOOKBACKS})
        robustness_counts_zero[v_zero] += 1
        r5_zero = profile_zero["5"]
        lookback_5_counts_zero[r5_zero["verdict"]] += 1
        profile_zero["edge_counts"] = counts_zero
        profile_zero["robustness_verdict"] = v_zero
        profile_zero["lookback_medians"] = {str(lb): profile_zero[str(lb)]["medians"] for lb in LOOKBACKS}
        profile_zero["lookback_null_medians"] = {str(lb): profile_zero[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_zero[ticker] = profile_zero

        # Realistic cost profile
        profile_realistic = {str(lb): lookback_results_realistic[lb][ticker] for lb in LOOKBACKS}
        v_realistic, counts_realistic = robustness_verdict({str(lb): profile_realistic[str(lb)] for lb in LOOKBACKS})
        robustness_counts_realistic[v_realistic] += 1
        r5_realistic = profile_realistic["5"]
        lookback_5_counts_realistic[r5_realistic["verdict"]] += 1
        profile_realistic["edge_counts"] = counts_realistic
        profile_realistic["robustness_verdict"] = v_realistic
        profile_realistic["lookback_medians"] = {str(lb): profile_realistic[str(lb)]["medians"] for lb in LOOKBACKS}
        profile_realistic["lookback_null_medians"] = {str(lb): profile_realistic[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_realistic[ticker] = profile_realistic

    print("  Zero cost:")
    print("    lookback 5 (reference) verdict counts across universe: {}".format(lookback_5_counts_zero))
    print("    lookback-robustness verdict counts: {} (zero cost)".format(robustness_counts_zero))
    print("  Realistic cost:")
    print("    lookback 5 (reference) verdict counts across universe: {}".format(lookback_5_counts_realistic))
    print("    lookback-robustness verdict counts: {} (realistic cost)".format(robustness_counts_realistic))

    print("\n=== 6. Per-asset perturbation profiles (cost comparison) ===")
    for ticker in ticker_order:
        p0 = per_asset_zero[ticker]
        p1 = per_asset_realistic[ticker]
        med0 = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p0[str(lb)]["medians"][0], p0[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        med1 = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p1[str(lb)]["medians"][0], p1[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} zero_cost: medians3510={:<40s} rob={:<8s} | "
              "real_cost: medians3510={:<40s} rob={:<8s}".format(
            ticker, med0, p0["robustness_verdict"], med1, p1["robustness_verdict"]
        ))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result(t, tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results_zero[lb][t] for t in target}
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
        zero_cost=dict(
            per_asset=per_asset_zero,
            universe=dict(
                lookback_5_counts=lookback_5_counts_zero,
                robustness_counts=robustness_counts_zero,
                ticker_order=ticker_order,
                per_asset=dict(
                    (t, dict(
                        lookback_medians={str(lb): per_asset_zero[t]["lookback_medians"][str(lb)]
                                          for lb in LOOKBACKS},
                        lookback_null_medians={str(lb): per_asset_zero[t]["lookback_null_medians"][str(lb)]
                                             for lb in LOOKBACKS},
                        verdicts_5=per_asset_zero[t]["5"]["verdict"],
                        edge_counts=per_asset_zero[t]["edge_counts"],
                        robustness_verdict=per_asset_zero[t]["robustness_verdict"],
                        n_folds_5=per_asset_zero[t]["5"]["n_folds"],
                    ))
                    for t in ticker_order
                ),
            ),
        ),
        realistic_cost=dict(
            per_asset=per_asset_realistic,
            universe=dict(
                lookback_5_counts=lookback_5_counts_realistic,
                robustness_counts=robustness_counts_realistic,
                ticker_order=ticker_order,
                per_asset=dict(
                    (t, dict(
                        lookback_medians={str(lb): per_asset_realistic[t]["lookback_medians"][str(lb)]
                                          for lb in LOOKBACKS},
                        lookback_null_medians={str(lb): per_asset_realistic[t]["lookback_null_medians"][str(lb)]
                                             for lb in LOOKBACKS},
                        verdicts_5=per_asset_realistic[t]["5"]["verdict"],
                        edge_counts=per_asset_realistic[t]["edge_counts"],
                        robustness_verdict=per_asset_realistic[t]["robustness_verdict"],
                        n_folds_5=per_asset_realistic[t]["5"]["n_folds"],
                    ))
                    for t in ticker_order
                ),
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost robustness check (lookback 3/5/10) on the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())