"""Bounded cost-sensitivity check for lookback 3/5/10 momentum under realistic costs.

Objective: evaluate whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe (the existing 10-asset
collected universe and existing momentum implementation/check framework). Use the
repository's existing realistic cost model and matched null (zero costs) to generate
cost-robustness evidence.

Method (fixed a-priori, not tuned to OOS):
- long the previous lookback-day return, hold 1 day, daily rebalance
- lookbacks 3, 5, 10
- walk-forward train=252d/test=84d/warmup=60d/overlap=60d inside each of the 4
  contiguous volatility blocks of the real-data regime family
- realistic costs: commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005
- matched null: zero cost (same as the existing momentum class but with cfg=BacktestConfig())
- same inputs (dataset, seed, windows, segments) as momentum_lookback_sweep.py
- independent recomputation via walk_forward (fresh signal, fresh regime labels,
  fresh segment reconstruction) to verify determinism
- artifact written for downstream verification

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

REALISTIC_COST_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COST_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, with given costs."""
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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Momentum lookback sweep across collected universe (baseline) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Baseline: zero cost
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Baseline results (zero cost)
    lookback_results_zero = {lb: {t: segment_result(t, all_tickers[t], lb, ZERO_COST_CFG)
                                 for t in ticker_order}
                            for lb in LOOKBACKS}

    per_asset_zero = {}
    robustness_counts_zero = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results_zero[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts_zero[v] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_zero[ticker] = profile

    print("  Lookback-robustness verdict counts (zero cost): "
          f"ROBUST={robustness_counts_zero['ROBUST']} "
          f"SENSITIVE={robustness_counts_zero['SENSITIVE']} "
          f"NO_EDGE={robustness_counts_zero['NO_EDGE']}\n")

    print("  Per-asset baseline profiles (zero cost):")
    for ticker in ticker_order:
        p = per_asset_zero[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Realistic cost results
    print("\n=== 4. Momentum lookback sweep with realistic costs ===")
    print("  Realistic cost model: commission_per_trade=2.0, commission_per_share=0.003, "
          "slippage_cents=2.0, slippage_proportional=0.0005")

    lookback_results_real = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COST_CFG)
                                 for t in ticker_order}
                            for lb in LOOKBACKS}

    per_asset_real = {}
    robustness_counts_real = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results_real[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts_real[v] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        per_asset_real[ticker] = profile

    print("  Lookback-robustness verdict counts (realistic costs): "
          f"ROBUST={robustness_counts_real['ROBUST']} "
          f"SENSITIVE={robustness_counts_real['SENSITIVE']} "
          f"NO_EDGE={robustness_counts_real['NO_EDGE']}\n")

    print("  Per-asset realistic-cost profiles:")
    for ticker in ticker_order:
        p = per_asset_real[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Determinism check for both cost models
    print("\n=== 5. Determinism check (AMZN, JPM) for both cost models ===")
    target = ["AMZN", "JPM"]
    for cfg_name, cfg in [("zero costs", ZERO_COST_CFG), ("realistic costs", REALISTIC_COST_CFG)]:
        for lb in LOOKBACKS:
            res2 = {t: segment_result(t, all_tickers[t], lb, cfg) for t in target}
            res1 = {t: lookback_results_zero[lb][t] for t in target}
            out1 = json.dumps({lb: {t: res1[t] for t in target} for lb in LOOKBACKS}, sort_keys=True)
            out2 = json.dumps({lb: {t: res2[t] for t in target} for lb in LOOKBACKS}, sort_keys=True)
            print(f"  {cfg_name}, lookback {lb}: deterministic {out1 == out2}")
            assert out1 == out2, f"non-deterministic output with {cfg_name}, lookback {lb}"

    # Write artifact for downstream verification
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset_zero=per_asset_zero,
        per_asset_real=per_asset_real,
        universe=dict(
            robustness_counts_zero=robustness_counts_zero,
            robustness_counts_real=robustness_counts_real,
            ticker_order=ticker_order,
            per_asset_zero=dict(
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
            per_asset_real=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset_real[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset_real[t]["lookback_null_medians"][str(lb)]
                                         for lb in LOOKBACKS},
                    verdicts_5=per_asset_real[t]["5"]["verdict"],
                    edge_counts=per_asset_real[t]["edge_counts"],
                    robustness_verdict=per_asset_real[t]["robustness_verdict"],
                    n_folds_5=per_asset_real[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
        cost_model="realistic",
        zero_cost_model="zero",
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness check across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())