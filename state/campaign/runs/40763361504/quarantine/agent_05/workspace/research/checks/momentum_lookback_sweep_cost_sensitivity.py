"""Real-data momentum lookback sweep with conservative transaction-cost stress.

Objective: test whether the momentum edge survives realistic transaction costs
(base and stress), using the same lookback robustness framework as the existing
momentum_lookback_sweep but with cost model parameters varied.

Scope: 10-asset collected universe, lookbacks 3/5/10, same walk-forward settings
as momentum_lookback_sweep. Cost model uses BacktestConfig parameters:
- commission_per_share: realistic and stress values
- slippage_cents: realistic and stress values  
- slippage_proportional: realistic and stress values

Method: For each lookback, run regime-stability stress testing with each
cost configuration, then compare against a matched coin-flip null benchmark
(run on the same cost configuration). The result is whether the momentum
edge survives cost stress.

Results recorded per lookback and per cost configuration.
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

# Cost configurations to test: (name, backtest_config)
COST_CONFIGS = [
    ("baseline_realistic", bt.BacktestConfig(
        commission_per_share=0.003,
        slippage_cents=5.0,
        slippage_proportional=0.001,
        warmup_periods=0,
    )),
    ("stress_conservative", bt.BacktestConfig(
        commission_per_share=0.01,
        slippage_cents=10.0,
        slippage_proportional=0.003,
        warmup_periods=0,
    )),
]
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_cost_sensitivity.json"


def segment_result(ticker, bars, lookback, cfg):
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
        cfg=cfg,
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
    """Robustness verdict for one lookback's scenario result (independent reimplementation)."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset (independent reimplementation)."""
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

    print("=== Cost-Sensitivity Check for Lookback 3 / 5 / 10 Momentum ===")
    print("Dataset: {} | {} entries | universe {}".format(
        manifest["dataset_id"], len(manifest["entries"]), len(manifest["universe"])))
    print("Walk-forward settings: train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, 4 contiguous volatility blocks".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("Cost configurations: {}".format([c[0] for c in COST_CONFIGS]))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed by cost_config_name -> lookback -> ticker -> segment results
    results = {}
    robustness_counts_by_cost = {}

    for cost_name, cfg in COST_CONFIGS:
        print("\n--- Processing cost configuration: {} ---".format(cost_name))
        print("  commission_per_share={}  slippage_cents={}  slippage_proportional={}".format(
            cfg.commission_per_share, cfg.slippage_cents, cfg.slippage_proportional))

        lookback_results = {lb: {t: segment_result(t, all_tickers[t], lb, cfg) for t in ticker_order}
                           for lb in LOOKBACKS}
        results[cost_name] = lookback_results

        # Robustness verdict counts (ROBUST/SENSITIVE/NO_EDGE)
        counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                             "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

        for ticker in ticker_order:
            profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
            v, counts_for_asset = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
            counts[v] += 1
            r5 = profile["5"]
            lookback_5_counts[r5["verdict"]] += 1

        robustness_counts_by_cost[cost_name] = {
            "lookback_5_verdict_counts": lookback_5_counts,
            "lookback_robustness_verdict_counts": counts,
            "cost_config": dict(
                commission_per_share=cfg.commission_per_share,
                slippage_cents=cfg.slippage_cents,
                slippage_proportional=cfg.slippage_proportional,
                warmup_periods=cfg.warmup_periods,
            ),
        }
        print("  lookback 5 verdict counts: {}".format(lookback_5_counts))
        print("  lookback-robustness verdict counts: {}".format(counts))

    # Determinism: verify that results are reproducible (run same calculation again)
    print("\n=== Determinism check ===")
    res2 = {}
    for cost_name, cfg in COST_CONFIGS:
        res2[cost_name] = {}
        for lb in LOOKBACKS:
            res2[cost_name][lb] = {}
            for ticker in ["AMZN", "JPM"]:
                res2[cost_name][lb][ticker] = segment_result(ticker, all_tickers[ticker], lb, cfg)

    out1 = json.dumps(results, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    if out1 != out2:
        print("ERROR: Non-deterministic output detected!")
        return 1

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_configurations=[{"name": c[0], "cfg": c[1].__dict__} for c in COST_CONFIGS],
        results=results,
        robustness_summary=robustness_counts_by_cost,
        determinism_check=out1 == out2,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. Transaction-cost sensitivity for momentum lookback sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
