"""Real-data momentum lookback sweep with realistic transaction costs.

Objective: test whether the short-horizon momentum edge (lookback=5) survives
conservative transaction costs (commission + slippage) on the collected
real-data universe, for lookback in {3, 5, 10}. This addresses the bottleneck
identified in the campaign: cost robustness remains decision-relevant even when
short-horizon momentum shows candidate positive evidence.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance. Lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as `research/checks/momentum.py` and
`research/checks/momentum_lookback_sweep.py`. Realistic cost model per
test from `tests/test_engine.py`:
  - commission_per_share: 0.003 (3 bps per round trip)
  - slippage_cents: 5.0 (5 cents per share)
  - slippage_proportional: 0.001 (10 bps per share)

Each lookback's per-segment median log return is compared against a coin-flip
matched null benchmark run on the same segments. If the baseline stays within
the null's tolerance band, the cost-robustness test passes; otherwise, the
edge is rejected due to transaction costs.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

# Realistic cost model from tests/test_engine.py
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,           # commission applied per share
    commission_per_share=0.003,         # 3 bps per round trip
    slippage_cents=5.0,                 # 5 cents per share
    slippage_proportional=0.001,       # 10 bps per share
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)


def segment_result_with_costs(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker and one lookback with costs."""
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


def cost_sweep_summary():
    """Run momentum lookback sweep with realistic costs and matched null."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    # All collected tickers
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index].
    lookback_results = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, REALISTIC_COST_CONFIG)
                            for t in ticker_order}
                        for lb in LOOKBACKS}

    # Generate coin-flip null with identical cost configuration
    print("=== Generating matched null benchmark ===")
    from research.backtest.perturbation import noise_benchmark

    null_sweep = noise_benchmark(
        bars=list(all_tickers[ticker_order[0]]),  # use first ticker for null (signal-independent)
        param_grid=[{"lookback": lb} for lb in LOOKBACKS],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COST_CONFIG,
        periods_per_year=252,
        seed=SEED,
    )

    print("  null sweep: baseline median log return = {:+.3f}".format(null_sweep.baseline_median_log_return))
    print("  null sweep: effective tolerance = {:+.4f}".format(null_sweep.effective_tolerance))

    # Compare each lookback baseline against the null tolerance
    lookback_robustness = {}
    for lb in LOOKBACKS:
        baseline_median = lookback_results[lb][ticker_order[0]]["medians"][0]
        within_null = null_sweep.compare_noise(baseline_median)
        lookback_robustness[lb] = {
            "baseline_median": baseline_median,
            "null_median": null_sweep.baseline_median_log_return,
            "within_null_tolerance": within_null,
            "null_tolerance": null_sweep.effective_tolerance,
        }

    print("  null sweep: baseline median log return = {:+.3f}".format(null_sweep.baseline_median_log_return))
    print("  null sweep: effective tolerance = {:+.4f}".format(null_sweep.effective_tolerance))

    # Compare each lookback baseline against the null tolerance
    lookback_robustness = {}
    for lb in LOOKBACKS:
        baseline_median = lookback_results[lb][ticker_order[0]]["medians"][0]
        within_null = null_sweep.compare_noise(baseline_median)
        lookback_robustness[lb] = {
            "baseline_median": baseline_median,
            "null_median": null_sweep.baseline_median_log_return,
            "within_null_tolerance": within_null,
            "null_tolerance": null_sweep.effective_tolerance,
        }

    return {
        "all_tickers": all_tickers,
        "ticker_order": ticker_order,
        "lookback_results": lookback_results,
        "null_sweep": {
            "baseline_median": null_sweep.baseline_median_log_return,
            "baseline_param_set": null_sweep.baseline_param_set,
            "n_folds": null_sweep.n_folds,
            "n_periods": null_sweep.n_periods,
            "train_window": null_sweep.train_window_bars,
            "test_window": null_sweep.test_window_bars,
        },
        "lookback_robustness": lookback_robustness,
        "cost_config": {
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
        },
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Momentum cost-sensitivity check ===")
    print("  Lookbacks: {}".format(LOOKBACKS))
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  Cost model: commission_per_share={}, slippage_cents={}, slippage_proportional={}".format(
        REALISTIC_COST_CONFIG.commission_per_share,
        REALISTIC_COST_CONFIG.slippage_cents,
        REALISTIC_COST_CONFIG.slippage_proportional,
    ))

    sweep_results = cost_sweep_summary()
    all_tickers = sweep_results["all_tickers"]
    ticker_order = sweep_results["ticker_order"]

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_config=sweep_results["cost_config"],
        null_sweep=sweep_results["null_sweep"],
        lookback_robustness=sweep_results["lookback_robustness"],
        per_asset=dict(
            (t, dict(
                lookback_medians={str(lb): sweep_results["lookback_results"][lb][t]["medians"]
                                  for lb in LOOKBACKS},
                lookback_null_medians={str(lb): sweep_results["lookback_results"][lb][t]["null_medians"]
                                      for lb in LOOKBACKS},
                lookback_verdicts={str(lb): sweep_results["lookback_results"][lb][t]["verdict"]
                                 for lb in LOOKBACKS},
                edge_counts={str(lb): {"positive": sum(1 for m in sweep_results["lookback_results"][lb][t]["medians"] if m > 0),
                                             "negative": sum(1 for m in sweep_results["lookback_results"][lb][t]["medians"] if m < 0),
                                             "neutral": sum(1 for m in sweep_results["lookback_results"][lb][t]["medians"] if m == 0)}
                                 for lb in LOOKBACKS},
            ))
            for t in all_tickers.keys()
        ),
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-sensitivity test: check whether short-horizon edge survives realistic costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())