"""Test the cost robustness of momentum on the collected universe.

Background: the momentum edge (lookback 5) was admitted as candidate positive
evidence in activation 37318950814. The evidence is short-horizon and mechanical;
cost robustness remains the key qualification. This check runs the bounded
cost-sensitivity test for lookback 3/5/10 momentum under the repository's
existing realistic cost model and a matched coin-flip null.

Falsification prediction: momentum either shows CONSISTENT_WITH_NOISE under
costs (no edge anywhere), REGIME_DEPENDENT (edge concentrates in one historical
regime mix), or REGIME_STABLE_LOSS (uniformly negative under costs). Either
outcome is durable negative evidence and closes the qualification gap.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return,
hold 1 day, daily rebalance. Test lookback 3, 5, 10 with realistic costs:
commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
slippage_proportional=0.0005 (same as in examples/ma_crossover_real_data.py).
Walk-forward per segment train=252d/test=84d/warmup=60d/overlap=60d, min 400
bars/segment, compared vs coin-flip null on the same segments.
Research/simulation only. No live trading or production execution.
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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"



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

    print("\n=== 3. Realistic cost model (same as examples/ma_crossover_real_data.py) ===")
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )
    print("  commission_per_trade=2.0 | commission_per_share=0.003")
    print("  slippage_cents=2.0 | slippage_proportional=0.0005")

    print("\n=== 4. Momentum cost robustness: lookback 3 / 5 / 10 across universe ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]

    print("  walk-forward per segment train={}d / test={}d / warmup={}d / overlap={}d"
          " min {} bars/segment, compared vs coin-flip null"
          "\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  segments: {} contiguous blocks by date, past-only volatility classifier"
          " (turbulent = block median trailing-{}-bar realized vol > series-wide median)\n"
          "".format(N_BLOCKS, WINDOW))

    results_all = {}
    for lb in LOOKBACKS:
        summary = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lb),),
            param_grid=[{"lookback": lb}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg,
        )
        results_all[lb] = {
            "verdict_counts": summary.verdict_counts,
            "ticker_order": summary.ticker_order,
            "assets": {
                ticker: {
                    "segments": [s.name for s in asset.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in asset.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in asset.scenarios],
                    "candidate_dispersion": round(asset.candidate_dispersion, 3),
                    "null_dispersion": round(asset.null_dispersion, 3),
                    "verdict": asset.overall_verdict,
                }
                for ticker, asset in summary.assets.items()
            }
        }
        print(f"  lookback={lb:2d}: {summary.inspect()}")
        print()

    # Determinism: rerun universe-level with the same cost config and assert identical output
    results2 = {}
    for lb in LOOKBACKS:
        summary2 = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lb),),
            param_grid=[{"lookback": lb}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg,
        )
        results2[lb] = {
            "verdict_counts": summary2.verdict_counts,
            "ticker_order": summary2.ticker_order,
            "assets": {
                ticker: {
                    "segments": [s.name for s in asset.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in asset.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in asset.scenarios],
                    "candidate_dispersion": round(asset.candidate_dispersion, 3),
                    "null_dispersion": round(asset.null_dispersion, 3),
                    "verdict": asset.overall_verdict,
                }
                for ticker, asset in summary2.assets.items()
            }
        }
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(results2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output with costs"

    # Write artifact that independent verification recomputes
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=LOOKBACKS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_cfg={
            "commission_per_trade": cfg.commission_per_trade,
            "commission_per_share": cfg.commission_per_share,
            "slippage_cents": cfg.slippage_cents,
            "slippage_proportional": cfg.slippage_proportional,
        },
        per_lookback=dict(
            (lb, dict(
                verdict_counts=res["verdict_counts"],
                ticker_order=res["ticker_order"],
                per_asset=dict(
                    (t, dict(
                        segments=res["assets"][t]["segments"],
                        medians=res["assets"][t]["medians"],
                        null_medians=res["assets"][t]["null_medians"],
                        candidate_dispersion=res["assets"][t]["candidate_dispersion"],
                        null_dispersion=res["assets"][t]["null_dispersion"],
                        verdict=res["assets"][t]["verdict"],
                    ))
                    for t in res["assets"]
                ),
            ))
            for lb, res in results_all.items()
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost robustness on the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
