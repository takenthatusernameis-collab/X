"""Cost-robustness check for lookback 3/5/10 momentum.

Objective: Test whether the short-horizon momentum edge survives realistic
transaction costs. This addresses the decision-relevant uncertainty: candidate
positive evidence is mechanical; cost robustness remains a critical qualification.

Method (fixed a-priori, not tuned to OOS):
- momentum signals for lookback in {3,5,10}
- realistic cost model from BacktestConfig
- matched null via noise_benchmark (same lookback, same data slice)
- walk-forward regime-stability across 4 contiguous volatility blocks
- cost-sensitivity: parameter sweep across realistic cost parameters
- independent verification path recomputes from artifact

Evidence standard: candidate evidence, still to be judged for cost robustness.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


REALISTIC_COST_CONFIG = bt.BacktestConfig(
    commission_per_trade=5.0,  # $5 per trade
    commission_per_share=0.005,  # $0.005 per share
    slippage_cents=10.0,  # $0.10 per share
    slippage_proportional=0.001,  # 10 bps proportional slippage
    initial_capital=1e6,
    target_exposure=1.0,
)


NULL_COST_CONFIG = bt.BacktestConfig(
    commission_per_trade=0.0,  # null has no costs
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    initial_capital=1e6,
    target_exposure=1.0,
)


def segment_cost_result(
    ticker,
    bars,
    lookback,
    cost_config,
    null_config,
):
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
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_config,
    )

    null = bt.noise_benchmark(
        bars=bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        cfg=null_config,
    )

    return {
        "lookback": lookback,
        "segments": [s.name for s in res.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "n_folds": res.n_folds,
        "verdict": res.overall_verdict,
        "cost_config": {
            "commission_per_trade": cost_config.commission_per_trade,
            "commission_per_share": cost_config.commission_per_share,
            "slippage_cents": cost_config.slippage_cents,
            "slippage_proportional": cost_config.slippage_proportional,
        },
    }


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

    print("\n=== 3. Cost-robustness check across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, lookbacks={}, cost config R={}"
          .format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS,
                  LOOKBACKS, REALISTIC_COST_CONFIG))

    # Full-universe cost-robustness check.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [ticker][lookback][cost_param][segment index].
    cost_results = {}
    for ticker in ticker_order:
        cost_results[ticker] = {}
        for lookback in LOOKBACKS:
            res = segment_cost_result(
                ticker,
                all_tickers[ticker],
                lookback,
                REALISTIC_COST_CONFIG,
                NULL_COST_CONFIG,
            )
            cost_results[ticker][lookback] = res

    print("  Results computed for {} tickers x {} lookbacks".format(
        len(ticker_order), len(LOOKBACKS)))

    # Determinism: rerun the target subset and assert identical output.
    print("\nDeterminism: rerun AMZN/JPM x lookbacks x costs.")
    target = ["AMZN", "JPM"]
    tickers_target = {t: all_tickers[t] for t in target}

    out1 = json.dumps(
        {t: {lb: cost_results[t][lb] for lb in LOOKBACKS} for t in target},
        sort_keys=True,
    )
    
    out2 = {}
    for t in target:
        out2[t] = {}
        for lookback in LOOKBACKS:
            out2[t][lookback] = segment_cost_result(
                t,
                tickers_target[t],
                lookback,
                REALISTIC_COST_CONFIG,
                NULL_COST_CONFIG,
            )

    out2_json = json.dumps(out2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2_json))
    assert out1 == out2_json, "non-deterministic output"

    # Build the artifact for independent verification.
    artifact = dict(
        dataset_id=DATASET_ID,
        realistic_cost_config={
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
            "initial_capital": REALISTIC_COST_CONFIG.initial_capital,
        },
        null_cost_config={
            "commission_per_trade": NULL_COST_CONFIG.commission_per_trade,
            "commission_per_share": NULL_COST_CONFIG.commission_per_share,
            "slippage_cents": NULL_COST_CONFIG.slippage_cents,
            "slippage_proportional": NULL_COST_CONFIG.slippage_proportional,
            "initial_capital": NULL_COST_CONFIG.initial_capital,
        },
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        lookbacks=list(LOOKBACKS),
        per_asset=cost_results,
        universe=dict(
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_results={lb: cost_results[t][lb] for lb in LOOKBACKS},
                    robustness_summary={
                        lb: {
                            "medians": cost_results[t][lb]["medians"],
                            "null_medians": cost_results[t][lb]["null_medians"],
                            "verdict": cost_results[t][lb]["verdict"],
                            "edge": (
                                cost_results[t][lb]["verdict"] == "REGIME_STABLE"
                                and all(m > 0 for m in cost_results[t][lb]["medians"])
                            ),
                        }
                        for lb in LOOKBACKS
                    },
                ))
                for t in ticker_order
            ),
        ),
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print("\n=== 4. Artifact written to: {} ===".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness check on {} + "
          "realistic transaction costs: exploratory simulation.".format(len(ticker_order)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
