"""Volatility gate on lookback-5 momentum through regime-stability and null framework.

Objective: Test one explicitly predeclared past-only volatility gate on lookback-5
momentum through the existing regime-stability and null framework. The gate
activates only when past-only realized volatility is below a threshold.

Method (fixed a-priori, not tuned to OOS):
- long the previous 5-day return; hold 1 day; daily rebalance
- walk-forward validation (train=252d / test=84d / warmup=60d / overlap=60d)
- 4 contiguous volatility blocks per asset (regime stability)
- past-only volatility gate: only trade when early-vol < 0.25 annualized
- matched null: coin-flip null benchmark using same parameters
- compare candidate vs null medians across segments; verdict via regime-stability

Evidence basis: MA regime filters were previously falsified; whether the same
class of conditioning helps the different momentum mechanism remains unresolved.
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
LOOKBACK = 5
VOL_THRESHOLD = 0.25  # annualized volatility threshold for the gate
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Volatility gate on lookback-5 momentum ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Gate: only trade when past-only volatility < {VOL_THRESHOLD} annualized")
    print(f"Parameters: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d")
    print(f"Volatility window: {WINDOW} bars for gate estimation")
    print()

    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"dataset id mismatch: {manifest['dataset_id']}"

    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    print("=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting volatility-gate run.")
        sys.exit(1)
    print()

    # Full-universe regime-stability with volatility gate
    print("=== 3. Volatility-gate momentum across collected universe ===")

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]

    baseline = (("lookback", LOOKBACK),)
    param_grid = [{"lookback": LOOKBACK}]

    print(f"  Testing momentum with volatility gate (threshold={VOL_THRESHOLD})")
    print(f"  Parameters: walk-forward {TRAIN}d train / {TEST}d test / {WARM}d warmup / {OVERLAP}d overlap")
    print(f"  {N_BLOCKS} volatility blocks per asset, min {MIN_SEGMENT_BARS} bars/segment")
    print()

    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals_with_volatility_gate(
            c, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD
        ),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=baseline,
        param_grid=param_grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=bt.BacktestConfig(),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )

    print("=== 4. Results summary ===")
    print(summary.inspect())

    # Determinism check: run the same analysis again and verify identical results
    print("\n=== 5. Determinism verification ===")
    tickers2 = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers2[ticker] = bt.load_ticker(ticker)[0]

    summary2 = bt.stress_segments_across_tickers(
        tickers=tickers2,
        signals_fn=lambda c, **kw: bt.momentum_signals_with_volatility_gate(
            c, lookback=LOOKBACK, vol_threshold=VOL_THRESHOLD
        ),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=baseline,
        param_grid=param_grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=bt.BacktestConfig(),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )

    results_identical = (
        summary.verdict_counts == summary2.verdict_counts and
        summary.candidate_dispersion == summary2.candidate_dispersion and
        summary.null_dispersion == summary2.null_dispersion
    )
    print(f"  Determinism check passed: {results_identical}")
    if not results_identical:
        print("  ERROR: Non-deterministic results!")
        sys.exit(1)

    # Write artifact for independent verification
    print("\n=== 6. Writing artifact ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        vol_threshold=VOL_THRESHOLD,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=dict(
            (t, dict(
                segments=[s.name for s in r.scenarios],
                medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                candidate_dispersion=round(r.candidate_dispersion, 3),
                null_dispersion=round(r.null_dispersion, 3),
                verdict=r.overall_verdict,
            ))
            for t, r in summary.assets.items()
        ),
        universe=dict(
            verdict_counts=summary.verdict_counts,
            ticker_order=summary.ticker_order,
            candidate_dispersion=summary.candidate_dispersion,
            null_dispersion=summary.null_dispersion,
        ),
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"  Artifact written to: {ARTIFACT_PATH}")
    print()
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum with volatility gate on lookback=5: exploratory simulation.")
    print()
    print("Interpretation:")
    print("- REGIME_STABLE = the volatility-gated momentum behaves the same way in every regime mix (same edge or no edge everywhere)")
    print("- REGIME_STABLE_LOSS = the volatility-gated momentum loses money in every regime mix (stable losses, not an edge)")
    print("- REGIME_DEPENDENT = results swing strongly across regimes relative to the coin-flip null (fits one regime, fails in others)")
    print("- CONSISTENT_WITH_NOISE = the volatility-gated momentum is indistinguishable from noise in every segment")

    return 0


if __name__ == "__main__":
    sys.exit(main())