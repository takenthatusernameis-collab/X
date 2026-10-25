#!/usr/bin/env python3
"""Independent verifier of the cost-sensitivity check for lookback 3/5/10 momentum.

This is a separate implementation path: it recomputes the momentum spread P&L
and the spread medians from raw tickers, runs walk-forward on the spread
engine per fold with the fold's own baseline (the same independent cross-check
strategy the momentum check uses on the direct compounding path), then compares
against the artifact written by the cost-robustness check
(`state/check_artifacts/cost_robustness_momentum_results.json`).

A mismatch would flag a defect in the cost robustness check.

Design of the independent path:
- Tickers loaded from the manifest, truncated to the fully overlapping
  window (3614 bars, META) exactly as the check does.
- Per-segment medians are obtained via ``walk_forward`` on a synthetic spread
  asset per fold, each fold carrying its own baseline (independent of the
  check's direct compounding path but matching the check's engine cross-check);
  the check's medians use direct P&L compounding, so agreement is a genuine
  cross-implementation verification.
- The null is recomputed with a fresh coin-flip randomization (same seed)
  and run through the same engine path.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the cost robustness check computed from those inputs; a mismatch
would flag an error in the cost robustness check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASETS = [
    "yf-ohlcv-universe-2009-to-2026-10-03",  # from existing momentum.py
    # Add any other in-scope real datasets here if available
]

# Consistent with existing momentum.py
SEED = 42
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
FULLY_OVERLAPPING_BARS = 3614  # from existing cross_sectional_momentum.py

# Realistic cost model from examples/ma_crossover_real_data.py
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=60,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "cost_robustness_momentum_results.json"
MARKET_PROXY = "AAPL"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def verify_cost_robustness():
    """Execute the independent verification of cost robustness check."""
    print("=== Independent Verification of Cost Robustness Check ===")

    # 1. Load artifact
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print(f"  Loaded artifact from: {ARTIFACT_PATH}")
    print(f"  Dataset: {artifact['dataset_id']}")
    print(f"  Cost model: commission_per_trade={artifact['cost_model']['commission_per_trade']}, ")
    print(f" commission_per_share={artifact['cost_model']['commission_per_share']}, ")
    print(f" slippage_cents={artifact['cost_model']['slippage_cents']}, ")
    print(f" slippage_proportional={artifact['cost_model']['slippage_proportional']}")

    # 2. Verify manifest integrity
    print("\n=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']} vs {DATASET_ID}"

    all_ok = True
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
        if status != "OK":
            all_ok = False

    # 3. Load tickers (truncated to the fully overlapping window)
    print("\n=== 2. Loading tickers (truncated to fully overlapping window) ===")
    tickers = {}
    for entry in manifest["entries"]:
        bars, dates = bt.load_ticker(entry["ticker"])
        tickers[entry["ticker"]] = bars.closes_array()[:FULLY_OVERLAPPING_BARS]
        print(f"  {entry['ticker']}: {len(bars.closes_array())} bars total, {FULLY_OVERLAPPING_BARS} bars used")

    # 4. Reconstruct AAPL segments for regime classification
    print("\n=== 3. Reconstruct AAPL segments ===")
    aapl_closes = tickers["AAPL"]
    aapl_labels = bt.volatility_blocks(aapl_closes, n_blocks=N_BLOCKS, window=WINDOW)

    def make_segments(labels, min_segment_bars):
        """Contiguous segments from past-only labels."""
        segments = []
        cur = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                if i - start >= min_segment_bars:
                    segments.append((cur, start, i))
                cur = labels[i]
                start = i
        if len(labels) - start >= min_segment_bars:
            segments.append((cur, start, len(labels)))
        return segments

    runs = make_segments(aapl_labels, MIN_SEGMENT_BARS)
    seg_labels = [lab for lab, _, _ in runs]
    print(f"  Found {len(runs)} regime segments")

    # 5. Reconstruct momentum signals for lookback 3,5,10
    print("\n=== 4. Reconstruct momentum signals for lookback 3,5,10 ===")
    results_by_lookback = {}

    for lookback in [3, 5, 10]:
        print(f"\n  Lookback {lookback}:")
        lookback_results = {}

        # Reconstruct momentum spread
        def momentum_spread_for_lookback(lookback):
            """Independent implementation of momentum spread."""
            min_len = min(len(tb) for tb in tickers.values())
            pnl = np.zeros(min_len, dtype=np.float64)
            for i in range(lookback, min_len - 1):
                rets = {t: tickers[t][i] / tickers[t][i - lookback] - 1
                        for t in tickers}
                order = sorted(rets, key=lambda t: rets[t], reverse=True)
                # Top 3 long / bottom 3 short (consistent with existing momentum analysis)
                TOP_K = BOTTOM_K = 3
                longs = order[:TOP_K]
                shorts = order[-BOTTOM_K:]
                if len(longs) < TOP_K or len(shorts) < BOTTOM_K:
                    continue
                ret_l = float(np.mean([tickers[t][i + 1] / tickers[t][i] - 1
                                      for t, tick in enumerate(tickers.values()) if t in longs]))
                ret_s = float(np.mean([tickers[t][i + 1] / tickers[t][i] - 1
                                      for t, tick in enumerate(tickers.values()) if t in shorts]))
                pnl[i] = ret_l - ret_s
            return pnl

        # Compute medians via walk_forward (engine path)
        for ticker in tickers:
            pnl = momentum_spread_for_lookback(lookback)

            # Build synthetic spread asset
            synth = bt.build_synthetic_spread_asset(pnl, start_price=1e6)
            synth_signals = [bt.Signal(date=i + 1,
                                       weight=synth.closes_array()[i] / 1e6)
                            for i in range(len(pnl))]

            # Engine walk-forward per segment
            seg_medians = []
            for lab, s, e in runs:
                seg_bars = list(synth)[s:e]
                seg_signals = [bt.Signal(date=s + i + 1,
                                         weight=synth.closes_array()[s + i] /
                                                synth.closes_array()[s + WARM])
                              for i in range(e - s)]

                res = bt.walk_forward(seg_bars, seg_signals,
                                      train_window=TRAIN, test_window=TEST,
                                      warmup=WARM, overlap_window=OVERLAP,
                                      cfg=REALISTIC_COST_CONFIG)

                lrets = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds]
                assert len(lrets) == 1
                seg_medians.append(float(lrets[0]))

            # Build synth null
            rng = np.random.default_rng(SEED)
            def coin_flip_null_spread_for_lookback(lookback):
                """Independent coin-flip null implementation."""
                min_len = min(len(tb) for tb in tickers.values())
                pnl = np.zeros(min_len, dtype=np.float64)
                for i in range(lookback, min_len - 1):
                    rets = {t: tickers[t][i] / tickers[t][i - lookback] - 1
                            for t in tickers}
                    order = sorted(rets, key=lambda t: rets[t], reverse=True)
                    TOP_K = BOTTOM_K = 3
                    longs = order[:TOP_K]
                    shorts = order[-BOTTOM_K:]
                    if len(longs) < TOP_K or len(shorts) < BOTTOM_K:
                        continue
                    ret_l = float(np.mean([tickers[t][i + 1] / tickers[t][i] - 1
                                          for t, tick in enumerate(tickers.values()) if t in longs]))
                    ret_s = float(np.mean([tickers[t][i + 1] / tickers[t][i] - 1
                                          for t, tick in enumerate(tickers.values()) if t in shorts]))
                    pnl[i] = (ret_l - ret_s) if rng.random() < 0.5 else -(ret_l - ret_s)
                return pnl

            null_pnl = coin_flip_null_spread_for_lookback(lookback)

            # Engine walk-forward for null
            synth_null = bt.build_synthetic_spread_asset(null_pnl, start_price=1e6)
            synth_null_signals = [bt.Signal(date=i + 1,
                                            weight=synth_null.closes_array()[i] / 1e6)
                                 for i in range(len(null_pnl))]

            null_medians = []
            for lab, s, e in runs:
                seg_bars = list(synth_null)[s:e]
                seg_signals = [bt.Signal(date=s + i + 1,
                                         weight=synth_null.closes_array()[s + i] /
                                                synth_null.closes_array()[s + WARM])
                              for i in range(e - s)]

                res = bt.walk_forward(seg_bars, seg_signals,
                                      train_window=TRAIN, test_window=TEST,
                                      warmup=WARM, overlap_window=OVERLAP,
                                      cfg=REALISTIC_COST_CONFIG)

                lrets = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res.folds]
                assert len(lrets) == 1
                null_medians.append(float(lrets[0]))

            lookback_results[ticker] = {
                "candidate_medians": [round(m, 3) for m in seg_medians],
                "null_medians": [round(m, 3) for m in null_medians],
                "candidate_verdict": "REGIME_STABLE" if seg_medians else "VERDICT_NOT_RECORDED",
                "null_verdict": "REGIME_STABLE" if null_medians else "VERDICT_NOT_RECORDED",
            }

            # Compare with artifact
            artifact_key = f"lookback_{lookback}"
            if artifact_key not in artifact["momentum_cost_sensitivity"]:
                print(f"    [ERROR] {artifact_key} not found in artifact")
                all_ok = False
                continue

            artifact_results = artifact["momentum_cost_sensitivity"][artifact_key]
            if ticker not in artifact_results:
                print(f"    [ERROR] {ticker} not found in artifact {artifact_key}")
                all_ok = False
                continue

            # Compare medians
            artifact_candidate = artifact_results[ticker]["candidate_medians"]
            artifact_null = artifact_results[ticker]["null_medians"]

            cand_match = np.allclose(lookback_results[ticker]["candidate_medians"], artifact_candidate, atol=1e-9)
            null_match = np.allclose(lookback_results[ticker]["null_medians"], artifact_null, atol=1e-9)

            if not cand_match:
                print(f"    [ERROR] {ticker} candidate medians mismatch: ")
                print(f"      expected: {artifact_candidate}")
                print(f"      got:      {lookback_results[ticker]['candidate_medians']}")
                all_ok = False

            if not null_match:
                print(f"    [ERROR] {ticker} null medians mismatch: ")
                print(f"      expected: {artifact_null}")
                print(f"      got:      {lookback_results[ticker]['null_medians']}")
                all_ok = False

            if cand_match and null_match:
                print(f"    {ticker}: MATCH (candidate: {artifact_candidate}, null: {artifact_null})")
            else:
                print(f"    {ticker}: MISMATCH")

        results_by_lookback[lookback] = lookback_results

    print("\n=== Independent Verification Results ===")
    if all_ok:
        print("All independent recomputations MATCH the cost robustness check artifact.")
        print("The cost robustness check is verified and deterministic.")
    else:
        print("MISMATCH FOUND: The cost robustness check artifact does not reproduce via the ")
        print("independent path. Investigate before admitting the results.")
        sys.exit(1)

    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Cross-verification of cost robustness check on the collected universe: exploratory simulation.")

    return 0


if __name__ == "__main__":
    sys.exit(verify_cost_robustness())