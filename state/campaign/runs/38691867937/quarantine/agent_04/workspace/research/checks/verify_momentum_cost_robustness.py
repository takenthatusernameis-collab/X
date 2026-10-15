"""Independent verifier for momentum cost robustness test.

This verifier recomputes momentum with realistic costs from a separate path
and compares against the main test artifact.
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


def momentum_signals(closes: np.ndarray, lookback: int):
    """Past-only momentum signals: long the previous N-day return.

    Bars before the lookback window are neutral. The signal author guarantees
    padding through the lookback window so no real signal exists before the
    warmup boundary.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    if lookback <= 0:
        return signals
    for i in range(lookback - 1, n):
        returns = (closes[i - lookback + 1 : i + 1] - closes[i - lookback]) / closes[i - lookback]
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if np.mean(returns) > 0 else -1.0)
    return signals


def verify_lookback(lookback: int, artifact_path: Path):
    """Verify momentum with realistic costs for a specific lookback."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
        periods_per_year=252,
    )

    # Load target ticker subset for verification (AMZN, JPM)
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars

    # Run momentum with realistic costs for target tickers
    results = {}
    for ticker in target:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)

        signals = momentum_signals(closes, lookback)
        if len(signals) != len(closes):
            raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")

        res = bt.stress_segments(
            signals_fn=lambda c, **kw: momentum_signals(c, lookback),
            bars=bars,
            signals=signals,
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

        results[ticker] = {
            "segments": [s.name for s in res.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
            "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
            "candidate_dispersion": round(res.candidate_dispersion, 3),
            "null_dispersion": round(res.null_dispersion, 3),
            "n_folds": res.n_folds,
            "verdict": res.overall_verdict,
        }

    # Load the artifact
    with open(artifact_path) as f:
        artifact = json.load(f)

    # Compare for target tickers
    artifact_results = artifact["realistic_cost"][lookback]

    for ticker in target:
        art_result = artifact_results[ticker]
        ver_result = results[ticker]

        # Compare key metrics
        comparisons = []
        for key in ["segments", "medians", "null_medians", "candidate_dispersion", "null_dispersion", "n_folds", "verdict"]:
            art_val = art_result[key]
            ver_val = ver_result[key]

            if key in ["segments", "verdict"]:
                match = art_val == ver_val
                comparisons.append((key, art_val, ver_val, match, "string equality"))
            else:
                # For numeric values, allow small floating-point differences
                if isinstance(art_val, list) and isinstance(ver_val, list):
                    if len(art_val) != len(ver_val):
                        comparisons.append((key, art_val, ver_val, False, "list length mismatch"))
                    else:
                        match = all(abs(a - v) < 0.001 for a, v in zip(art_val, ver_val))
                        comparisons.append((key, art_val, ver_val, match, "numeric tolerance"))
                else:
                    match = abs(art_val - ver_val) < 0.001
                    comparisons.append((key, art_val, ver_val, match, "numeric tolerance"))

        # Report
        print(f"  {ticker} (lookback {lookback}):")
        all_match = all(c[3] for c in comparisons)
        print(f"    VERIFICATION: {"PASS" if all_match else "FAIL"}")
        if not all_match:
            for key, art_val, ver_val, match, reason in comparisons:
                if not match:
                    print(f"      {key}: artifact={art_val}, verifier={ver_val} ({reason})")
        else:
            print(f"      All metrics match (tolerance 0.001)")

    return all(c[3] for c in comparisons)


def main() -> int:
    ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

    print("=== Independent verifier for momentum cost robustness ===")
    print(f"Artifact path: {ARTIFACT_PATH}")
    print()

    # Check if artifact exists
    if not ARTIFACT_PATH.exists():
        print(f"ERROR: Artifact not found at {ARTIFACT_PATH}")
        print("Run momentum_cost_robustness.py first to generate the artifact.")
        return 1

    # Verify each lookback
    print("=== Verification gates ===")
    print("GATE 1: Manifest integrity and data preflight")

    # Manifest integrity
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        if status == "MISMATCH":
            print(f"  MISMATCH: {entry['ticker']}")
            return 1
    print("  Manifest integrity: PASS")

    # Data preflight
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED")
        return 1
    print("  Data preflight: PASS")
    print()

    # Individual lookback verifications
    all_pass = True
    for lookback in LOOKBACKS:
        print(f"  lookback {lookback}:")
        pass_this = verify_lookback(lookback, ARTIFACT_PATH)
        if pass_this:
            print(f"    lookback {lookback}: VERIFIED")
        else:
            print(f"    lookback {lookback}: FAILED")
            all_pass = False
        print()

    if all_pass:
        print("=== VERIFICATION RESULT: ALL PASS ===")
        return 0
    else:
        print("=== VERIFICATION RESULT: SOME FAILURES ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
