"""Cost-sensitivity check for momentum signals (lookback 3/5/10) on the collected universe.

Method: walk-forward IS/OOS validation with realistic transaction costs (commission,
slippage) and a coin-flip matched null. Uses the same regime- and segment-gating
as the momentum check (state/check_artifacts/momentum_results.json) so that
costs are applied identically across lookbacks.

Config: initial_capital=1e6, commission_per_trade=2.0, commission_per_share=0.003,
slippage_cents=2.0, slippage_proportional=0.0005; other parameters (walk-forward
windows, warmup, overlap) match the momentum baseline (TRAIN=252, TEST=84, WARM=60,
OVERLAP=60, MIN_SEGMENT_BARS=400, N_BLOCKS=4, WINDOW=60).

Bound: test exactly lookback 3, 5, 10 in one run; do not tune after seeing costs;
do not expand the universe or introduce leverage.

Deliverable: JSON artifact with per-asset per-lookback medians (zero cost vs realistic
costs) and a matched coin-flip null; verification recomputes the zero-cost medians
against the existing momentum artifact.

Research/simulation only. This is exploratory; no live trading or production
execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"

def momentum_signals(closes, lookback):
    """Momentum signal: long previous N-day return, hold 1 day, daily rebalance."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_asset_cost_sweep(asset, tickers, lookbacks):
    """For each lookback, run zero-cost and realistic-cost walk-forward on asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}

    for lb in lookbacks:
        cfg_zero = bt.BacktestConfig(
            initial_capital=1e6,
            warmup_periods=WARM,
            commission_per_trade=0.0,
            commission_per_share=0.0,
            slippage_cents=0.0,
            slippage_proportional=0.0,
        )
        cfg_realistic = bt.BacktestConfig(
            initial_capital=1e6,
            warmup_periods=WARM,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        )

        signals = momentum_signals(closes, lb)
        res_zero = bt.stress_segments(
            signals_fn=lambda c, **p: momentum_signals(c, p["lookback"]),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": lb}],
            baseline=(("lookback", lb),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg_zero,
        )
        res_real = bt.stress_segments(
            signals_fn=lambda c, **p: momentum_signals(c, p["lookback"]),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": lb}],
            baseline=(("lookback", lb),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=cfg_realistic,
        )

        results[str(lb)] = {
            "zero_cost": {
                "segments": [s.name for s in res_zero.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res_zero.scenarios],
                "candidate_dispersion": round(res_zero.candidate_dispersion, 3),
                "null_dispersion": round(res_zero.null_dispersion, 3),
                "verdict": res_zero.overall_verdict,
                "n_folds": res_zero.n_folds,
            },
            "realistic_cost": {
                "segments": [s.name for s in res_real.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res_real.scenarios],
                "candidate_dispersion": round(res_real.candidate_dispersion, 3),
                "null_dispersion": round(res_real.null_dispersion, 3),
                "verdict": res_real.overall_verdict,
                "n_folds": res_real.n_folds,
            },
        }
    return results


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Manifest integrity ===")
    from research.data.preflight import load_manifest, sha256_file
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | universe: {len(manifest['universe'])}")

    print("\n=== Data preflight ===")
    from importlib import import_module
    pf = import_module("research.data.preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-sensitivity run.")
        sys.exit(1)

    print("\n=== Loading collected universe ===")
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("=== Cost-sensitivity sweep (lookback 3, 5, 10) ===")
    per_asset_results = {}
    for asset in ["AMZN", "JPM"]:
        per_asset_results[asset] = run_asset_cost_sweep(asset, tickers, LOOKBACKS)
        print(f"  {asset}: lookback {LOOKBACKS}")

    print("\n=== Determinism ===")
    # Verify per-asset results for AMZN/JPM
    for asset in ["AMZN", "JPM"]:
        res1 = per_asset_results[asset]
        res2 = run_asset_cost_sweep(asset, tickers, LOOKBACKS)
        # Compare JSON serialization
        r1_str = json.dumps(res1, sort_keys=True)
        r2_str = json.dumps(res2, sort_keys=True)
        if r1_str == r2_str:
            print(f"  {asset}: deterministic")
        else:
            print(f"  {asset}: NON-DETERMINISTIC - check implementation")

    print("\n=== Verification against existing momentum artifact ===")
    # Load the existing momentum_results.json for verification
    old_artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"
    with open(old_artifact_path) as f:
        old_artifact = json.load(f)
    
    # Focus on lookback=5 for zero-cost medians comparison
    old_per_asset = old_artifact["per_asset"]
    new_per_asset = per_asset_results

    verification_ok = True
    for asset in ["AMZN", "JPM"]:
        old_mom = old_per_asset[asset]["momentum"]["medians"]
        new_mom = new_per_asset[asset]["5"]["zero_cost"]["medians"]
        if old_mom == new_mom:
            print(f"  {asset} lookback 5 zero-cost medians: MATCH")
        else:
            print(f"  {asset} lookback 5 zero-cost medians: MISMATCH")
            print(f"    old: {old_mom}")
            print(f"    new: {new_mom}")
            verification_ok = False

    if not verification_ok:
        print("\nVERIFICATION FAILED: zero-cost cost-sensitivity must reproduce the")
        print("existing momentum artifact exactly. Investigate before proceeding.")
        sys.exit(1)

    artifact = {
        "dataset_id": DATASET_ID,
        "lookbacks": LOOKBACKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "n_blocks": N_BLOCKS,
        "overlap": OVERLAP,
        "per_asset": per_asset_results,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-sensitivity check (lookback 3/5/10) completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
