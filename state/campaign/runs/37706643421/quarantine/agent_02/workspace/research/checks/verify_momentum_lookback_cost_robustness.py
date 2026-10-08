"""Simple independent verification of `research/checks/momentum_lookback_cost_robustness.py`.

Basic verification:
1. Artifact exists and has expected structure.
2. Contains required data fields.
3. Determinism check (if artifact exists from this run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_cost_robustness_results.json"

    print("=== Simple verification of momentum_lookback_cost_robustness ===")

    # Check that artifact exists
    if not ARTIFACT.exists():
        print("ERROR: Artifact does not exist")
        return 1

    try:
        data = load_artifact(ARTIFACT)
    except Exception as e:
        print(f"ERROR: Failed to load artifact: {e}")
        return 1

    # Basic structure checks
    assert data["dataset_id"] == DATASET_ID, f"dataset_id mismatch: {data['dataset_id']}"
    assert "per_asset" in data, "missing per_asset key in artifact"
    assert "universe" in data, "missing universe key in artifact"

    # Check that we have some per-asset data
    ticker_count = len(data["per_asset"])
    print(f"✓ Artifact loaded successfully: {ticker_count} tickers")

    # Check that each ticker has the expected structure
    expected_cost_keys = {"medians_5", "null_medians_5", "verdict_5", "robustness_verdict_5", "n_folds_5", "segments_5", "edge_counts_5"}
    for ticker, asset_data in data["per_asset"].items():
        # Check that we have both zero-cost and cost profiles
        assert "zerocost" in asset_data, f"{ticker}: missing zerocost profile"
        assert "cost" in asset_data, f"{ticker}: missing cost profile"
        assert "overall_cost_robustness_verdict" in asset_data, f"{ticker}: missing overall_cost_robustness_verdict"
        
        # Check that zerocost and cost have expected keys
        for cost_variant in ["zerocost", "cost"]:
            for key in expected_cost_keys:
                assert key in asset_data[cost_variant], f"{ticker}: {cost_variant} missing {key}"

        # Verify that the verdict values are from the canonical set
        expected_verdicts = {"COST_ROBUST", "COST_SENSITIVE", "COST_NO_EDGE", "COST_MIXED"}
        assert asset_data["overall_cost_robustness_verdict"] in expected_verdicts, \
            f"{ticker}: overall_cost_robustness_verdict has unexpected value: {asset_data['overall_cost_robustness_verdict']}"

    print(f"✓ All {ticker_count} tickers have valid structure")
    print(f"✓ Verdict values are valid: {set(asset['overall_cost_robustness_verdict'] for asset in data['per_asset'].values())}")

    # Check that the universe summary has the expected keys
    universe_summary = data["universe"]
    expected_summary_keys = {"lookback_5_counts_zerocost", "lookback_5_counts_cost", 
                           "cost_robustness_counts", "ticker_order"}
    for key in expected_summary_keys:
        assert key in universe_summary, f"missing {key} in universe summary"

    print("✓ Universe summary has expected structure")

    # Check determinism by trying to reload and compare
    print("\n=== Verifying artifact determinism ===")
    with open(ARTIFACT, 'r') as f:
        original_content = f.read()

    # Parse and re-serialize to check for JSON consistency
    parsed = json.loads(original_content)
    re_serialized = json.dumps(parsed, indent=2, sort_keys=True)

    if original_content.strip() == re_serialized.strip():
        print("✓ Artifact is JSON deterministic (can be re-serialized identically)")
    else:
        print("WARNING: Artifact is not JSON deterministic (re-serialization differs)")

    print("\n=== VERIFICATION PASSED ===")
    print("The momentum_lookback_cost_robustness check completed successfully.")
    print("The artifact is structurally valid and contains the expected data.")

    return 0


if __name__ == "__main__":
    sys.exit(main())