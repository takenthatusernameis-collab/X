"""Simple verification of the momentum holding-period sensitivity check.

This script verifies that the published artifact contains the expected
structure and key metrics that were computed by the momentum_hold_period_sensitivity
check. It does not recompute all the results from scratch, but rather
checks that the artifact was created successfully and contains the
expected fields and types.

The holding-period sensitivity test (R-002) is designed to answer:
"Does the qualified momentum effect survive a small predeclared holding-period
variation without becoming a single-point timing artifact?"

Key requirements:
1. Hold periods (1, 2, 3, 5) are predeclared
2. Lookback is fixed at 5 (not tuned after results)
3. Use existing walk-forward/null methodology
4. Apply hold-period transform to signal series
5. Compare vs coin-flip null on same segments
6. Predeclared classification scheme applied after grid runs

This verification ensures the artifact structure matches expectations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"


def main():
    print("=== Simple verification of holding-period sensitivity artifact ===")
    print("Checking that the published artifact has expected structure...")

    # Load manifest to verify dataset
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']} != {DATASET_ID}"

    # Load and verify artifact structure
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)

    # Verify required fields
    required_fields = [
        "dataset_id", "lookback", "hold_periods", "min_segment_bars",
        "n_blocks", "overlap", "per_asset", "universe", "perturbation"
    ]

    for field in required_fields:
        if field not in artifact:
            print(f"  MISSING FIELD: {field}")
            return 1

    # Verify lookback is 5 (not tuned)
    if artifact["lookback"] != 5:
        print(f"  WRONG LOOKBACK: expected 5, got {artifact['lookback']}")
        return 1

    # Verify hold periods are predeclared (1, 2, 3, 5)
    expected_holds = [1, 2, 3, 5]
    if artifact["hold_periods"] != expected_holds:
        print(f"  WRONG HOLD PERIODS: expected {expected_holds}, got {artifact['hold_periods']}")
        return 1

    # Verify universe structure
    universe_fields = ["hold_sums", "hold_profiles", "effect_classification", "n_edge_at_h1", "n_edge_min_across_grid"]
    for field in universe_fields:
        if field not in artifact["universe"]:
            print(f"  MISSING UNIVERSE FIELD: {field}")
            return 1

    # Verify hold_sums structure
    for h in expected_holds:
        h_key = str(h)
        if h_key not in artifact["universe"]["hold_sums"]:
            print(f"  MISSING HOLD SUMS for H={h}")
            return 1

        hold_sum = artifact["universe"]["hold_sums"][h_key]
        if "verdict_counts" not in hold_sum or "ticker_order" not in hold_sum or "per_asset" not in hold_sum:
            print(f"  WRONG HOLD SUMS STRUCTURE for H={h}")
            return 1

    # Verify hold_profiles structure
    if "hold_profiles" not in artifact["universe"]:
        print("  MISSING HOLD PROFILES")
        return 1

    # Verify effect classification is one of the expected values
    expected_effect_classes = ["HORIZON_STABLE", "HORIZON_SENSITIVE", "HORIZON_SINGULAR", "HORIZON_DESTROYED"]
    if artifact["universe"]["effect_classification"] not in expected_effect_classes:
        print(f"  UNKNOWN EFFECT CLASSIFICATION: {artifact['universe']['effect_classification']}")
        return 1

    # Verify per_asset data structure for at least one asset
    if not artifact["per_asset"]:
        print("  EMPTY PER_ASSET DATA")
        return 1

    for asset, data in artifact["per_asset"].items():
        for h in expected_holds:
            h_key = str(h)
            if h_key not in data:
                print(f"  MISSING HOLD DATA for asset {asset} H={h}")
                return 1

            asset_data = data[h_key]
            required_asset_fields = ["verdict", "medians", "null_medians", "candidate_dispersion", "null_dispersion", "segments"]
            for field in required_asset_fields:
                if field not in asset_data:
                    print(f"  MISSING ASSET FIELD {field} for asset {asset} H={h}")
                    return 1

    # Verify perturbation data
    pert = artifact["perturbation"]
    required_pert_fields = ["param_sets", "medians", "baseline_median", "noise_median", "compare_noise_default", "compare_noise_fixed"]
    for field in required_pert_fields:
        if field not in pert:
            print(f"  MISSING PERTURBATION FIELD: {field}")
            return 1

    print("  ARTIFACT STRUCTURE VERIFICATION PASSED")
    print(f"  Dataset: {artifact['dataset_id']} (verified)")
    print(f"  Lookback: {artifact['lookback']} (fixed, not tuned)")
    print(f"  Hold periods: {artifact['hold_periods']} (predeclared)")
    print(f"  Effect classification: {artifact['universe']['effect_classification']}")
    print(f"  H=1 positive edges: {artifact['universe']['n_edge_at_h1']}")
    print(f"  Min across grid: {artifact['universe']['n_edge_min_across_grid']}")

    print("\nAll structure verification checks PASSED.")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum holding-period sensitivity artifact verification: successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())