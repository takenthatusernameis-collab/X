#!/usr/bin/env python3
"""
Simple verification of momentum cost-robustness artifact.

This verifies that the cost-sensitivity check results are properly stored
and contain the expected structure.
"""
import json
from pathlib import Path

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"


def main():
    print("=== Momentum Cost-Robustness Artifact Verification ===")

    try:
        with open(ARTIFACT_PATH, "r") as f:
            artifact = json.load(f)

        print("✓ Artifact loaded successfully")

        # Verify required fields
        required_fields = [
            "dataset_id",
            "seed",
            "lookback_grid",
            "cost_scenarios",
            "manifest",
        ]

        missing_fields = [field for field in required_fields if field not in artifact]
        if missing_fields:
            print(f"✗ Missing fields: {missing_fields}")
            return 1

        print("✓ All required fields present")

        # Verify data
        print(f"  Dataset: {artifact['dataset_id']}")
        print(f"  Seed: {artifact['seed']}")
        print(f"  Lookback grid: {artifact['lookback_grid']}")
        print(f"  Cost scenarios: {len(artifact['cost_scenarios'])}")

        # Verify each cost scenario
        total_regime_stable = 0
        total_assets_checked = 0

        for cost_scenario in artifact["cost_scenarios"]:
            cost_name = cost_scenario.get("cost_name", "unknown")
            print(f"\nCost scenario: {cost_name}")

            for lookback in artifact["lookback_grid"]:
                key = str(lookback)
                if key in cost_scenario.get("results_by_lookback", {}):
                    results = cost_scenario["results_by_lookback"][key]
                    n_stable = sum(1 for data in results.get("per_asset_verdicts", {}).values() if data == "REGIME_STABLE")
                    n_total = len(results.get("per_asset_verdicts", {}))

                    total_regime_stable += n_stable
                    total_assets_checked += n_total

                    print(f"  Lookback {lookback}: {n_stable}/{n_total} REGIME_STABLE assets")

        if total_assets_checked > 0:
            overall_stable_rate = (total_regime_stable / total_assets_checked) * 100
            print(f"\nOverall REGIME_STABLE rate: {total_regime_stable}/{total_assets_checked} ({overall_stable_rate:.1f}%)")

        print("\n✓ Verification completed successfully")
        return 0

    except FileNotFoundError:
        print(f"✗ Artifact file not found: {ARTIFACT_PATH}")
        return 1
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in artifact: {e}")
        return 1
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
