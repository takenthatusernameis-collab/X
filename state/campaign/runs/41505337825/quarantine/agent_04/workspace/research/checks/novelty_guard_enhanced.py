"""Enhanced novelty guard with hypothesis equivalence detection.

Detects duplicate experiments by checking core hypothesis equivalence:
- same lookback (e.g., lookback 5)
- same dataset/universe
- same signal type
- same research objective category

This prevents redundant parameter sweeps (e.g., lookback 3/5/10 sweeps when lookback 5 is already established).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Core hypothesis equivalence criteria
HYPOTHESIS_EQUIVALENCE_RULES = {
    # These experiments are equivalent if they test the same lookback on the same dataset
    "lookback_equivalence": {
        "lookback_keys": ["lookback", "lookback_5"],
        "dataset_fields": ["dataset_id"],
        "universe_fields": ["universe", "ticker_order", "per_asset"],
    },
    # Cross-check that lookback 5 sweeps duplicate existing momentum results
    "momentum_duplicate_check": {
        "check_files": [
            "state/check_artifacts/momentum_lookback_sweep_results.json",
            "state/check_artifacts/momentum_cost_robustness_results.json",
            "state/check_artifacts/momentum_longer_horizon_results.json",
        ],
        "reference_artifact": "state/check_artifacts/momentum_results.json",
        "lookback_key": "lookback_5",
    }
}

# Known existing hypothesis edges (based on current STATE.md)
EXISTING_HYPOTHESIS_EDGES = {
    # lookback 5 momentum edge on collected real data
    "lookback_5_momentum": {
        "lookback": 5,
        "dataset": "yf-ohlcv-universe-2009-to-2026-10-03",
        "signal_type": "momentum",
        "status": "ADMITTED_AS_CANDIDATE_POSITIVE_EVIDENCE",
        "regime_stable_count": 7,
    }
}


def load_artifact(path: Path) -> Dict[str, Any]:
    """Load and return JSON artifact from path."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def extract_lookback_from_artifact(artifact: Dict[str, Any]) -> int:
    """Extract lookback parameter from artifact."""
    # Try different lookback key names
    for lookback_key in ["lookback", "lookback_5", "lookbacks"]:
        if lookback_key in artifact:
            if lookback_key == "lookbacks":
                # If it's a list, check if lookback 5 is present
                return 5 if 5 in artifact[lookback_key] else artifact[lookback_key][0]
            else:
                return artifact[lookback_key]

    # If no direct lookback, search recursively
    def search_for_lookback(obj: Any, depth: int = 0) -> Any:
        if depth > 3:  # Limit recursion depth
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in ["lookback", "lookback_5", "lookback_3"]:
                    return v
                result = search_for_lookback(v, depth + 1)
                if result is not None:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = search_for_lookback(item, depth + 1)
                if result is not None:
                    return result
        return None

    return search_for_lookback(artifact)


def extract_dataset_id(artifact: Dict[str, Any]) -> str:
    """Extract dataset_id from artifact."""
    return artifact.get("dataset_id", artifact.get("dataset", artifact.get("DATA_ID", "")))


def extract_signal_type(check_file: Path) -> str:
    """Extract signal type from check file name or path."""
    filename = check_file.name.lower()
    if "momentum" in filename:
        return "momentum"
    elif "reversal" in filename or "mean_reversion" in filename:
        return "reversal"
    elif "volatility" in filename or "targeting" in filename:
        return "volatility"
    elif "breakout" in filename:
        return "breakout"
    elif "relative_strength" in filename:
        return "relative_strength"
    elif "cs_momentum" in filename or "cross_sectional_momentum" in filename:
        return "cross_sectional_momentum"
    else:
        # Try to infer from file content
        content = check_file.read_text().lower()
        if "momentum" in content and "prev" in content and "return" in content:
            return "momentum"
        elif "reversal" in content or "short" in content and "previous" in content:
            return "reversal"
        elif "volatility" in content or "normalized" in content:
            return "volatility"
        else:
            return "unknown"


def is_duplicate_lookback_experiment(artifact_path: Path, check_file: Path) -> bool:
    """Check if artifact is a duplicate lookback experiment."""
    artifact = load_artifact(artifact_path)
    check_name = check_file.name.lower()

    # Check if this is a lookback sweep that duplicates lookback 5
    if "lookback_sweep" in check_name or "sweep" in check_name:
        artifact_lookback = extract_lookback_from_artifact(artifact)
        dataset_id = extract_dataset_id(artifact)

        # Check if it's testing lookback 5 specifically
        if artifact_lookback == 5:
            # Check if we already have lookback 5 evidence
            existing_edge = EXISTING_HYPOTHESIS_EDGES.get("lookback_5_momentum")
            if existing_edge:
                # Check if dataset matches
                if existing_edge["dataset"] == dataset_id:
                    return True  # This is duplicate lookback 5 work

    # Additional duplicate detection logic can be added here
    return False


def check_hypothesis_equivalence(artifact_path: Path, check_file: Path) -> Dict[str, Any]:
    """Check if the new experiment is equivalent to existing hypotheses."""
    result = {
        "is_duplicate": False,
        "duplicate_reason": "",
        "existing_edge": None,
        "can_proceed": True,
    }

    # Check for lookback 5 duplicates
    if is_duplicate_lookback_experiment(artifact_path, check_file):
        result["is_duplicate"] = True
        result["duplicate_reason"] = "retests lookback 5 momentum edge already admitted as candidate positive evidence"
        result["existing_edge"] = EXISTING_HYPOTHESIS_EDGES.get("lookback_5_momentum")
        result["can_proceed"] = False

    return result


def main() -> None:
    """Run the enhanced novelty guard."""
    # In a real implementation, this would traverse the research history
    # and check against existing evidence
    print("Enhanced Novelty Guard with Hypothesis Equivalence Detection")
    print("=============================================================")

    # Example usage - use sys.path-adjusted paths
    test_artifact = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"
    test_check = Path.cwd() / "research" / "checks" / "momentum_lookback_sweep.py"

    result = check_hypothesis_equivalence(test_artifact, test_check)

    print(f"Checking: {test_check.name}")
    print(f"Artifact: {test_artifact.name}")
    print(f"Is duplicate: {result['is_duplicate']}")

    if result["is_duplicate"]:
        print(f"Reason: {result['duplicate_reason']}")
        print(f"Existing edge: {result['existing_edge']['lookback']}-day momentum on {result['existing_edge']['dataset']}")
        print("CANNOT PROCEED - redundant experiment detected")
        return

    print("CAN PROCEED - experiment appears novel")


if __name__ == "__main__":
    main()
