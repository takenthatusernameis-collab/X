#!/usr/bin/env python3
"""
Minimal novelty guard for research/checks.

Prevents equivalent experiments by scanning the research/checks/check_registry.json
before launching any new check. Rejects duplicates unless they serve one of the
allowed exceptions:
  - new hypothesis,
  - diagnostic purpose,
  - materially new data,
  - independent verification need,
  - clearly different information objective.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Path to the check registry
CHECK_REGISTRY_PATH = Path(__file__).parent / "check_registry.json"


def load_check_registry() -> Dict[str, Any]:
    """Load the check registry from JSON file."""
    with open(CHECK_REGISTRY_PATH, 'r') as f:
        return json.load(f)


def extract_equivalent_signature(check_config: Dict[str, Any]) -> str:
    """Extract an equivalent signature for a check configuration.

    The signature combines the objective (primary question) and key parameters
    that determine if a check is essentially the same as another.
    """
    objective = check_config.get("objective", "")
    # Key parameters that define equivalence
    lookback = check_config.get("lookback")
    data_requirements = tuple(sorted(check_config.get("data_requirements", [])))
    universe = check_config.get("universe")

    # Build signature components
    components = [objective]
    if lookback is not None:
        components.append(f"lookback:{lookback}")
    if data_requirements:
        components.append(f"data:{data_requirements}")
    if universe is not None:
        components.append(f"universe:{universe}")

    return "|".join(components)


def check_allowed_exceptions(new_check: Dict[str, Any], existing_check: Dict[str, Any]) -> bool:
    """Check if the new check qualifies for one of the allowed exceptions.

    Returns True if the new check should be allowed despite being equivalent.
    """
    new_purpose = new_check.get("purpose", "")
    existing_purpose = existing_check.get("purpose", "")

    # Check for new hypothesis
    if "hypothesis" in new_purpose.lower() or "new" in new_purpose.lower():
        return True

    # Check for diagnostic purpose
    if "diagnostic" in new_purpose.lower():
        return True

    # Check for materially new data
    new_data = new_check.get("data_requirements", [])
    existing_data = existing_check.get("data_requirements", [])
    if set(new_data) - set(existing_data):
        return True

    # Check for independent verification need
    if "verify" in new_purpose.lower() or "independent" in new_purpose.lower():
        return True

    # Check for clearly different information objective
    if new_check.get("objective") != existing_check.get("objective"):
        return True

    return False


def should_launch_check(new_check_config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Determine if a new check should launch based on novelty guard rules.

    Returns a tuple of (should_launch, reason).
    """
    try:
        registry = load_check_registry()
    except (FileNotFoundError, json.JSONDecodeError):
        # If registry doesn't exist or is invalid, allow the check to launch
        return True, "registry unavailable"

    new_signature = extract_equivalent_signature(new_check_config)
    new_name = new_check_config.get("name", "unnamed_check")

    # Check for equivalent existing checks
    for existing_name, existing_check in registry.items():
        existing_signature = extract_equivalent_signature(existing_check)
        if new_signature == existing_signature:
            # Check if allowed exceptions apply
            if check_allowed_exceptions(new_check_config, existing_check):
                continue  # Allow the check
            else:
                return False, f"equivalent check '{existing_name}' already exists"

    # No equivalent checks found, or all were allowed by exceptions
    return True, "no equivalent checks found"


if __name__ == "__main__":
    # Example usage for testing
    test_check = {
        "name": "test_check",
        "objective": "Test whether momentum survives transaction costs",
        "lookback": 3,
        "data_requirements": ["yf-ohlcv-universe-2009-to-2026-10-03"],
        "universe": ["AAPL", "MSFT", "GOOGL"],
        "purpose": "new hypothesis test"
    }

    should_launch, reason = should_launch_check(test_check)
    print(f"Should launch: {should_launch}")
    print(f"Reason: {reason}")
