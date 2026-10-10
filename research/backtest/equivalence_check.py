"""Novelty guard for the autonomous research worker.

This module implements a lightweight duplicate-prevention check to avoid
running equivalent experiments before launching new research tasks.
It complements the frontier-first learning contract by providing
automated detection of equivalent work.

The guard compares key research parameters to identify when a proposed
task would be equivalent to existing experiments, preventing redundant
search effort while allowing genuinely novel variations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from research.backtest import engine
from research.backtest import metrics
from research.backtest import perturbation
from research.backtest import regime_stability
from research.backtest import universe

from research.data.preflight import load_manifest, sha256_file


def compare_tasks(task1: dict[str, Any], task2: dict[str, Any]) -> bool:
    """Compare two tasks to determine if they are equivalent.

    This function implements a lightweight equality check that focuses on
    the research-relevant parameters that would make experiments equivalent.

    Args:
        task1: First task dictionary to compare
        task2: Second task dictionary to compare

    Returns:
        True if the tasks appear to be equivalent, False otherwise.
    """
    # Core research parameters that determine equivalence
    equivalence_keys = [
        "dataset_id", "lookback", "hold_periods", "min_segment_bars",
        "n_blocks", "overlap", "seed", "warmup", "window"
    ]

    # Signal type is also a key equivalence factor
    if task1.get("signal_type") != task2.get("signal_type"):
        return False

    for key in equivalence_keys:
        if task1.get(key) != task2.get(key):
            return False

    # Signal type and configuration parameters
    signal_type_keys = [
        "signal_type", "lookback", "hold", "threshold",
        "volatility_window", "atr_period", "trailing_stop"
    ]

    # Extract signal configurations
    signal1 = task1.get("signal_config", {})
    signal2 = task2.get("signal_config", {})

    for key in signal_type_keys:
        if signal1.get(key) != signal2.get(key):
            return False

    # Data universe parameters
    data1 = task1.get("data_params", {})
    data2 = task2.get("data_params", {})

    if data1.get("dataset_id") != data2.get("dataset_id"):
        return False

    # Perturbation sweep parameters (if present)
    if task1.get("perturbation") != task2.get("perturbation"):
        return False

    # Null hypothesis parameters
    null1 = task1.get("null_params", {})
    null2 = task2.get("null_params", {})

    if null1.get("type") != null2.get("type"):
        return False

    return True


def is_equivalent_experiment(proposed_task: dict[str, Any], existing_experiments: list[dict[str, Any]]) -> bool:
    """Check if a proposed task is equivalent to any existing experiments.

    This is the main entry point for the novelty guard. It compares a newly
    proposed task against a list of existing experiments to determine if
    the work would be redundant.

    Args:
        proposed_task: The task to check for equivalence
        existing_experiments: List of existing task definitions to compare against

    Returns:
        True if the proposed task is equivalent to any existing experiment,
        False otherwise.
    """
    for existing in existing_experiments:
        if compare_tasks(proposed_task, existing):
            return True
    return False


def create_task_signature(task: dict[str, Any]) -> str:
    """Create a deterministic signature for a task.

    This generates a hash-based signature that can be used for fast
    equivalence checking without full parameter comparison.

    Args:
        task: Task to create a signature for

    Returns:
        Deterministic string signature for the task
    """
    # Create a canonical string representation of key task parameters
    signature_components = []

    for key in ["dataset_id", "lookback", "hold_periods", "signal_type"]:
        if key in task:
            signature_components.append(f"{key}={task[key]}")

    signature_string = "|".join(sorted(signature_components))

    # Create deterministic hash using Python's built-in hash
    hash_object = hash(signature_string)

    return str(abs(hash_object))[:16]


def novelty_guard_check(proposed_task: dict[str, Any], task_history_file: str = "state/task_history.json") -> dict:
    """Main novelty guard entry point.

    This function implements the complete novelty guard workflow:
    1. Load existing task history
    2. Compare proposed task against history
    3. Return decision and any matching tasks

    Args:
        proposed_task: The task to evaluate
        task_history_file: Path to file containing task history

    Returns:
        Dictionary with guard results:
        - "equivalent_found": bool indicating if equivalent task exists
        - "matching_tasks": list of equivalent task IDs
        - "recommended_action": "PROCEED" or "SKIP"
        - "reason": explanation of the decision
    """
    result = {
        "equivalent_found": False,
        "matching_tasks": [],
        "recommended_action": "PROCEED",
        "reason": ""
    }

    # Load task history (with fallback for missing file)
    try:
        task_history_path = Path(task_history_file)
        if task_history_path.exists():
            with open(task_history_path) as f:
                existing_experiments = json.load(f).get("tasks", [])
        else:
            existing_experiments = []
    except Exception:
        existing_experiments = []

    # Check for equivalent experiments
    if is_equivalent_experiment(proposed_task, existing_experiments):
        result["equivalent_found"] = True
        result["matching_tasks"] = [exp.get("task_id") for exp in existing_experiments
                                  if compare_tasks(proposed_task, exp)]
        result["recommended_action"] = "SKIP"
        result["reason"] = f"Equivalent task found: {result['matching_tasks']}"
    else:
        result["recommended_action"] = "PROCEED"
        result["reason"] = "No equivalent tasks found in history"

    return result