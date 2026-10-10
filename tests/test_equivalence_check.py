"""Tests for the novelty guard equivalence check.

This test suite validates the lightweight duplicate-prevention functionality
that complements the frontier-first learning contract.
"""

import pytest

from research.backtest.equivalence_check import (
    compare_tasks,
    is_equivalent_experiment,
    create_task_signature,
    novelty_guard_check
)


def test_compare_tasks_identical():
    """Test that identical tasks are correctly identified as equivalent."""
    task1 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "hold_periods": (1, 2, 3, 5),
        "signal_type": "momentum",
        "seed": 42
    }

    task2 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "hold_periods": (1, 2, 3, 5),
        "signal_type": "momentum",
        "seed": 42
    }

    assert compare_tasks(task1, task2) is True


def test_compare_tasks_different_dataset():
    """Test that tasks with different datasets are not equivalent."""
    task1 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    task2 = {
        "dataset_id": "yf-ohlcv-universe-2023-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    assert compare_tasks(task1, task2) is False


def test_compare_tasks_different_signal_type():
    """Test that tasks with different signal types are not equivalent."""
    task1 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    task2 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "mean_reversion"
    }

    assert compare_tasks(task1, task2) is False


def test_is_equivalent_experiment_single():
    """Test single experiment equivalence check."""
    proposed = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    existing = [
        {
            "task_id": "R-001",
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 5,
            "signal_type": "momentum"
        }
    ]

    assert is_equivalent_experiment(proposed, existing) is True


def test_is_equivalent_experiment_multiple():
    """Test equivalence check against multiple experiments."""
    proposed = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    existing = [
        {
            "task_id": "R-001",
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 3,
            "signal_type": "mean_reversion"
        },
        {
            "task_id": "R-002",
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 5,
            "signal_type": "momentum"
        }
    ]

    assert is_equivalent_experiment(proposed, existing) is True


def test_is_equivalent_experiment_no_match():
    """Test when proposed task has no equivalent."""
    proposed = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    existing = [
        {
            "task_id": "R-001",
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 3,
            "signal_type": "mean_reversion"
        }
    ]

    assert is_equivalent_experiment(proposed, existing) is False


def test_create_task_signature_deterministic():
    """Test that task signatures are deterministic."""
    task1 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    task2 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    sig1 = create_task_signature(task1)
    sig2 = create_task_signature(task2)

    assert sig1 == sig2


def test_create_task_signature_different():
    """Test that different tasks have different signatures."""
    task1 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    task2 = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 3,
        "signal_type": "momentum"
    }

    sig1 = create_task_signature(task1)
    sig2 = create_task_signature(task2)

    assert sig1 != sig2


def test_novelty_guard_check_no_equivalence():
    """Test novelty guard when no equivalent tasks found."""
    proposed = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    result = novelty_guard_check(proposed, task_history_file="/tmp/nonexistent.json")

    assert result["equivalent_found"] is False
    assert result["recommended_action"] == "PROCEED"
    assert "No equivalent tasks found" in result["reason"]


def test_novelty_guard_check_equivalence_found():
    """Test novelty guard when equivalent tasks found."""
    import json
    import tempfile

    # Create a temporary task history file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        task_history = {
            "tasks": [
                {
                    "task_id": "R-001",
                    "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
                    "lookback": 5,
                    "signal_type": "momentum"
                }
            ]
        }
        json.dump(task_history, f)
        temp_file = f.name

    try:
        proposed = {
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 5,
            "signal_type": "momentum"
        }

        result = novelty_guard_check(proposed, temp_file)

        assert result["equivalent_found"] is True
        assert result["matching_tasks"] == ["R-001"]
        assert result["recommended_action"] == "SKIP"
        assert "Equivalent task found" in result["reason"]
    finally:
        import os
        os.unlink(temp_file)


def test_novelty_guard_check_missing_file():
    """Test novelty guard when task history file is missing."""
    proposed = {
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "signal_type": "momentum"
    }

    result = novelty_guard_check(proposed, task_history_file="/tmp/nonexistent.json")

    assert result["equivalent_found"] is False
    assert result["recommended_action"] == "PROCEED"
    assert "No equivalent tasks found" in result["reason"]


def test_equivalent_experiment_real_scenario():
    """Test a realistic duplicate experiment scenario."""
    # These represent similar momentum experiments with same core parameters
    existing_experiments = [
        {
            "task_id": "R-002",
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 5,
            "hold_periods": (1, 2, 3, 5),
            "signal_type": "momentum",
            "seed": 42
        },
        {
            "task_id": "R-004",
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 5,
            "hold_periods": (1, 2, 3, 5),
            "signal_type": "momentum",
            "seed": 42
        }
    ]

    proposed = {
        "task_id": "R-006",
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 5,
        "hold_periods": (1, 2, 3, 5),
        "signal_type": "momentum",
        "seed": 42
    }

    assert is_equivalent_experiment(proposed, existing_experiments) is True


def test_equivalent_experiment_different_parameters():
    """Test that tasks with different parameters are not equivalent."""
    existing_experiments = [
        {
            "task_id": "R-002",
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 5,
            "hold_periods": (1, 2, 3, 5),
            "signal_type": "momentum",
            "seed": 42
        }
    ]

    proposed = {
        "task_id": "R-006",
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "lookback": 3,  # Different lookback
        "hold_periods": (1, 2, 3, 5),
        "signal_type": "momentum",
        "seed": 42
    }

    assert is_equivalent_experiment(proposed, existing_experiments) is False