"""Novelty guard to prevent equivalent experiment re-execution.

This module implements the minimal novelty guard called for in LEARNING_EFFICIENCY.md
to convert accumulated history into actual avoidance of wasted search.

It computes a deterministic hash for a signal class's specification (lookback,
holding period, and any other relevant parameters) and checks whether that
same signal has already been tested and is currently admitted as evidence.

The guard operates independently of the frontier-first selection mechanism,
providing a simple, deterministic check for equivalent experiments.

Usage::

    # Before executing a new experiment, compute its novelty hash
    hash = compute_signal_hash(signal_type="momentum", lookback=5, hold_days=1)

    # Check whether this signal has already been executed
    is_duplicate = is_equivalent_experiment(hash)

    if not is_duplicate:
        # Execute the experiment
        run_experiment(signal_hash=hash)
    else:
        # Skip because we've already tested this signal class
        skip_experiment("novelty guard prevents re-execution")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from research.backtest.data import generate_bars
from research.backtest.engine import Signal


@dataclass(frozen=True)
class SignalSpec:
    """Specification for a signal class that determines its hash identity."""
    signal_type: str
    lookback: int
    hold_days: int
    regime_features: Optional[Dict[str, Any]] = None


def compute_signal_hash(spec: SignalSpec) -> str:
    """Compute deterministic hash for a signal specification.

    The hash identifies equivalent experiments: same signal type,
    lookback, holding period, and regime features (if any).

    Args:
        spec: Signal specification for the experiment

    Returns:
        SHA256 hash string (hex format)
    """
    spec_dict = asdict(spec)

    # Sort keys for deterministic hashing across Python versions
    spec_json = json.dumps(spec_dict, sort_keys=True, separators=(',', ':'))

    hash_obj = hashlib.sha256(spec_json.encode('utf-8'))
    return hash_obj.hexdigest()


def is_equivalent_experiment(spec_hash: str, state_path: str = "state/STATE.md") -> bool:
    """Check whether an equivalent experiment has already been executed.

    Looks for the hash in the durable evidence state (STATE.md) to determine
    whether this signal class is currently admitted as evidence.

    Args:
        spec_hash: Hash of the signal specification to check
        state_path: Path to the durable evidence state file

    Returns:
        True if an equivalent experiment exists in the evidence base,
        False otherwise
    """
    try:
        state_content = Path(state_path).read_text()
        
        # Check for the hash in various verdict contexts
        # Pattern 1: Check for supported/admitted signals that contain the hash
        if f"hash={spec_hash[:8]}" in state_content.lower():
            return True

        # Pattern 2: Check for momentum or other supported verdict sections
        # that indicate this signal has been tested and admitted
        momentum_pattern = "ADMITTED AS CANDIDATE POSITIVE EVIDENCE"
        if momentum_pattern in state_content and spec_hash[:16] in state_content:
            return True

        # Pattern 3: Direct hash-based detection
        hash_variations = [
            f"hash={spec_hash}",
            f"hash={spec_hash[:16]}",
            f"hash={spec_hash[:8]}",
            f"{spec_hash[:16]}..."
        ]

        for hash_var in hash_variations:
            if hash_var in state_content:
                return True

        # Pattern 4: Check for lookback-sweep verification artifacts
        # that match the hash
        if "lookback" in spec_hash and "momentum" in spec_hash.lower():
            if "lookback-sweep verification" in state_content and spec_hash[:16] in state_content:
                return True

    except Exception:
        pass

    return False


def should_skip_experiment(signal_type: str, lookback: int, hold_days: int,
                           regime_features: Optional[Dict[str, Any]] = None) -> bool:
    """Determine whether an experiment should be skipped due to novelty guard.

    This is the main entry point for the novelty guard. It computes the
    signal hash and checks whether an equivalent experiment has already been
    executed and is currently admitted as evidence.

    Args:
        signal_type: Type of signal (e.g., "momentum", "mean_reversion")
        lookback: Lookback period for the signal
        hold_days: Holding period in days
        regime_features: Optional regime features (e.g., volatility filters)

    Returns:
        True if the experiment should be skipped (duplicate),
        False if it should proceed (novel)
    """
    spec = SignalSpec(
        signal_type=signal_type,
        lookback=lookback,
        hold_days=hold_days,
        regime_features=regime_features
    )

    spec_hash = compute_signal_hash(spec)
    return is_equivalent_experiment(spec_hash)


def novelty_guard(func):
    """Decorator to enforce novelty guard on research experiments.

    Usage::

        @novelty_guard(signal_type="momentum", lookback=5, hold_days=1)
        def run_momentum_experiment():
            # Your experiment code here
            pass
    """
    def wrapper(*args, **kwargs):
        # Check if this experiment is a duplicate
        if should_skip_experiment(**func.__novelty_guard_args__):
            print(f"Novelty guard: skipping equivalent experiment ({func.__name__})")
            return None

        # Proceed with the experiment
        print(f"Novelty guard: proceeding with novel experiment ({func.__name__})")
        return func(*args, **kwargs)

    return wrapper
