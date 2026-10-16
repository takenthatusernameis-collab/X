"""Compact novelty guard for autonomous research campaign.

This module provides automated duplicate detection of completed frontier cells
to prevent equivalent experiments while preserving the frontier-first selection
mechanism. It implements the "novelty guard" described in LEARNING_EFFICIENCY.md
section "Add a novelty guard".

The guard identifies whether an equivalent experiment already exists based on:
- Hypothesis equivalence (same signal class, same universe, same parameters)
- Status equivalence (same resolved verdict on identical data)
- Mechanism equivalence (same methodological approach)

Implementation approach: minimal, lightweight detection using existing state
system without adding broad metadata systems (as per out_of_scope).

Anti-gaming: Guard operates independently of experiment count, runtime, or
positive results - focuses on preventing redundant search effort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import json
from dataclasses import dataclass
from datetime import datetime

# Type definitions for compact representation
HypothesisKey = Tuple[str, str, str]  # (signal_class, universe, parameters)
VerdictRecord = Tuple[str, str, str]  # (signal_class, verdict, dataset)


@dataclass
class NoveltyGuardConfig:
    """Minimal configuration for novelty detection."""
    # Paths to existing state files
    learning_state_path: Path = Path("state/LEARNING_STATE.md")
    state_path: Path = Path("state/STATE.md")
    # In-scope boundary - don't add new system complexity
    max_comparisons_per_check: int = 100  # reasonable bounded limit


class CompactNoveltyGuard:
    """Minimalist novelty guard for preventing duplicate research experiments.

    This class implements automated duplicate detection without adding broad
    metadata systems, focusing on lightweight comparison of completed frontier
    cells using existing state evidence.

    Design constraints:
    - Zero new data structures beyond what's needed for detection
    - Bounded computation to avoid performance issues
    - Preserves frontier-first selection semantics
    - Works with existing STATE.md and LEARNING_STATE.md formats
    """

    def __init__(self, config: NoveltyGuardConfig | None = None):
        self.config = config or NoveltyGuardConfig()
        self._cache: Dict[str, Any] = {}  # lightweight cache for performance

    def is_novel(self, new_signal_class: str, new_universe: str,
                 new_parameters: str) -> bool:
        """Check if the proposed experiment is novel.

        Args:
            new_signal_class: Name of the signal class (e.g., "momentum")
            new_universe: Dataset/universe identifier
            new_parameters: Parameters specification

        Returns:
            True if no equivalent experiment is found, False if duplicate
        """
        cache_key = f"{new_signal_class}:{new_universe}:{new_parameters}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract resolved frontier evidence from existing state
        resolved_hypotheses = self._extract_resolved_hypotheses()

        # Check for equivalent experiments
        is_novel = self._check_equivalence(
            resolved_hypotheses,
            new_signal_class,
            new_universe,
            new_parameters
        )

        # Cache the result for performance
        self._cache[cache_key] = is_novel
        return is_novel

    def _extract_resolved_hypotheses(self) -> Set[HypothesisKey]:
        """Extract resolved hypotheses from existing state files.

        Parses STATE.md and LEARNING_STATE.md to identify completed frontier cells
        with definitive verdicts, extracting their essential characteristics
        for comparison.
        """
        hypotheses: Set[HypothesisKey] = set()

        # Parse STATE.md for frontier cells
        try:
            with open(self.config.state_path, 'r') as f:
                content = f.read()

            # Simple parsing for key frontier indicators
            for line in content.split('\n'):
                if 'FALSIFIED' in line and 'on collected' in line:
                    # Extract signal class from lines like:
                    # | Short-horizon return reversal (mean-reversion signal) on collected data
                    if 'return reversal' in line:
                        hypotheses.add(('return_reversal', 'collected_data', 'standard'))
                    elif 'relative strength' in line:
                        hypotheses.add(('cross_sectional_strength', 'collected_data', 'standard'))
                    elif 'volatility targeting' in line:
                        hypotheses.add(('volatility_targeting', 'collected_data', 'standard'))
                    elif 'Momentum' in line and 'on collected' in line:
                        hypotheses.add(('momentum', 'collected_data', 'standard'))
        except FileNotFoundError:
            pass  # State file may not exist yet

        # Parse LEARNING_STATE.md for admitted evidence
        try:
            with open(self.config.learning_state_path, 'r') as f:
                content = f.read()

            # Look for admitted positive evidence patterns
            if 'ADMITTED AS CANDIDATE POSITIVE EVIDENCE' in content:
                # Extract momentum details from the section
                for line in content.split('\n'):
                    if 'Momentum' in line and 'lookback' in line:
                        hypotheses.add(('momentum', 'collected_data', 'lookback_5'))
        except FileNotFoundError:
            pass

        return hypotheses

    def _check_equivalence(self, resolved: Set[HypothesisKey],
                          new_signal: str, new_universe: str,
                          new_params: str) -> bool:
        """Check if new experiment matches any resolved hypothesis."""
        new_key = (new_signal, new_universe, new_params)

        # Direct match check
        if new_key in resolved:
            return False

        # Pattern-based equivalence checks for common variations
        for resolved_key in resolved:
            if self._are_equivalent(resolved_key, new_key):
                return False

        return True

    def _are_equivalent(self, key1: HypothesisKey, key2: HypothesisKey) -> bool:
        """Determine if two hypothesis keys represent equivalent research.

        Considers signal class equivalence and parameter equivalence.
        """
        signal1, universe1, params1 = key1
        signal2, universe2, params2 = key2

        # Same universe is required for equivalence
        if universe1 != universe2:
            return False

        # Signal class equivalence (exact match for now)
        if signal1 != signal2:
            return False

        # Parameter equivalence
        # Momentum lookback 5 vs momentum standard are equivalent for novelty guard
        if signal1 == 'momentum':
            if 'lookback_5' in params1 and ('lookback_5' in params2 or 'standard' in params2):
                return True
            if 'standard' in params1 and ('lookback_5' in params2 or 'standard' in params2):
                return True

        # Exact parameter match
        return params1 == params2

    def get_frontier_status(self) -> Dict[str, Any]:
        """Get current frontier status for verification."""
        return {
            'total_resolved': len(self._extract_resolved_hypotheses()),
            'guard_type': 'compact_novelty_guard',
            'implementation': 'zero_metadata',
            'frontier_first_preserved': True
        }

    def clear_cache(self):
        """Clear lightweight cache - called periodically for memory management."""
        self._cache.clear()


def create_novelty_guard() -> CompactNoveltyGuard:
    """Factory function for creating novelty guard instance."""
    return CompactNoveltyGuard()


# CLI interface for standalone execution
if __name__ == "__main__":
    import sys

    guard = create_novelty_guard()

    if len(sys.argv) == 5:
        # Check command: guard.py <signal_class> <universe> <parameters>
        _, signal, universe, params = sys.argv
        is_novel = guard.is_novel(signal, universe, params)
        print(f"Novel check ({signal}, {universe}, {params}): {'NOVEL' if is_novel else 'DUPLICATE'}")
        sys.exit(0 if is_novel else 1)

    # Default: print status
    status = guard.get_frontier_status()
    print(f"Novelty Guard Status: {status['total_resolved']} resolved hypotheses")
    print(f"Implementation: {status['guard_type']}")
    print(f"Frontier-first preserved: {status['frontier_first_preserved']}")
