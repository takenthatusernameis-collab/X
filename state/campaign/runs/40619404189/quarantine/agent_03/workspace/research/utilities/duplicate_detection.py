"""Duplicate detection and novelty guard utility.

This module implements a minimal novelty guard to prevent repeating equivalent
experiments while preserving the frontier-first selection efficiency.

Based on LEARNING_EFFICIENCY.md's requirement for "Add a novelty guard" - before
launching an experiment, identify whether an equivalent experiment already exists.

The guard operates on experiment signatures derived from key characteristics:
- Signal class/type
- Lookback/horizon parameters  
- Data universe/tickers
- Core research question/hypothesis
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass

# Version tracking to detect when experiment definitions change
EXPERIMENT_DEFINITION_VERSION = "1.0"
@dataclass
class ExperimentSignature:
    """Unique identifier for an experiment based on its defining characteristics."""
    signal_type: str
    lookback: Optional[int] = None
    horizon: Optional[int] = None
    ticker_universe: Optional[str] = None
    research_question_hash: Optional[str] = None
    data_type: Optional[str] = None  # synthetic vs real data
    
    def to_key(self) -> str:
        """Generate deterministic string key for the experiment."""
        parts = []
        if self.signal_type:
            parts.append(f"type:{self.signal_type}")
        if self.lookback is not None:
            parts.append(f"lookback:{self.lookback}")
        if self.horizon is not None:
            parts.append(f"horizon:{self.horizon}")
        if self.ticker_universe:
            parts.append(f"universe:{self.ticker_universe}")
        if self.research_question_hash:
            parts.append(f"q:{self.research_question_hash}")
        if self.data_type:
            parts.append(f"data:{self.data_type}")
        return "|".join(sorted(parts))
@dataclass 
class ExperimentRegistry:
    """Persistent registry of completed/falsified experiments."""
    
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.experiments: Dict[str, List[ExperimentSignature]] = {}
        self._ensure_registry_file()
        self._load_registry()
    
    def _ensure_registry_file(self):
        """Ensure registry file exists."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({})
    
    def _load_registry(self):
        """Load registry from disk."""
        try:
            if self.registry_path.stat().st_size > 0:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                self.experiments = data.get("experiments", {})
                # Backwards compatibility: validate structure
                for key, sigs in self.experiments.items():
                    if isinstance(sigs, list):
                        self.experiments[key] = [ExperimentSignature(**s) if isinstance(s, dict) else s 
                                              for s in sigs]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Start fresh if registry is corrupted
            self.experiments = {}
    
    def _write_registry(self, data: Dict):
        """Write registry to disk atomically."""
        # Write to temp then rename for atomicity
        temp_path = self.registry_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        temp_path.replace(self.registry_path)
    
    def is_duplicate(self, experiment: ExperimentSignature) -> bool:
        """Check if experiment is equivalent to any completed/falsified experiment."""
        key = experiment.to_key()
        if key not in self.experiments:
            return False
        
        # Detailed comparison - ensure all significant characteristics match
        for completed in self.experiments[key]:
            if self._is_equivalent(completed, experiment):
                return True
        
        return False
    
    def _is_equivalent(self, completed: ExperimentSignature, candidate: ExperimentSignature) -> bool:
        """Detailed equivalence check between two experiment signatures."""
        # Core equivalence: signal type is primary discriminator
        if completed.signal_type != candidate.signal_type:
            return False
            
        # Lookback equivalence
        if (completed.lookback is None) != (candidate.lookback is None):
            return False
        if completed.lookback is not None and candidate.lookback is not None:
            if completed.lookback != candidate.lookback:
                return False
                
        # Horizon equivalence  
        if (completed.horizon is None) != (candidate.horizon is None):
            return False
        if completed.horizon is not None and candidate.horizon is not None:
            if completed.horizon != candidate.horizon:
                return False
                
        # Universe equivalence (more flexible - allow superset/subset matches)
        if completed.ticker_universe is not None and candidate.ticker_universe is not None:
            # Both have universe defined - check if they match or one is subset of the other
            comp_set = set(completed.ticker_universe.split(','))
            cand_set = set(candidate.ticker_universe.split(','))
            if not (comp_set.issubset(cand_set) or cand_set.issubset(comp_set)):
                return False
                
        # Data type can differ (synthetic vs real may be acceptable repeats)
        # Question hash helps detect equivalent research questions
        
        return True
    
    def register_experiment(self, experiment: ExperimentSignature, status: str = "FALSIFIED"):
        """Register a completed experiment in the registry."""
        key = experiment.to_key()
        
        # Ensure list exists
        if key not in self.experiments:
            self.experiments[key] = []
            
        # Add if not already registered (avoid duplicates in registry)
        if not any(self._is_equivalent(sig, experiment) for sig in self.experiments[key]):
            self.experiments[key].append(experiment)
            
            # Write to disk
            data = {"experiments": self.experiments, "version": EXPERIMENT_DEFINITION_VERSION}
            self._write_registry(data)
    
    def get_equivalence_summary(self) -> Dict:
        """Get summary of registered experiments for audit/logging."""
        return {
            "total_unique_experiments": len(self.experiments),
            "experiment_types": list(self.experiments.keys()),
            "version": EXPERIMENT_DEFINITION_VERSION
        }
# Convenience functions for common experiment types
def extract_signature_from_artifact(artifact_path: Path) -> Optional[ExperimentSignature]:
    """Extract experiment signature from a research artifact file."""
    try:
        with open(artifact_path, 'r') as f:
            artifact = json.load(f)
        
        # Heuristic extraction based on artifact structure
        # This is a simplified implementation - can be enhanced per artifact type
        
        # For momentum artifacts
        if "lookback" in artifact:
            return ExperimentSignature(
                signal_type="momentum",
                lookback=artifact.get("lookback"),
                data_type="real" if "dataset_id" in artifact and "yf-ohlcv" in artifact["dataset_id"] else "synthetic"
            )
        
        # Add more artifact type detection as needed
        
        return None
        
    except (json.JSONDecodeError, KeyError):
        return None
class NoveltyGuard:
    """Main novelty guard interface."""
    
    def __init__(self, registry_path: Path = Path("state/experiment_registry.json")):
        self.registry = ExperimentRegistry(registry_path)
    
    def check_novelty(self, experiment: ExperimentSignature, description: str = "") -> Tuple[bool, Optional[str]]:
        """Check if experiment is novel (not equivalent to completed work).
        
        Returns:
            Tuple of (is_novel, reason_if_not_novel)
        """
        if self.registry.is_duplicate(experiment):
            # Find the duplicate for detailed reason
            key = experiment.to_key()
            if key in self.registry.experiments:
                for completed in self.registry.experiments[key]:
                    if self._are_equivalent(experiment, completed):
                        reason = (f"Equivalent experiment already completed: {completed.to_key()}. "
                                 f"Status: completed/falsified. New experiment not novel.")
                        return False, reason
        
        return True, None
    
    def _are_equivalent(self, exp1: ExperimentSignature, exp2: ExperimentSignature) -> bool:
        """Check if two experiment signatures are equivalent."""
        return (exp1.signal_type == exp2.signal_type and
                exp1.lookback == exp2.lookback and
                exp1.horizon == exp2.horizon and
                exp1.ticker_universe == exp2.ticker_universe)
    
    def record_completed_experiment(self, experiment: ExperimentSignature, status: str = "FALSIFIED"):
        """Record a completed experiment in the registry."""
        self.registry.register_experiment(experiment, status)
    
    def audit_registry(self) -> Dict:
        """Get audit summary of the experiment registry."""
        return self.registry.get_equivalence_summary()

# Backward compatibility aliases
ExperimentSignature = ExperimentSignature
NoveltyGuard = NoveltyGuard
def detect_duplicate_experiments() -> Dict:
    """Convenience function for duplicate detection audit."""
    guard = NoveltyGuard()
    return guard.audit_registry()
if __name__ == "__main__":
    # Simple test/demo
    guard = NoveltyGuard()
    
    # Test momentum experiment
    momentum_exp = ExperimentSignature(
        signal_type="momentum",
        lookback=5,
        horizon=1,
        ticker_universe="AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,JNJ,XOM",
        data_type="real"
    )
    
    print("Testing novelty guard...")
    print(f"Experiment: {momentum_exp.to_key()}")
    
    # First registration should pass (novel)
    is_novel, reason = guard.check_novelty(momentum_exp)
    print(f"First check - Novel: {is_novel}, Reason: {reason}")
    
    if is_novel:
        guard.record_completed_experiment(momentum_exp, "FALSIFIED")
    
    # Second check should fail (duplicate)
    is_novel2, reason2 = guard.check_novelty(momentum_exp)
    print(f"Second check - Novel: {is_novel2}, Reason: {reason2}")
    
    print("\nRegistry audit:")
    print(json.dumps(guard.audit_registry(), indent=2))
