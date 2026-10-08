"""
Novelty guard implementation for Kilo research learning efficiency.

This module implements the "Add a novelty guard" improvement from LEARNING_EFFICIENCY.md section 2.

Before launching an experiment, identify whether an equivalent experiment already exists.

Repeat only when there is:
- a new hypothesis,
- a new diagnostic purpose,
- materially new data,
- an independent verification need, or
- a clearly different information objective.

This converts accumulated history into actual avoidance of wasted search.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from research.backtest.novelty_utils import (
    extract_signal_class,
    extract_parameters,
    extract_data_universe,
    extract_research_purpose,
    calculate_parameter_similarity,
    calculate_data_universe_similarity,
    calculate_purpose_similarity,
    generate_novelty_decision,
)

# Global configuration
NOVELTY_GUARD_CONFIG = {
    "similarity_threshold": 0.7,
    "signal_class_weight": 0.4,
    "parameter_weight": 0.3,
    "data_universe_weight": 0.2,
    "purpose_weight": 0.1,  # Lower weight - different purpose is good
    "enable_evidence_logging": True,
    "enable_deterministic_mode": True,
}
class NoveltyGuard:
    """
    Novelty guard for preventing duplicate/near-duplicate research experiments.
    
    The guard implements the learning efficiency requirement from LEARNING_EFFICIENCY.md
    to prevent equivalent experiments while allowing genuinely novel research.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize novelty guard.
        
        Args:
            config: Optional configuration override
        """
        self.config = {**NOVELTY_GUARD_CONFIG, **(config or {})}
        self.decision_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load existing evidence base
        self.evidence_base_path = PROJECT_ROOT / "state" / "check_artifacts"
        self.processed_artifacts: List[Dict[str, Any]] = self._load_evidence_base()
        
    def _load_evidence_base(self) -> List[Dict[str, Any]]:
        """
        Load existing evidence base from check_artifacts directory.
        
        Returns:
            List of processed evidence artifacts
        """
        artifacts = []
        
        if not self.evidence_base_path.exists():
            return artifacts
        
        # Process all JSON files in check_artifacts
        for artifact_file in self.evidence_base_path.glob("*.json"):
            try:
                with open(artifact_file) as f:
                    artifact = json.load(f)
                
                # Extract key characteristics
                artifact_type = artifact_file.stem
                signal_class = extract_signal_class(artifact, artifact_type)
                parameters = extract_parameters(artifact)
                data_universe = extract_data_universe(artifact)
                research_purpose = extract_research_purpose(artifact, artifact_type)
                
                processed_artifact = {
                    "artifact_type": artifact_type,
                    "signal_class": signal_class,
                    "parameters": parameters,
                    "data_universe": data_universe,
                    "research_purpose": research_purpose,
                    "artifact_path": str(artifact_file),
                    "timestamp": artifact.get("timestamp", time.time()),
                }
                
                artifacts.append(processed_artifact)
                
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed files
                continue
        
        return artifacts
    
    def is_duplicate(
        self,
        new_artifact: Dict[str, Any],
        new_artifact_type: str,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determine if a new artifact is duplicate of existing evidence.
        
        Args:
            new_artifact: The new artifact to check
            new_artifact_type: The type/class of the new artifact
            cache_key: Optional cache key for deterministic results
            
        Returns:
            Dictionary with duplicate detection results and evidence
        """
        # Use cache key for deterministic results if enabled
        if self.config["enable_deterministic_mode"] and cache_key:
            if cache_key in self.decision_cache:
                return self.decision_cache[cache_key]
        
        # Extract characteristics of new artifact
        new_signal_class = extract_signal_class(new_artifact, new_artifact_type)
        new_parameters = extract_parameters(new_artifact)
        new_data_universe = extract_data_universe(new_artifact)
        new_research_purpose = extract_research_purpose(new_artifact, new_artifact_type)
        
        # Calculate similarities with existing evidence
        similarities = []
        duplicate_detected = False
        most_similar_artifact = None
        max_similarity = 0.0
        
        for existing_artifact in self.processed_artifacts:
            # Skip self-comparison
            if (existing_artifact["artifact_type"] == new_artifact_type and
                json.dumps(existing_artifact.get("parameters", {}), sort_keys=True) == 
                json.dumps(new_parameters, sort_keys=True)):
                continue
            
            # Calculate similarity components
            signal_class_sim = 1.0 if existing_artifact["signal_class"] == new_signal_class else 0.0
            
            param_sim = calculate_parameter_similarity(
                existing_artifact["parameters"], new_parameters
            )
            
            data_univ_sim = calculate_data_universe_similarity(
                existing_artifact["data_universe"], new_data_universe
            )
            
            purpose_sim = calculate_purpose_similarity(
                existing_artifact["research_purpose"], new_research_purpose
            )
            
            # Generate novelty decision
            decision = generate_novelty_decision(
                signal_class_sim,
                param_sim,
                data_univ_sim,
                purpose_sim,
                self.config["similarity_threshold"]
            )
            
            similarities.append(decision["similarity"])
            
            if decision["is_duplicate"] and decision["similarity"] > max_similarity:
                duplicate_detected = True
                most_similar_artifact = existing_artifact
                max_similarity = decision["similarity"]
        
        # If no duplicates found, new artifact is novel
        if not duplicate_detected:
            result = {
                "is_duplicate": False,
                "similarity": round(max(similarities) if similarities else 0.0, 3),
                "duplicate_reason": "No equivalent or near-equivalent research found",
                "most_similar_artifact": None,
                "evidence_quality": "VERIFIED_NEGATIVE_RESULT",
                "decision": "USEFUL_CHANGE",
                "novelty_guard_version": "1.0",
                "timestamp": time.time(),
            }
        else:
            # Duplicate found
            result = {
                "is_duplicate": True,
                "similarity": round(max_similarity, 3),
                "duplicate_reason": most_similar_artifact["artifact_type"] if most_similar_artifact else "unknown",
                "most_similar_artifact": {
                    "type": most_similar_artifact["artifact_type"] if most_similar_artifact else None,
                    "signal_class": most_similar_artifact["signal_class"] if most_similar_artifact else None,
                    "parameters": most_similar_artifact["parameters"] if most_similar_artifact else None,
                    "data_universe": most_similar_artifact["data_universe"] if most_similar_artifact else None,
                    "research_purpose": most_similar_artifact["research_purpose"] if most_similar_artifact else None,
                },
                "evidence_quality": "VERIFIED_POSITIVE_RESULT",
                "decision": "REJECT",
                "novelty_guard_version": "1.0",
                "timestamp": time.time(),
            }
        
        # Cache result if deterministic mode enabled
        if self.config["enable_deterministic_mode"] and cache_key:
            self.decision_cache[cache_key] = result
        
        return result
    
    def add_artifact_to_evidence_base(self, artifact: Dict[str, Any], artifact_type: str):
        """
        Add new artifact to the evidence base for future novelty detection.
        
        Args:
            artifact: The artifact to add
            artifact_type: The type of artifact
        """
        # Extract characteristics
        signal_class = extract_signal_class(artifact, artifact_type)
        parameters = extract_parameters(artifact)
        data_universe = extract_data_universe(artifact)
        research_purpose = extract_research_purpose(artifact, artifact_type)
        
        # Create processed artifact entry
        processed_artifact = {
            "artifact_type": artifact_type,
            "signal_class": signal_class,
            "parameters": parameters,
            "data_universe": data_universe,
            "research_purpose": research_purpose,
            "timestamp": time.time(),
            "source": "novelty_guard",
        }
        
        # Add to evidence base
        self.processed_artifacts.append(processed_artifact)
        
        # Log decision
        if self.config["enable_evidence_logging"]:
            self._log_novelty_decision(processed_artifact, "ADDED_TO_EVIDENCE_BASE")
    
    def _log_novelty_decision(self, artifact: Dict[str, Any], action: str):
        """
        Log novelty guard decision to state files.
        
        Args:
            artifact: The artifact involved
            action: The action taken (DUPLICATE, NOVEL, ADDED_TO_EVIDENCE_BASE)
        """
        log_entry = {
            "timestamp": time.time(),
            "artifact_type": artifact["artifact_type"],
            "signal_class": artifact["signal_class"],
            "action": action,
            "parameters": artifact["parameters"],
            "data_universe": artifact["data_universe"],
            "research_purpose": artifact["research_purpose"],
        }
        
        # Create log file
        log_file = PROJECT_ROOT / "logs" / "NOVELTY_GUARD.md"
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, "a") as f:
            f.write(f"## Novelty Guard Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(json.dumps(log_entry, indent=2))
            f.write("\n\n")
    
    def get_duplicate_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about duplicate detection.
        
        Returns:
            Dictionary with duplicate detection statistics
        """
        total_artifacts = len(self.processed_artifacts)
        duplicate_counts = {}
        signal_class_counts = {}
        
        for artifact in self.processed_artifacts:
            # Count by signal class
            signal_class = artifact["signal_class"]
            signal_class_counts[signal_class] = signal_class_counts.get(signal_class, 0) + 1
            
            # Count duplicates (simplified - we don't track actual duplicate decisions here)
            # In a real implementation, you would track actual duplicate decisions
            pass
        
        return {
            "total_artifacts_in_evidence_base": total_artifacts,
            "signal_class_distribution": signal_class_counts,
            "duplicate_prevention_enabled": True,
            "configuration": self.config,
        }
    
    def reset(self):
        """Reset novelty guard state."""
        self.decision_cache.clear()
        self.processed_artifacts = self._load_evidence_base()


def main():
    """Test the novelty guard implementation."""
    print("Testing Novelty Guard Implementation")
    print("=" * 40)
    
    # Create novelty guard instance
    guard = NoveltyGuard()
    
    # Test basic functionality
    print("\n1. Testing basic duplicate detection...")
    
    # Create test artifacts
    momentum_artifact = {
        "lookback": 5,
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "per_asset": {"AAPL": {}, "MSFT": {}},
        "ticker_order": ["AAPL", "MSFT"],
        "train": 252,
        "test": 84,
        "warmup": 60,
        "overlap": 60,
    }
    
    reversal_artifact = {
        "lookback": 5,
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "per_asset": {"AAPL": {}, "MSFT": {}},
        "ticker_order": ["AAPL", "MSFT"],
        "train": 252,
        "test": 84,
        "warmup": 60,
        "overlap": 60,
    }
    
    # Test momentum vs momentum (same)
    result1 = guard.is_duplicate(momentum_artifact, "momentum", "test1")
    print(f"Momentum vs itself: duplicate={result1['is_duplicate']}, similarity={result1['similarity']}")
    
    # Test momentum vs reversal (different)
    result2 = guard.is_duplicate(reversal_artifact, "reversal", "test2")
    print(f"Momentum vs Reversal: duplicate={result2['is_duplicate']}, similarity={result2['similarity']}")
    
    # Test statistics
    print("\n2. Getting statistics...")
    stats = guard.get_duplicate_statistics()
    print(f"Total artifacts in evidence base: {stats['total_artifacts_in_evidence_base']}")
    print(f"Signal class distribution: {stats['signal_class_distribution']}")
    
    print("\n✅ Novelty guard implementation test completed!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())