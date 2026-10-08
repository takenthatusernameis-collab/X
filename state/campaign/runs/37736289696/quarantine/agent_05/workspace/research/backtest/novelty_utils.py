"""
Novelty guard utility functions.

This module provides helper functions for the novelty guard implementation,
including similarity calculation, evidence extraction, and duplicate detection.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def extract_signal_class(artifact: Dict[str, Any], artifact_type: str) -> str:
    """
    Extract signal class from artifact.
    
    Args:
        artifact: The artifact dictionary
        artifact_type: The type of artifact (momentum, reversal, etc.)
        
    Returns:
        The signal class
    """
    # Map artifact types to signal classes
    signal_class_map = {
        "momentum": "momentum",
        "momentum_hold_period": "momentum",
        "momentum_sweep": "momentum",
        "mean_reversion": "reversal",
        "cross_sectional_momentum": "cross_sectional",
        "cross_sectional_regime_diagnostic": "cross_sectional",
        "cross_sectional_relative_strength": "cross_sectional",
        "volatility_targeting": "volatility_targeting",
        "breakout_20day_cont": "breakout",
        "regime_adaptive_ma": "regime_adaptive",
    }
    
    # Try to extract from artifact content
    if artifact_type in signal_class_map:
        return signal_class_map[artifact_type]
    
    # Fallback: try to infer from artifact content
    content = json.dumps(artifact).lower()
    
    if "momentum" in content:
        return "momentum"
    elif "reversal" in content or "mean_reversion" in content:
        return "reversal"
    elif "cross_sectional" in content:
        return "cross_sectional"
    elif "volatility_targeting" in content or "volatility" in content:
        return "volatility_targeting"
    elif "breakout" in content:
        return "breakout"
    elif "regime_adaptive" in content or "regime" in content:
        return "regime_adaptive"
    
    # Default
    return "unknown"
def extract_parameters(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters from artifact.
    
    Args:
        artifact: The artifact dictionary
        
    Returns:
        Dictionary of extracted parameters
    """
    parameters = {}
    
    # Look for common parameter patterns
    artifact_content = json.dumps(artifact).lower()
    
    # Look for lookback/hold periods
    lookback_match = re.search(r'"lookback"\s*:\s*(\d+)', artifact_content)
    if lookback_match:
        parameters['lookback'] = int(lookback_match.group(1))
    
    hold_match = re.search(r'"hold"\s*:\s*(\d+\.?\d*)', artifact_content)
    if hold_match:
        parameters['hold'] = float(hold_match.group(1))
    
    # Look for other common parameters
    top_k_match = re.search(r'"top_k"\s*:\s*(\d+)', artifact_content)
    if top_k_match:
        parameters['top_k'] = int(top_k_match.group(1))
    
    bottom_k_match = re.search(r'"bottom_k"\s*:\s*(\d+)', artifact_content)
    if bottom_k_match:
        parameters['bottom_k'] = int(bottom_k_match.group(1))
    
    # Extract from artifact structure
    if 'lookback' in artifact:
        parameters['lookback'] = artifact['lookback']
    if 'hold_periods' in artifact:
        parameters['hold_periods'] = artifact['hold_periods']
    if 'top_k' in artifact:
        parameters['top_k'] = artifact['top_k']
    if 'bottom_k' in artifact:
        parameters['bottom_k'] = artifact['bottom_k']
    
    return parameters
def extract_data_universe(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data universe information from artifact.
    
    Args:
        artifact: The artifact dictionary
        
    Returns:
        Dictionary of data universe characteristics
    """
    universe = {}
    
    # Extract from artifact structure
    if 'dataset_id' in artifact:
        universe['dataset'] = artifact['dataset_id']
    
    if 'universe' in artifact:
        universe['description'] = artifact['universe']
    
    # Extract universe size from structure
    if 'per_asset' in artifact:
        universe['assets_count'] = len(artifact['per_asset'])
    
    if 'ticker_order' in artifact:
        universe['tickers'] = artifact['ticker_order']
    
    # Extract time periods
    if 'train' in artifact and 'test' in artifact:
        universe['time_period'] = f"{artifact['train']}/{artifact['test']}"
    
    return universe
def calculate_parameter_similarity(params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two parameter sets.
    
    Args:
        params1: First parameter set
        params2: Second parameter set
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not params1 and not params2:
        return 1.0  # Both empty - identical
    
    if not params1 or not params2:
        return 0.0  # One empty - no similarity
    
    similarity_score = 0.0
    total_weight = 0.0
    
    # Compare common keys
    common_keys = set(params1.keys()) & set(params2.keys())
    
    for key in common_keys:
        val1 = params1[key]
        val2 = params2[key]
        
        if key in ['lookback', 'hold', 'top_k', 'bottom_k']:
            # Numeric parameters - compare with tolerance
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 == val2:
                    weight = 1.0
                elif abs(val1 - val2) <= 1:  # Allow small differences
                    weight = 0.5
                else:
                    weight = 0.0
            else:
                weight = 0.0
        elif key in ['hold_periods', 'lookbacks']:
            # Sequence parameters - check overlap
            if isinstance(val1, list) and isinstance(val2, list):
                overlap = len(set(val1) & set(val2))
                if overlap > 0:
                    weight = overlap / max(len(val1), len(val2))
                else:
                    weight = 0.0
            else:
                weight = 0.0
        else:
            # Other parameters - exact match only
            weight = 1.0 if val1 == val2 else 0.0
        
        similarity_score += weight
        total_weight += 1.0
    
    if total_weight == 0:
        return 0.0
    
    return similarity_score / total_weight
def calculate_data_universe_similarity(univ1: Dict[str, Any], univ2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two data universes.
    
    Args:
        univ1: First data universe
        univ2: Second data universe
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not univ1 and not univ2:
        return 1.0  # Both empty - identical
    
    if not univ1 or not univ2:
        return 0.0  # One empty - no similarity
    
    similarity_score = 0.0
    total_weight = 0.0
    
    # Compare dataset IDs
    if univ1.get('dataset') == univ2.get('dataset'):
        similarity_score += 1.0
        total_weight += 1.0
    
    # Compare asset counts
    count1 = univ1.get('assets_count', 0)
    count2 = univ2.get('assets_count', 0)
    
    if count1 == count2 and count1 > 0:
        similarity_score += 1.0
        total_weight += 1.0
    elif count1 > 0 and count2 > 0:
        # Some similarity based on size ratio
        ratio = min(count1, count2) / max(count1, count2)
        similarity_score += ratio
        total_weight += 1.0
    
    # Compare time periods
    time1 = univ1.get('time_period', '')
    time2 = univ2.get('time_period', '')
    
    if time1 == time2 and time1:
        similarity_score += 1.0
        total_weight += 1.0
    
    if total_weight == 0:
        return 0.0
    
    return similarity_score / total_weight
def extract_research_purpose(artifact: Dict[str, Any], artifact_type: str) -> str:
    """
    Extract research purpose from artifact.
    
    Args:
        artifact: The artifact dictionary
        artifact_type: The type of artifact
        
    Returns:
        Research purpose classification
    """
    # Map artifact types to research purposes
    purpose_map = {
        "momentum": "hypothesis_testing",
        "momentum_hold_period": "robustness_testing",
        "momentum_sweep": "robustness_testing",
        "mean_reversion": "hypothesis_falsification",
        "cross_sectional_momentum": "hypothesis_testing",
        "cross_sectional_regime_diagnostic": "regime_analysis",
        "cross_sectional_relative_strength": "hypothesis_falsification",
        "volatility_targeting": "hypothesis_falsification",
        "breakout_20day_cont": "signal_validation",
        "regime_adaptive_ma": "robustness_testing",
    }
    
    if artifact_type in purpose_map:
        return purpose_map[artifact_type]
    
    # Try to infer from artifact content
    content = json.dumps(artifact).lower()
    
    if "sensitivity" in content or "robustness" in content:
        return "robustness_testing"
    elif "falsified" in content or "negative" in content:
        return "hypothesis_falsification"
    elif "supported" in content or "positive" in content:
        return "hypothesis_validation"
    elif "validate" in content:
        return "validation"
    elif "test" in content:
        return "hypothesis_testing"
    
    return "unknown"
def calculate_purpose_similarity(purpose1: str, purpose2: str) -> float:
    """
    Calculate similarity between research purposes.
    
    Args:
        purpose1: First research purpose
        purpose2: Second research purpose
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if purpose1 == purpose2:
        return 1.0
    
    # Different but related purposes
    related_pairs = [
        ("hypothesis_testing", "hypothesis_falsification"),
        ("hypothesis_validation", "hypothesis_testing"),
        ("robustness_testing", "regime_analysis"),
        ("validation", "hypothesis_testing"),
    ]
    
    # Check if purposes are related
    for p1, p2 in related_pairs:
        if (purpose1 == p1 and purpose2 == p2) or (purpose1 == p2 and purpose2 == p1):
            return 0.7
    
    # Unrelated purposes
    return 0.0
def generate_novelty_decision(
    signal_class_match: float,
    parameter_similarity: float,
    data_universe_similarity: float,
    purpose_similarity: float,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Generate novelty guard decision based on similarity scores.
    
    Args:
        signal_class_match: Signal class match score
        parameter_similarity: Parameter similarity score
        data_universe_similarity: Data universe similarity score
        purpose_similarity: Research purpose similarity score
        threshold: Duplicate threshold (default: 0.7)
        
    Returns:
        Dictionary with decision and evidence
    """
    # Calculate weighted similarity
    # Signal class is most important, purpose is least important
    weighted_similarity = (
        signal_class_match * 0.4 +
        parameter_similarity * 0.3 +
        data_universe_similarity * 0.2 +
        (1.0 - purpose_similarity) * 0.1  # Subtract purpose similarity (different = better)
    )
    
    is_duplicate = weighted_similarity >= threshold
    
    # Determine decision reason
    if is_duplicate:
        decision_reason = "equivalent or near-equivalent research request"
        if weighted_similarity >= 0.9:
            decision_reason = "virtually identical research request"
        elif weighted_similarity >= 0.8:
            decision_reason = "highly similar research request"
    else:
        decision_reason = "genuinely novel research request"
        if weighted_similarity <= 0.3:
            decision_reason = "completely different research request"
        elif weighted_similarity <= 0.5:
            decision_reason = "distinctly different research request"
    
    return {
        "is_duplicate": is_duplicate,
        "similarity": round(weighted_similarity, 3),
        "signal_class_match": signal_class_match,
        "parameter_overlap": parameter_similarity,
        "data_universe_match": data_universe_similarity,
        "purpose_similarity": purpose_similarity,
        "decision_reason": decision_reason,
        "threshold_used": threshold,
    }
def validate_novelty_guard_config():
    """
    Validate novelty guard configuration and thresholds.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    # Test configuration consistency
    test_scenarios = [
        # (signal_match, param_sim, data_sim, purpose_sim, expected_duplicate)
        (1.0, 1.0, 1.0, 1.0, True),    # Identical in all aspects
        (1.0, 1.0, 1.0, 0.0, True),    # Different purpose, identical otherwise
        (1.0, 1.0, 0.0, 1.0, True),    # Different data universe
        (1.0, 0.0, 1.0, 1.0, True),    # Different parameters
        (0.0, 1.0, 1.0, 1.0, False),   # Different signal class
        (0.5, 0.5, 0.5, 0.5, False),   # All moderately different
    ]
    
    threshold = 0.7
    
    for i, (sc, ps, ds, purpose_sim, expected) in enumerate(test_scenarios):
        decision = generate_novelty_decision(sc, ps, ds, purpose_sim, threshold)
        actual = decision['is_duplicate']
        
        if actual != expected:
            print(f"❌ Configuration test {i+1} failed:")
            print(f"   Input: signal_match={sc}, param_sim={ps}, data_sim={ds}, purpose_sim={purpose_sim}")
            print(f"   Expected: {expected}, Got: {actual}")
            print(f"   Decision: {decision['decision_reason']}")
            return False
    
    return True
def main():
    """Run novelty guard utility tests."""
    print("Novelty Guard Utility Functions Test")
    print("=" * 40)
    
    # Test configuration
    print("Testing configuration...")
    if validate_novelty_guard_config():
        print("✓ Configuration validation passed\n")
    else:
        print("✗ Configuration validation failed\n")
        return 1
    
    # Test function examples
    print("Testing utility functions...")
    
    # Test signal class extraction
    test_artifact = {
        "lookback": 5,
        "dataset_id": "test_dataset",
        "per_asset": {"AAPL": {}, "MSFT": {}}
    }
    
    signal_class = extract_signal_class(test_artifact, "momentum")
    print(f"✓ Signal class extraction: {signal_class}")
    
    # Test parameter extraction
    parameters = extract_parameters(test_artifact)
    print(f"✓ Parameter extraction: {parameters}")
    
    # Test data universe extraction
    universe = extract_data_universe(test_artifact)
    print(f"✓ Data universe extraction: {universe}")
    
    # Test parameter similarity
    params1 = {"lookback": 5, "hold": 1}
    params2 = {"lookback": 5, "hold": 2}
    param_sim = calculate_parameter_similarity(params1, params2)
    print(f"✓ Parameter similarity: {param_sim:.2f}")
    
    # Test data universe similarity
    univ1 = {"dataset": "dataset1", "assets_count": 10}
    univ2 = {"dataset": "dataset1", "assets_count": 10}
    univ_sim = calculate_data_universe_similarity(univ1, univ2)
    print(f"✓ Data universe similarity: {univ_sim:.2f}")
    
    # Test research purpose extraction
    purpose = extract_research_purpose(test_artifact, "momentum")
    print(f"✓ Research purpose extraction: {purpose}")
    
    # Test purpose similarity
    purpose1 = "hypothesis_testing"
    purpose2 = "hypothesis_falsification"
    purpose_sim = calculate_purpose_similarity(purpose1, purpose2)
    print(f"✓ Purpose similarity: {purpose_sim:.2f}")
    
    # Test novelty decision generation
    decision = generate_novelty_decision(1.0, 0.8, 1.0, 1.0)
    print(f"✓ Novelty decision: duplicate={decision['is_duplicate']}, similarity={decision['similarity']:.2f}")
    
    print("\n✅ All novelty guard utility function tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())