#!/usr/bin/env python3
"""
Test script for duplicate-avoidance check with actual momentum experiment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_duplicate_avoidance import (
    extract_frontier_cell_characteristics,
    check_duplicate_before_launch,
    _extract_key_characteristics_from_spec,
    _extract_signature_features_from_hypothesis
)


def main():
    print("=== Testing Duplicate-Avoidance Check with Actual Momentum Experiment ===\n")
    
    # Extract actual resolved frontier cells from STATE.md
    resolved_cells = extract_frontier_cell_characteristics()
    print(f"Extracted {len(resolved_cells)} resolved frontier cells from STATE.md")
    
    # Define the actual momentum experiment specification from agent_02.json
    momentum_experiment_spec = {
        'area_hypothesis': 'Momentum (long the previous 5-day return, hold 1 day)',
        'spec': {
            'signal_type': 'momentum',
            'data_universe': 'collected_10_assets', 
            'methodology': 'walk_forward_is_oos',
            'lookback_period': 5,
            'holding_period': 1,
            'task_id': 'R-001'
        }
    }
    
    print(f"\nTesting against momentum experiment (task_id: {momentum_experiment_spec['spec']['task_id']})")
    print(f"Area Hypothesis: {momentum_experiment_spec['area_hypothesis']}")
    print(f"Signal Type: {momentum_experiment_spec['spec']['signal_type']}")
    print(f"Data Universe: {momentum_experiment_spec['spec']['data_universe']}")
    print(f"Methodology: {momentum_experiment_spec['spec']['methodology']}")
    
    # Extract characteristics for display
    characteristics = _extract_key_characteristics_from_spec(str(momentum_experiment_spec))
    features = _extract_signature_features_from_hypothesis(momentum_experiment_spec['area_hypothesis'])
    
    print(f"\nExtracted characteristics:")
    print(f"  Signal Type: {characteristics['signal_type']}")
    print(f"  Data Universe: {characteristics['data_universe']}")
    print(f"  Methodology: {characteristics['methodology']}")
    print(f"  Signature Features: {features['signal_class']}")
    print(f"  Testing Gates: {features['testing_gates']}")
    
    # Check for duplicates
    duplicate_check = check_duplicate_before_launch(momentum_experiment_spec, resolved_cells)
    
    print(f"\n=== Duplicate Check Results ===")
    print(f"Total resolved cells: {len(resolved_cells)}")
    print(f"Exact duplicates found: {duplicate_check['duplicate_count']}")
    print(f"Near-duplicates found: {duplicate_check['near_duplicate_count']}")
    print(f"Launch allowed: {duplicate_check['launch_allowed']}")
    print(f"Launch recommended: {duplicate_check['launch_recommended']}")
    
    if duplicate_check['duplicates']:
        print(f"\n⚠️  EXACT DUPLICATES FOUND (should have prevented launch):")
        for i, dup in enumerate(duplicate_check['duplicates'], 1):
            print(f"  {i}. {dup['cell_id']}")
            print(f"     Status: {dup['status']}")
            print(f"     Reason: {dup['reason']}")
    
    if duplicate_check['near_duplicates']:
        print(f"\n⚠️  NEAR-DUPLICATES FOUND (high similarity):")
        for i, near_dup in enumerate(duplicate_check['near_duplicates'], 1):
            print(f"  {i}. {near_dup['cell_id']}")
            print(f"     Status: {near_dup['status']}")
            print(f"     Similarity Score: {near_dup['similarity_score']:.2f}")
            print(f"     Reason: {near_dup['reason']}")
    
    print(f"\n=== Analysis of Momentum Cell in Resolved Cells ===")
    
    # Find the momentum cell in resolved cells
    momentum_cell = None
    for cell in resolved_cells:
        if 'momentum' in cell['area_hypothesis'].lower():
            momentum_cell = cell
            break
    
    if momentum_cell:
        print(f"Found momentum cell in resolved frontier:")
        print(f"  Area Hypothesis: {momentum_cell['area_hypothesis']}")
        print(f"  Status: {momentum_cell['status']}")
        key_char = momentum_cell.get('key_characteristics', {})
        print(f"  Signal Type: {key_char.get('signal_type', 'Unknown')}")
        print(f"  Data Universe: {key_char.get('data_universe', 'Unknown')}")
        print(f"  Methodology: {key_char.get('methodology', 'Unknown')}")
        signature_features = momentum_cell.get('signature_features', {})
        print(f"  Signature Features: {signature_features.get('signal_class', 'Unknown')}")
        
        # Compare with new experiment
        print(f"\n=== Feature Comparison ===")
        print(f"New Experiment Characteristics:")
        print(f"  Signal Type Match: {characteristics['signal_type'] == key_char.get('signal_type', 'Unknown')}")
        print(f"  Data Universe Match: {characteristics['data_universe'] == key_char.get('data_universe', 'Unknown')}")
        print(f"  Methodology Match: {characteristics['methodology'] == key_char.get('methodology', 'Unknown')}")
        print(f"  Signature Features Match: {features['signal_class'] == signature_features.get('signal_class', 'Unknown')}")
        
        similarity = 0
        if characteristics['signal_type'] == key_char.get('signal_type', 'Unknown'):
            similarity += 0.4
        if characteristics['data_universe'] == key_char.get('data_universe', 'Unknown'):
            similarity += 0.3
        if characteristics['methodology'] == key_char.get('methodology', 'Unknown'):
            similarity += 0.2
        if features['signal_class'] == signature_features.get('signal_class', 'Unknown'):
            similarity += 0.1
        
        print(f"  Overall Similarity Score: {similarity:.2f}")
    
    print(f"\n=== Conclusion ===")
    
    if momentum_cell and duplicate_check['duplicate_count'] > 0:
        print(f"❌ PROBLEM: Momentum experiment specification MATCHES resolved frontier cell!")
        print(f"   This indicates the duplicate-avoidance check failed or the momentum experiment should NOT have been launched.")
        print(f"   Evidence: The resolved momentum cell should have been detected as an exact duplicate.")
    elif duplicate_check['launch_recommended']:
        print(f"✅ PASS: No exact duplicates found. Launch recommended.")
        print(f"   The duplicate-avoidance check correctly identified this as a novel experiment.")
    else:
        print(f"⚠️  WARNING: Near-duplicates found but no exact duplicates.")
        print(f"   Consider whether this represents sufficient new information to justify launch.")
    
    print(f"\n=== Controller Integration Note ===")
    print(f"This test demonstrates the proposed duplicate-avoidance check.")
    print(f"For production, this check should run BEFORE experiment launch to prevent equivalent re-testing.")
    
    return duplicate_check


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result['launch_recommended'] else 1)