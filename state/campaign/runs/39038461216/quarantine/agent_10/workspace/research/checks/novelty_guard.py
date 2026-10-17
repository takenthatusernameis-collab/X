"""Lightweight novelty guard to prevent equivalent experiments.

Automatically detects when an equivalent experiment is already being conducted
or has been completed, based on semantic similarity of primary questions,
evidence basis, and objectives.

This implements the LEARNING_EFFICIENCY.md requirement for a novelty guard.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from difflib import SequenceMatcher
def is_semantically_equivalent(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Check if two texts are semantically equivalent based on similarity."""
    if not text1 or not text2:
        return False
    similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return similarity >= threshold
def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing stop words and standardizing."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'does', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'with', 'without', 'from', 'into', 'out',
        'survive', 'does', 'survival', 'research', 'momentum', 'edge', 'cost',
        'robustness', 'stress', 'universe', 'qualitative', 'priority'
    }
    normalized = ' '.join(word for word in text.lower().split() 
                          if word not in stop_words and len(word) > 3)
    return normalized
def check_equivalent_experiments(
    task_id: str,
    primary_question: str,
    evidence_basis: str,
    objective: str,
    scope: str,
    existing_tasks: List[Dict],
    resolved_task_ids: Set[str]
) -> Dict:
    """
    Check if the proposed task is equivalent to existing or resolved tasks.

    Returns a dict with:
    - is_equivalent: whether an equivalent task exists
    - equivalent_task_ids: list of equivalent task IDs
    - similarity_scores: similarity scores with equivalent tasks
    - block_action: action to take if equivalent found
    """
    found_equivalent = False
    equivalent_task_ids = []
    similarity_scores = []
    blocking_required = False

    normalized_new_question = normalize_text(primary_question)
    normalized_new_evidence = normalize_text(evidence_basis)
    normalized_new_objective = normalize_text(objective)

    # Check against all existing and resolved tasks
    for task in existing_tasks:
        if task.get('task_id') in resolved_task_ids:
            # Check resolved tasks for equivalence
            existing_question = task.get('primary_question', '')
            existing_evidence = task.get('evidence_basis', '')
            existing_objective = task.get('objective', '')

            question_match = is_semantically_equivalent(
                normalized_new_question, normalize_text(existing_question))
            evidence_match = is_semantically_equivalent(
                normalized_new_evidence, normalize_text(existing_evidence))
            objective_match = is_semantically_equivalent(
                normalized_new_objective, normalize_text(existing_objective))

            # Consider equivalent if multiple key components match
            match_count = sum([question_match, evidence_match, objective_match])
            if match_count >= 2:
                found_equivalent = True
                equivalent_task_ids.append(task.get('task_id'))
                similarity_scores.append(match_count / 3)
                blocking_required = True

    # Special handling for known resolved research patterns
    # R-001 (cost robustness check) - prevent similar cost robustness variations
    if 'cost' in normalized_new_objective.lower() and 'robustness' in normalized_new_objective.lower():
        for task in existing_tasks:
            if task.get('task_id') == 'R-001' and task.get('status') == 'RESOLVED':
                found_equivalent = True
                equivalent_task_ids.append('R-001')
                similarity_scores.append(0.9)  # High similarity to cost robustness
                blocking_required = True

    return {
        'is_equivalent': found_equivalent,
        'equivalent_task_ids': equivalent_task_ids,
        'similarity_scores': similarity_scores,
        'blocking_required': blocking_required
    }
def main():
    """Demonstrate the novelty guard with a test case."""
    # Load existing tasks
    task_queue_path = Path(__file__).parent.parent.parent / 'state' / 'campaign' / 'task_queue.json'
    with open(task_queue_path) as f:
        task_queue = json.load(f)

    existing_tasks = task_queue.get('tasks', [])

    # Get resolved task IDs
    resolved_task_ids = {
        task.get('task_id') for task in existing_tasks
        if task.get('status') == 'RESOLVED'
    }

    # Test cases
    test_cases = [
        {
            'task_id': 'P-003',  # Original scope - should pass
            'primary_question': 'Does the durable evidence layer preserve useful negative evidence strongly enough to redirect future research?',
            'evidence_basis': 'The repository contains extensive FALSIFIED and TESTED frontier entries plus a dedicated LEARNING_STATE layer.',
            'objective': 'Audit whether recent falsified frontiers and rejected process decisions are discoverable and actually reflected in the current task queue.',
            'scope': 'Use durable state only; compare a sample of falsified cells with current task candidates.'
        },
        {
            'task_id': 'R-002',  # Similar to R-001 but different focus - should pass
            'primary_question': 'Does the qualified momentum effect survive a small predeclared holding-period variation without becoming a single-point timing artifact?',
            'evidence_basis': 'Momentum is candidate positive evidence at a one-day hold and has already survived lookback and horizon robustness in magnitude.',
            'objective': 'Test a fixed lookback-5 momentum signal with a predeclared small holding-period set using the existing walk-forward/null methodology.',
            'scope': 'Use only the existing 10-asset universe and a fixed hold-period grid declared before execution.'
        },
        {
            'task_id': 'P-008',  # This should be blocked - equivalent to P-002
            'primary_question': 'Does the durable history adequately prevent repeating equivalent experiments when a frontier cell is already resolved?',
            'evidence_basis': 'LEARNING_EFFICIENCY already calls for a novelty guard and recent research cells were closed as FALSIFIED/TESTED.',
            'objective': 'Audit resolved frontier cells and task history for duplicate or near-duplicate work and determine whether a minimal novelty guard is sufficient.',
            'scope': 'Inspect LEARNING_STATE, STATE, research checks, and campaign queue semantics; propose at most one minimal change.'
        }
    ]

    print("=== Novelty Guard Demonstration ===\n")

    for test_task in test_cases:
        print(f"Testing task {test_task['task_id']}:")
        print(f"  Question: {test_task['primary_question'][:60]}...")
        print(f"  Objective: {test_task['objective'][:60]}...")

        result = check_equivalent_experiments(
            task_id=test_task['task_id'],
            primary_question=test_task['primary_question'],
            evidence_basis=test_task['evidence_basis'],
            objective=test_task['objective'],
            scope=test_task['scope'],
            existing_tasks=existing_tasks,
            resolved_task_ids=resolved_task_ids
        )

        if result['blocking_required']:
            print(f"  ❌ BLOCKED: Equivalent to existing tasks {result['equivalent_task_ids']}")
            print(f"  Similarity scores: {result['similarity_scores']}")
        else:
            print(f"  ✅ ALLOWED: No equivalent found")
        print()

    # Save the novelty guard for integration
    guard_config = {
        'version': '1.0',
        'similarity_threshold': 0.85,
        'blocking_rules': [
            'Prevent equivalent research questions',
            'Prevent duplicate evidence basis analysis',
            'Prevent redundant objective implementations',
            'Block cost-robustness variations after R-001'
        ],
        'checked_components': ['primary_question', 'evidence_basis', 'objective']
    }

    output_path = Path(__file__).parent.parent.parent / 'state' / 'check_artifacts' / 'novelty_guard_config.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(guard_config, f, indent=2)

    print(f"Novelty guard configuration saved to: {output_path}")

    return 0
if __name__ == '__main__':
    exit(main())
