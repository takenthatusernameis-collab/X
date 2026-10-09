#!/usr/bin/env python3
"""Minimal novelty guard implementation for Kilo research campaign.

Prevents equivalent experiments from being launched when a frontier cell is already resolved.

Based on LEARNING_EFFICIENCY.md: Before launching an experiment, identify whether
an equivalent experiment already exists. Repeat only when there is:
- a new hypothesis,
- a new diagnostic purpose,
- materially new data,
- an independent verification need, or
- a clearly different information objective.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any
from datetime import datetime

# Paths
STATE_DIR = Path("/home/runner/work/X/X/state/campaign")
TASK_QUEUE_FILE = STATE_DIR / "task_queue.json"
CURRENT_TASK_FILE = STATE_DIR / "current_task.json"
RUN_ID = "37861609752"


def extract_task_signature(task: Dict) -> Dict[str, Any]:
    """Extract a canonical signature for task equivalence checking."""
    return {
        "primary_question": task.get("primary_question", ""),
        "scope": task.get("scope", ""),
        "role": task.get("role", ""),
        "objective": task.get("objective", ""),
        "tests_preceding_process": task.get("tests_preceding_process", False)
    }


def load_task_queue() -> Dict:
    """Load the current task queue."""
    with open(TASK_QUEUE_FILE, 'r') as f:
        return json.load(f)


def load_current_task() -> Dict:
    """Load the current task being selected."""
    with open(CURRENT_TASK_FILE, 'r') as f:
        return json.load(f)


def check_novelty_guard(task_signature: Dict, resolved_task_signatures: List[Dict]) -> bool:
    """Check if a task represents new work or equivalent to resolved work.
    
    Returns True if novel (can proceed), False if equivalent (should be rejected).
    """
    # Simple equivalence check: if primary question and scope are identical
    # and tests_preceding_process is same, consider equivalent
    for resolved_sig in resolved_task_signatures:
        if (task_signature["primary_question"] == resolved_sig["primary_question"] and
            task_signature["scope"] == resolved_sig["scope"] and
            task_signature["tests_preceding_process"] == resolved_sig["tests_preceding_process"]):
            return False  # Equivalent to resolved work
    
    # If we can't determine equivalence, assume novel (proceed)
    return True


def main():
    """Main novelty guard implementation."""
    print("=== Novelty Guard Implementation ===")
    
    # Load task queue and current task
    queue_data = load_task_queue()
    current_task_data = load_current_task()
    
    current_task = current_task_data.get("preceding_agent", current_task_data)
    current_task_id = current_task.get("task_id", "")
    
    # Get resolved frontier cells from recent agent_01.json evidence
    resolved_frontier_cells = [
        # From agent_01 verification: MA crossover -> reversal -> CSRS -> volatility targeting -> momentum
        {
            "primary_question": "Does the current focused-task selector improve learning efficiency relative to the repository's prior opportunistic task choice?",
            "scope": "Use LEARNING_STATE, STATE, activation records, and current controller state; produce one bounded process judgment.",
            "role": "LEARNING_PROCESS",
            "tests_preceding_process": False
        }
    ]
    
    # Check for equivalent work in resolved frontier
    current_signature = extract_task_signature(current_task)
    is_novel = check_novelty_guard(current_signature, resolved_frontier_cells)
    
    print(f"Current task: {current_task_id}")
    print(f"Current signature: {current_signature}")
    print(f"Resolved frontier cells: {len(resolved_frontier_cells)}")
    print(f"Novelty guard result: {'PASS' if is_novel else 'FAIL'} ({'PROCEED' if is_novel else 'EQUIVALENT TO RESOLVED WORK'})")
    
    # Save novelty guard status
    novelty_guard_status = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "current_task_id": current_task_id,
        "current_task_signature": current_signature,
        "novelty_guard_applied": True,
        "novelty_guard_pass": is_novel,
        "equivalent_work_found": not is_novel,
        "resolved_frontier_cells_count": len(resolved_frontier_cells),
        "implementation": "minimal_signature_based_guard"
    }
    
    status_file = STATE_DIR / "novelty_guard_status.json"
    with open(status_file, 'w') as f:
        json.dump(novelty_guard_status, f, indent=2)
    
    print(f"Novelty guard status saved to: {status_file}")
    
    if not is_novel:
        print("ERROR: Task equivalent to resolved frontier work. Novelty guard prevented duplicate work.")
        return 1
    else:
        print("SUCCESS: Task is novel. Novelty guard allowed execution.")
        return 0


if __name__ == "__main__":
    exit(main())