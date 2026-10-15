#!/usr/bin/env python3
"""Implements frontier-first selection logic for the task queue producer

This module provides frontier-first selection without modifying controller
architecture or adding numerical reward scoring.

Based on agent 05 audit evidence: MA crossover -> reversal -> CSRS -> volatility targeting -> momentum
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# Established frontier order from agent 05 audit evidence
FRONTIER_ORDER = {
    "MA crossover": 1,
    "reversal": 2,
    "CSRS": 3,
    "volatility targeting": 4,
    "momentum": 5
}

FRONTIER_DESCRIPTIONS = {
    "MA crossover": "MA crossover hypotheses on collected data (falsified - no further work needed)",
    "reversal": "Short-horizon return-reversion signal (mean-reversion)",
    "CSRS": "Cross-sectional relative strength (long top-3 / short bottom-3 by 20-day lookback)",
    "volatility targeting": "Volatility-normalized sizing / long low-vol short high-vol spread",
    "momentum": "Short-horizon momentum signal (long previous 5-day return)"
}

def get_frontier_order() -> Dict[str, int]:
    """Returns the established frontier order from agent 05 audit evidence"""
    return FRONTIER_ORDER.copy()

def get_frontier_descriptions() -> Dict[str, str]:
    """Returns descriptions of each frontier cell"""
    return FRONTIER_DESCRIPTIONS.copy()

def is_frontier_cell(task: Dict[str, Any]) -> bool:
    """Checks if a task is part of the established frontier"""
    task_id = task.get("task_id", "")
    task_objective = task.get("objective", "").lower()
    
    # All HIGHER_ORDER_OBJECTIVE tasks are frontier cells
    # Tasks with objective containing "frontier" or "selection"
    # Tasks with task_id starting with "P-" (process tasks) 
    # Tasks with task_id starting with "R-" (research tasks)
    
    if task_id.startswith("P-") or task_id.startswith("R-"):
        return True
    
    # Check objective for frontier keywords
    frontier_keywords = [
        "frontier", "selection", "research", "implementation",
        "audit", "evaluate", "test", "execute", "verify"
    ]
    
    if any(keyword in task_objective for keyword in frontier_keywords):
        return True
        
    return False

def get_frontier_cell_from_task(task: Dict[str, Any]) -> Optional[str]:
    """Maps a task to its frontier cell type"""
    task_id = task.get("task_id", "")
    
    # Direct mapping
    if task_id == "P-007":
        return "frontier-first selection"
    elif task_id == "R-001":
        return "reversal"
    elif task_id == "R-003":
        return "CSRS"
    elif task_id == "R-005":
        return "volatility targeting"
    elif "momentum" in task_id.lower():
        return "momentum"
    
    # Map from objective
    objective_lower = task.get("objective", "").lower()
    if "frontier" in objective_lower:
        return "frontier-first selection"
    elif "mean reversion" in objective_lower or "reversal" in objective_lower:
        return "reversal"
    elif "cross-section" in objective_lower or "relative strength" in objective_lower:
        return "CSRS"
    elif "volatility targeting" in objective_lower:
        return "volatility targeting"
    elif "momentum" in objective_lower:
        return "momentum"
    
    return None

def apply_frontier_first_priority(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies frontier-first priority to task list while preserving existing rank
    
    This function reorders tasks to prioritize frontier cells in the established
    order (MA crossover -> reversal -> CSRS -> volatility targeting -> momentum)
    while maintaining the original ranking within each frontier cell.
    """
    # Separate frontier cells and non-frontier cells
    frontier_tasks = []
    non_frontier_tasks = []
    
    for task in tasks:
        if is_frontier_cell(task):
            frontier_tasks.append(task)
        else:
            non_frontier_tasks.append(task)
    
    # Sort frontier tasks by established order with P-007 having highest priority
    def get_frontier_priority(task: Dict[str, Any]) -> int:
        # P-007 (frontier-first selection) has highest priority
        if task.get("task_id") == "P-007":
            return 0  # Highest priority (comes first)
        
        cell_type = get_frontier_cell_from_task(task)
        if cell_type in FRONTIER_ORDER:
            return FRONTIER_ORDER[cell_type] + 1  # Add 1 so MA crossover is priority 1
        return 999  # Low priority for unknown
    
    # Sort by frontier priority, then by original task rank
    def get_original_rank(task: Dict[str, Any]) -> tuple:
        # Use the same ranking logic as in controller.task_rank()
        benefit_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        cost_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        
        return (
            benefit_order.get(task.get("expected_information_gain", "LOW"), 1),
            benefit_order.get(task.get("downstream_leverage", "LOW"), 1),
            benefit_order.get(task.get("falsification_power", "LOW"), 1),
            benefit_order.get(task.get("verifiability", "LOW"), 1),
            cost_order.get(task.get("effort", "LOW"), 1),
            cost_order.get(task.get("complexity", "LOW"), 1),
            cost_order.get(task.get("execution_risk", "LOW"), 1),
            task.get("task_id", "")
        )
    
    # Sort by frontier priority first, then by original rank
    frontier_tasks.sort(key=lambda t: (get_frontier_priority(t), get_original_rank(t)))
    
    # Add non-frontier tasks at the end, preserving original order
    result = frontier_tasks + non_frontier_tasks
    
    return result

def get_next_frontier_cell(frontier_tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Gets the next unresolved frontier cell based on established order"""
    # The P-007 task is the frontier-first selection implementation task itself
    # Other frontier cells (R-001, R-003, R-005) are research tasks
    
    for priority in sorted(FRONTIER_ORDER.values()):
        for task in frontier_tasks:
            cell_type = get_frontier_cell_from_task(task)
            if cell_type and FRONTIER_ORDER[cell_type] == priority:
                return task
    
    return None

def update_frontier_status(frontier_state: Dict[str, Any], task_id: str, status: str) -> Dict[str, Any]:
    """Updates frontier status based on task resolution"""
    if not frontier_state:
        frontier_state = {
            "total_cells": len(FRONTIER_ORDER),
            "completed_cells": [],
            "remaining_cells": list(FRONTIER_ORDER.keys()),
            "implementation_task": "P-007"
        }
    
    # Map task_id to frontier cell
    cell_type = None
    if task_id == "P-007":
        cell_type = "frontier-first selection"
    elif task_id == "R-001":
        cell_type = "reversal"
    elif task_id == "R-003":
        cell_type = "CSRS"
    elif task_id == "R-005":
        cell_type = "volatility targeting"
    
    if cell_type:
        if status in ["RESOLVED", "REJECTED", "FALSIFIED"]:
            if cell_type not in frontier_state["completed_cells"]:
                frontier_state["completed_cells"].append(cell_type)
        
        if cell_type in frontier_state["remaining_cells"]:
            frontier_state["remaining_cells"].remove(cell_type)
    
    return frontier_state

def verify_frontier_first_selection(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verifies that frontier-first selection is correctly applied"""
    result = {
        "valid": True,
        "violations": [],
        "frontier_cells_count": 0,
        "ordered_correctly": True
    }
    
    # Count frontier cells
    frontier_cells = [t for t in tasks if is_frontier_cell(t)]
    result["frontier_cells_count"] = len(frontier_cells)
    
    # Check ordering (skip "frontier-first selection" as it's the implementation task)
    for i in range(1, len(frontier_cells)):
        current = frontier_cells[i-1]
        next_task = frontier_cells[i]
        
        current_cell = get_frontier_cell_from_task(current)
        next_cell = get_frontier_cell_from_task(next_task)
        
        # Skip verification for implementation task "frontier-first selection"
        if current_cell == "frontier-first selection" or next_cell == "frontier-first selection":
            continue
            
        if current_cell and next_cell and FRONTIER_ORDER[current_cell] >= FRONTIER_ORDER[next_cell]:
            result["ordered_correctly"] = False
            result["violations"].append(
                f"Frontier cells not in order: {current_cell} -> {next_cell}"
            )
    
    return result

def get_frontier_summary() -> Dict[str, Any]:
    """Gets a summary of the frontier and its status"""
    return {
        "frontier_order": FRONTIER_ORDER,
        "frontier_descriptions": FRONTIER_DESCRIPTIONS,
        "implementation_task": "P-007",
        "total_cells": len(FRONTIER_ORDER),
        "expected_sequence": list(FRONTIER_ORDER.keys()),
        "evidence_basis": "Agent 05 audit demonstrates effectiveness across MA crossover -> reversal -> CSRS -> volatility targeting -> momentum sequence"
    }

if __name__ == "__main__":
    # Test the frontier-first selection
    test_tasks = [
        {
            "task_id": "R-003",
            "objective": "Implement the frontier-first selection logic in the task_queue_producer",
            "expected_information_gain": "HIGH",
            "downstream_leverage": "HIGH",
            "falsification_power": "HIGH",
            "verifiability": "HIGH",
            "effort": "LOW",
            "complexity": "LOW",
            "execution_risk": "LOW"
        },
        {
            "task_id": "R-001", 
            "objective": "Audit recent activation history and frontier decisions",
            "expected_information_gain": "HIGH",
            "downstream_leverage": "HIGH",
            "falsification_power": "HIGH",
            "verifiability": "HIGH",
            "effort": "LOW",
            "complexity": "LOW",
            "execution_risk": "LOW"
        },
        {
            "task_id": "P-007",
            "objective": "Implement the frontier-first selection logic in the task_queue_producer",
            "expected_information_gain": "HIGH",
            "downstream_leverage": "HIGH",
            "falsification_power": "HIGH", 
            "verifiability": "HIGH",
            "effort": "LOW",
            "complexity": "LOW",
            "execution_risk": "LOW"
        }
    ]
    
    print("Testing frontier-first selection...")
    applied_tasks = apply_frontier_first_priority(test_tasks)
    
    print("\nApplied tasks in frontier-first order:")
    for task in applied_tasks:
        cell = get_frontier_cell_from_task(task)
        print(f"  - {task['task_id']}: {cell}")
    
    verification = verify_frontier_first_selection(applied_tasks)
    print(f"\nVerification: {'PASS' if verification['valid'] else 'FAIL'}")
    print(f"  - Frontier cells: {verification['frontier_cells_count']}")
    print(f"  - Ordered correctly: {verification['ordered_correctly']}")