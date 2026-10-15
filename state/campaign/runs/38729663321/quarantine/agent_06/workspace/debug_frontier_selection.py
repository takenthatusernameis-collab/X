#!/usr/bin/env python3
"""Debug script for frontier-first selection"""

from research.backtest.frontier_first_selection import (
    is_frontier_cell,
    get_frontier_cell_from_task,
    get_frontier_order,
    apply_frontier_first_priority
)

# Test R-001 task
r001_task = {
    "task_id": "R-001",
    "objective": "Audit recent activation history",
    "expected_information_gain": "HIGH",
    "downstream_leverage": "HIGH",
    "falsification_power": "HIGH",
    "verifiability": "HIGH",
    "effort": "LOW",
    "complexity": "LOW",
    "execution_risk": "LOW"
}

print("Testing R-001 task:")
print(f"  task_id: {r001_task['task_id']}")
print(f"  objective: {r001_task['objective']}")
print(f"  is_frontier_cell: {is_frontier_cell(r001_task)}")
print(f"  get_frontier_cell_from_task: {get_frontier_cell_from_task(r001_task)}")
print()

# Test P-007 task
p007_task = {
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

print("Testing P-007 task:")
print(f"  task_id: {p007_task['task_id']}")
print(f"  objective: {p007_task['objective']}")
print(f"  is_frontier_cell: {is_frontier_cell(p007_task)}")
print(f"  get_frontier_cell_from_task: {get_frontier_cell_from_task(p007_task)}")
print()

# Test R-003 task
r003_task = {
    "task_id": "R-003",
    "objective": "Implement frontier-first selection logic",
    "expected_information_gain": "HIGH",
    "downstream_leverage": "HIGH",
    "falsification_power": "HIGH",
    "verifiability": "HIGH",
    "effort": "LOW",
    "complexity": "LOW",
    "execution_risk": "LOW"
}

print("Testing R-003 task:")
print(f"  task_id: {r003_task['task_id']}")
print(f"  objective: {r003_task['objective']}")
print(f"  is_frontier_cell: {is_frontier_cell(r003_task)}")
print(f"  get_frontier_cell_from_task: {get_frontier_cell_from_task(r003_task)}")
print()

# Test the full apply_frontier_first_priority logic
tasks = [p007_task, r003_task, r001_task]

print("Applying frontier-first priority to tasks:")
for task in tasks:
    print(f"  - {task['task_id']}: frontier_cell={is_frontier_cell(task)}, cell_type={get_frontier_cell_from_task(task)}")

applied_tasks = apply_frontier_first_priority(tasks)
print(f"\nApplied tasks order: {[t['task_id'] for t in applied_tasks]}")

# Check if R-001 is in the list
r001_in_list = any(t['task_id'] == 'R-001' for t in applied_tasks)
print(f"R-001 in applied list: {r001_in_list}")