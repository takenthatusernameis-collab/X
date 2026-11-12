"""Read-only duplicate-avoidance audit for task P-002.

Answers, from durable evidence only:
  1. Which check scripts cover the same lookback grid (near-duplicate checks)?
  2. Which of those scripts are orphaned (no persisted artifact, no cross-reference)?
  3. Which LEARNING_PROCESS tasks in the live queue share an audit subject?

Mutates no state. Prints a JSON summary to stdout.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKS = ROOT / "research" / "checks"
ARTIFACTS = ROOT / "state" / "check_artifacts"
QUEUE = ROOT / "state" / "campaign" / "task_queue.json"


def lookbacks_of(path: Path) -> list[int]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "LOOKBACKS":
                    try:
                        return list(ast.literal_eval(node.value))
                    except Exception:
                        return []
    return []


def artifact_name(path: Path) -> str:
    m = re.search(r'"(momentum_[a-z_]+_results\.json)"', path.read_text())
    return m.group(1) if m else ""


def references(name: str) -> list[str]:
    hits = []
    for p in CHECKS.glob("*.py"):
        if p.name == name:
            continue
        if name in p.read_text():
            hits.append(p.name)
    return hits


momentum_checks = sorted(p.name for p in CHECKS.glob("momentum*.py"))
grid = {}
for name in momentum_checks:
    p = CHECKS / name
    lbs = lookbacks_of(p)
    if lbs:
        grid[name] = {
            "lookbacks": lbs,
            "artifact": artifact_name(p),
            "artifact_exists": bool(artifact_name(p)) and (ARTIFACTS / artifact_name(p)).exists(),
            "referenced_by": references(name),
        }

# Overlap = two checks whose lookback grids share >= 2 values.
overlaps = []
names = list(grid)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        shared = sorted(set(grid[names[i]]["lookbacks"]) & set(grid[names[j]]["lookbacks"]))
        if len(shared) >= 2:
            overlaps.append({"a": names[i], "b": names[j], "shared_lookbacks": shared})

# Orphaned = has lookbacks, no artifact, and no cross-reference.
orphans = [n for n, v in grid.items() if not v["artifact_exists"] and not v["referenced_by"]]

queue = json.loads(QUEUE.read_text())["tasks"]
lp = [t for t in queue if t["role"] == "LEARNING_PROCESS"]
task_view = {
    t["task_id"]: {"status": t["status"], "objective": t["objective"]} for t in lp
}
same_subject_pairs = [
    ("P-002", "P-003", "both audit durable history / task-queue reuse of prior work"),
    ("P-001", "P-007", "both audit whether the highest-value selector improves learning efficiency"),
]

print(json.dumps({
    "momentum_check_grids": grid,
    "overlapping_check_pairs": overlaps,
    "orphaned_checks": orphans,
    "learning_process_tasks": task_view,
    "near_duplicate_task_pairs": [
        {
            "a": a,
            "b": b,
            "a_status": task_view.get(a, {}).get("status"),
            "b_status": task_view.get(b, {}).get("status"),
            "reason": reason,
        }
        for a, b, reason in same_subject_pairs
    ],
}, indent=2, sort_keys=True))