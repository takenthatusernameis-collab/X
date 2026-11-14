#!/usr/bin/env python3
"""Read-only audit of the controller's focused-task selector (campaign 43548892244).

Purpose (task P-001): produce deterministic, re-runnable evidence about what the
ordinal selector actually decides, so the selector judgment does not rest on
agent confidence. It reuses the controller's own `task_rank` / `candidate_tasks_for_role`
and only reads durable state; it writes nothing.

It reports, for the role that is currently being served:
  1. the exact candidate ordering the selector produces, with rank tuples;
  2. tie groups (candidates the selector cannot separate on qualitative labels);
  3. whether the ordering reproduces the `selection_basis.compared_against`
     recorded in the live contract;
  4. how many selections have actually been observed, and how many per-agent
     outcome records exist to evaluate them (the evidence base for any claim
     that the selector improves learning efficiency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

import campaign_controller as controller  # noqa: E402

RUN_ID = "43548892244"


def main() -> int:
    tasks = controller.load_queue()
    status = json.loads((controller.run_dir(RUN_ID) / "campaign_status.json").read_text(encoding="utf-8"))
    contract = json.loads((controller.run_dir(RUN_ID) / "agents" / "agent_01_contract.json").read_text(encoding="utf-8"))

    role = contract["role"]
    # `task_rank` ends with task_id, so the qualitative part is rank[:-1]; a task
    # that is already SELECTED must be treated as it stood when the ranking ran.
    candidates = sorted(
        [t for t in tasks if t["role"] == role and t["status"] in {"OPEN", "SELECTED"}],
        key=controller.task_rank,
    )
    print(f"role={role} rankable_candidates(open+selected)={len(candidates)}")

    print("\n[1] selector ordering (controller.task_rank, ascending):")
    for pos, task in enumerate(candidates):
        rank = controller.task_rank(task)
        print(f"  {pos + 1}. {task['task_id']}  status={task['status']}  rank={rank}")

    print("\n[2] tie groups on the qualitative part of the rank (task_id is then the tie-break):")
    groups: dict[tuple, list[str]] = {}
    for task in candidates:
        groups.setdefault(controller.task_rank(task)[:-1], []).append(task["task_id"])
    tied = {k: v for k, v in groups.items() if len(v) > 1}
    if not tied:
        print("  none")
    for rank, ids in sorted(tied.items()):
        print(f"  tied={ids} qualitative_rank={rank} -> resolved by task_id ascending")

    print("\n[3] recorded selection_basis vs reproduced ordering:")
    recorded = contract["selection_basis"]["compared_against"]
    reproduced = [t["task_id"] for t in candidates[: len(recorded)]]
    print(f"  recorded  ={recorded}")
    print(f"  reproduced={reproduced}")
    print(f"  match={recorded == reproduced}")
    print(f"  selected_task={contract['task_id']} is_first_candidate={candidates[0]['task_id'] == contract['task_id']}")

    print("\n[4] observed selections and outcome records:")
    history = status.get("selection_history", [])
    print(f"  selection_history_entries={len(history)}")
    for entry in history:
        print(f"    agent={entry['agent_number']:02d} role={entry['role']} task={entry['task_id']}")
    agent_dir = controller.run_dir(RUN_ID) / "agents"
    records = sorted(p.name for p in agent_dir.glob("agent_*.json") if "_contract" not in p.name and "_before" not in p.name)
    print(f"  existing_agent_outcome_records={records}")

    print("\n[5] how much of the queue the labels actually separate:")
    for role_name in ("LEARNING_PROCESS", "HIGHER_ORDER_OBJECTIVE"):
        role_tasks = [t for t in tasks if t["role"] == role_name]
        if not role_tasks:
            continue
        distinct = len({controller.task_rank(t)[:-1] for t in role_tasks})
        print(f"  {role_name}: tasks={len(role_tasks)} distinct_qualitative_rank_tuples={distinct}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())