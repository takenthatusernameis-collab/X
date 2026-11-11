"""Controller-schema self-check for the agent 02 durable record.

Runs the same field/severity/question checks the controller validator applies
(.github/scripts/validate_campaign_agent.py::validate_record) without touching
worktree state. Read-only.

Usage: python3 scratch/validate_agent_02_record.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import campaign_controller as controller  # noqa: E402

RUN = ROOT / "state" / "campaign" / "runs" / "43012272038" / "agents"


def main() -> int:
    record = json.loads((RUN / "agent_02.json").read_text())
    contract = json.loads((RUN / "agent_02_contract.json").read_text())

    required = [
        "agent_number", "global_agent_number", "campaign_slot", "role", "task_id",
        "objective", "bottleneck", "question", "action", "changed", "verified",
        "unverified", "observed_effect", "uncertainty_targeted",
        "uncertainty_reduced", "process_decision", "research_result", "decision",
        "next", "candidate_tasks", "complexity_added", "failure_class",
        "task_selection_observation",
    ]
    failures = [f"missing:{f}" for f in required if f not in record]
    if record["agent_number"] != 2:
        failures.append("agent_number_mismatch")
    if record.get("campaign_slot") != 2:
        failures.append("campaign_slot_mismatch")
    if record.get("global_agent_number") != contract.get("global_agent_number"):
        failures.append("global_agent_number_mismatch")
    if record["task_id"] != contract["task_id"]:
        failures.append("task_id_mismatch")
    if record["role"] != contract["role"]:
        failures.append("role_mismatch")
    if record["decision"] not in controller.DECISIONS:
        failures.append("invalid_decision")
    if record["process_decision"] not in controller.PROCESS_DECISIONS:
        failures.append("invalid_process_decision")
    if record["task_selection_observation"] not in controller.SELECTION_OBSERVATIONS:
        failures.append("invalid_task_selection_observation")
    if record["failure_class"] not in controller.FAILURE_CLASSES:
        failures.append("invalid_failure_class")
    for key in ("changed", "verified", "unverified"):
        if not isinstance(record[key], list):
            failures.append(f"not_list:{key}")
    if not isinstance(record["candidate_tasks"], list):
        failures.append("candidate_tasks_not_list")
    for idx, task in enumerate(record["candidate_tasks"]):
        try:
            controller.validate_task(task)
        except SystemExit as exc:
            failures.append(f"candidate_{idx}_invalid:{exc}")
    if len([x for x in record["next"].splitlines() if x.strip()]) != 1:
        failures.append("next_must_be_one_bounded_action")
    print("record fields:", len(record))
    print("candidate tasks:", [t["task_id"] for t in record["candidate_tasks"]])
    print("failures:", failures if failures else "NONE")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())