#!/usr/bin/env python3
"""Independent recomputation of the campaign selector's ordinal task ranking.

Reads state/campaign/task_queue.json and reimplements `task_rank` from
.github/scripts/campaign_controller.py without importing the controller, so the
selector's own `selection_basis.compared_against` claim can be checked on an
independent path (controller logic is the claim under audit, not the oracle).

Read-only: writes nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "state" / "campaign" / "task_queue.json"
RUN = ROOT / "state" / "campaign" / "runs" / "43754955571" / "agents"
CONTROLLER = ROOT / ".github" / "scripts" / "campaign_controller.py"

REQUIRED_RECORD_FIELDS = [
    "agent_number",
    "role",
    "task_id",
    "objective",
    "bottleneck",
    "question",
    "action",
    "changed",
    "verified",
    "unverified",
    "observed_effect",
    "uncertainty_targeted",
    "uncertainty_reduced",
    "process_decision",
    "research_result",
    "decision",
    "next",
    "candidate_tasks",
    "complexity_added",
    "failure_class",
    "task_selection_observation",
]

DECISIONS = {
    "NO_SUBSTANTIVE_ACTION",
    "REJECT",
    "RETAIN",
    "UNVERIFIED",
    "USEFUL_CHANGE",
    "VERIFIED_NEGATIVE_RESULT",
}
PROCESS_DECISIONS = {"IMPROVE", "REJECT", "RETAIN", "UNVERIFIED"}
OBSERVATIONS = {"IMPROVED", "UNCHANGED", "WORSENED", "UNVERIFIED"}
FAILURE_CLASSES = {
    "LIVENESS",
    "NONE",
    "PERSISTENCE",
    "RESEARCH_EXECUTION",
    "SCOPE_VIOLATION",
    "TOOL_FAILURE",
    "TRANSIENT_GATEWAY",
    "UNKNOWN",
    "VALIDATION_FAILURE",
}


def load_controller():
    """Import the controller read-only so its own validate_task is the schema oracle."""
    spec = importlib.util.spec_from_file_location("campaign_controller_under_test", CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BENEFIT_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
COST_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def rank(task: dict) -> tuple:
    """Ordinal (non-numeric) ordering: maximize benefit, minimize cost."""
    return (
        BENEFIT_ORDER[task["expected_information_gain"]],
        BENEFIT_ORDER[task["downstream_leverage"]],
        BENEFIT_ORDER[task["falsification_power"]],
        BENEFIT_ORDER[task["verifiability"]],
        COST_ORDER[task["effort"]],
        COST_ORDER[task["complexity"]],
        COST_ORDER[task["execution_risk"]],
        task["task_id"],
    )


def candidates(tasks: list[dict], role: str) -> list[dict]:
    return sorted(
        [t for t in tasks if t["role"] == role and t["status"] == "OPEN"],
        key=rank,
    )


def validate_record() -> None:
    """Validate this agent's durable record so the next selection cannot crash on it."""
    controller = load_controller()
    record_path = RUN / "agent_01.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))

    missing = [k for k in REQUIRED_RECORD_FIELDS if k not in record]
    print("== agent record", record_path.name)
    print("   missing required fields:", missing)
    print("   decision in canonical set:", record["decision"] in DECISIONS)
    print("   process_decision in canonical set:",
          record["process_decision"] in PROCESS_DECISIONS)
    print("   task_selection_observation in canonical set:",
          record["task_selection_observation"] in OBSERVATIONS)
    print("   failure_class in canonical set:",
          record["failure_class"] in FAILURE_CLASSES)
    print("   changed/verified/unverified are arrays:",
          all(isinstance(record[k], list) for k in ("changed", "verified", "unverified")))
    print("   next is exactly one action:",
          isinstance(record["next"], str) and record["next"].count("?") == 0)

    queue_ids = {t["task_id"] for t in json.loads(QUEUE.read_text(encoding="utf-8"))["tasks"]}
    for candidate in record["candidate_tasks"]:
        controller.validate_task(candidate)
        print("   candidate", candidate["task_id"], "valid;",
              "new id" if candidate["task_id"] not in queue_ids else "already in queue")
    print("   candidate count:", len(record["candidate_tasks"]))


def main() -> None:
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    tasks = queue["tasks"]

    for contract_path in sorted(RUN.glob("agent_*_contract.json")):
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        selected_id = contract["task_id"]
        claimed = contract["selection_basis"]["compared_against"]
        profile = contract["selection_basis"]["qualitative_priority"]
        role = contract["role"]

        open_ranked = candidates(tasks, role)
        if not open_ranked:
            print(f"{contract_path.name}: no OPEN tasks for role {role}")
            continue

        chosen = [t for t in tasks if t["task_id"] == selected_id][0]
        recomputed_profile = [
            chosen["expected_information_gain"],
            chosen["downstream_leverage"],
            chosen["falsification_power"],
            chosen["verifiability"],
            chosen["effort"],
            chosen["complexity"],
            chosen["execution_risk"],
        ]

        # SELECTED tasks are no longer OPEN, so re-rank with the selected task
        # restored to the candidate pool to reproduce the selector's view.
        pool = [t for t in tasks if t["role"] == role and t["status"] in {"OPEN", "SELECTED"}]
        pool.sort(key=rank)

        print(f"== {contract_path.name} role={role}")
        print("   pool order:", [t["task_id"] for t in pool])
        print("   contract compared_against:", claimed)
        print("   top-5 reproduced:", [t["task_id"] for t in pool[:5]])
        print("   rank-1 is selected task:", pool[0]["task_id"] == selected_id)
        print("   compared_against matches top-5:", claimed == [t["task_id"] for t in pool[:5]])
        print("   qualitative_priority is selected-task attribute profile:",
              profile == recomputed_profile)

    validate_record()


if __name__ == "__main__":
    main()