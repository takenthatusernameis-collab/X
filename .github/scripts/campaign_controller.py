#!/usr/bin/env python3
"""Lightweight controller for the sequential 10-agent research campaign.

The controller owns task validation, qualitative prioritization, durable task
contracts, per-agent state transitions, and final campaign synthesis inputs.
It deliberately avoids a numeric reward score.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
RUNS_ROOT = ROOT / "state" / "campaign" / "runs"
QUEUE_PATH = ROOT / "state" / "campaign" / "task_queue.json"
CURRENT_TASK = ROOT / "state" / "campaign" / "current_task.json"

ROLE_BY_AGENT = {n: ("LEARNING_PROCESS" if n % 2 else "HIGHER_ORDER_OBJECTIVE") for n in range(1, 11)}
TASK_STATUSES = {"OPEN", "SELECTED", "RESOLVED", "REJECTED", "DEFERRED"}
DECISIONS = {
    "USEFUL_CHANGE",
    "VERIFIED_NEGATIVE_RESULT",
    "RETAIN",
    "REJECT",
    "UNVERIFIED",
    "NO_SUBSTANTIVE_ACTION",
}
PROCESS_DECISIONS = {"IMPROVE", "RETAIN", "REJECT", "UNVERIFIED"}
SEVERITIES = {"HIGH", "MEDIUM", "LOW"}
SELECTION_OBSERVATIONS = {"IMPROVED", "UNCHANGED", "WORSENED", "UNVERIFIED"}
FAILURE_CLASSES = {
    "NONE",
    "TRANSIENT_GATEWAY",
    "TOOL_FAILURE",
    "RESEARCH_EXECUTION",
    "VALIDATION_FAILURE",
    "LIVENESS",
    "PERSISTENCE",
    "SCOPE_VIOLATION",
    "UNKNOWN",
}

REQUIRED_TASK_FIELDS = [
    "task_id",
    "role",
    "status",
    "primary_question",
    "bottleneck",
    "objective",
    "scope",
    "out_of_scope",
    "deliverable",
    "success_criterion",
    "stop_condition",
    "verification_requirement",
    "evidence_basis",
    "expected_information_gain",
    "downstream_leverage",
    "falsification_power",
    "verifiability",
    "effort",
    "complexity",
    "execution_risk",
    "tests_preceding_process",
]


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / str(run_id)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "
", encoding="utf-8")


def question_is_single(question: str) -> bool:
    q = question.strip()
    if not q or len(q) > 500:
        return False
    if q.count("?") > 1:
        return False
    banned = (
        "improve everything",
        "continue research",
        "find something interesting",
        "advance the repository",
    )
    lower = q.lower()
    return not any(x in lower for x in banned)


def validate_task(task: dict[str, Any], *, expected_role: str | None = None) -> None:
    missing = [k for k in REQUIRED_TASK_FIELDS if k not in task]
    if missing:
        raise SystemExit(f"task missing fields: {missing}")
    if not isinstance(task["task_id"], str) or not task["task_id"].strip():
        raise SystemExit("task_id must be non-empty")
    if expected_role is not None and task["role"] != expected_role:
        raise SystemExit(f"task {task['task_id']} has role {task['role']}, expected {expected_role}")
    if task["status"] not in TASK_STATUSES:
        raise SystemExit(f"task {task['task_id']} has invalid status {task['status']!r}")
    if not question_is_single(str(task["primary_question"])):
        raise SystemExit(f"task {task['task_id']} violates focused-question firewall")
    for key in REQUIRED_TASK_FIELDS[4:14]:
        if not isinstance(task[key], str) or not task[key].strip():
            raise SystemExit(f"task {task['task_id']} field {key} must be non-empty text")
    for key in (
        "expected_information_gain",
        "downstream_leverage",
        "falsification_power",
        "verifiability",
        "effort",
        "complexity",
        "execution_risk",
    ):
        if task[key] not in SEVERITIES:
            raise SystemExit(f"task {task['task_id']} field {key} must be HIGH/MEDIUM/LOW")
    if not isinstance(task["tests_preceding_process"], bool):
        raise SystemExit(f"task {task['task_id']} tests_preceding_process must be boolean")


def load_queue() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        raise SystemExit("durable task queue is missing")
    raw = load_json(QUEUE_PATH)
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("durable task queue has no tasks")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise SystemExit("task queue contains non-object")
        if task["task_id"] in seen:
            raise SystemExit(f"duplicate task_id {task['task_id']}")
        seen.add(task["task_id"])
        validate_task(task)
    return tasks


def save_queue(tasks: list[dict[str, Any]]) -> None:
    write_json(
        QUEUE_PATH,
        {
            "schema_version": 1,
            "selection_principle": "qualitative_evidence_backed_priority",
            "tasks": tasks,
        },
    )


def severity_order(value: str) -> int:
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[value]


def task_rank(task: dict[str, Any]) -> tuple[Any, ...]:
    # Ordinal qualitative ordering, not a fabricated score.
    return (
        severity_order(task["expected_information_gain"]),
        severity_order(task["downstream_leverage"]),
        severity_order(task["falsification_power"]),
        severity_order(task["verifiability"]),
        severity_order(task["effort"]),
        severity_order(task["complexity"]),
        severity_order(task["execution_risk"]),
        task["task_id"],
    )


def candidate_tasks_for_role(tasks: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return sorted(
        [t for t in tasks if t["role"] == role and t["status"] == "OPEN"],
        key=task_rank,
    )


def init_campaign(run_id: str, agent_count: int) -> dict[str, Any]:
    if agent_count not in {2, 10}:
        raise SystemExit("agent_count must be 2 or 10")
    tasks = load_queue()
    for role in sorted(set(ROLE_BY_AGENT[n] for n in range(1, agent_count + 1))):
        if not any(t["role"] == role for t in tasks):
            raise SystemExit(f"durable task queue has no tasks for role {role}")
    for task in tasks:
        if task["status"] == "SELECTED":
            task["status"] = "OPEN"
    save_queue(tasks)

    d = run_dir(run_id)
    (d / "agents").mkdir(parents=True, exist_ok=True)
    write_json(d / "task_queue_initial.json", {"tasks": tasks})
    status = {
        "schema_version": 1,
        "run_id": str(run_id),
        "agent_count": agent_count,
        "status": "IN_PROGRESS",
        "phase": "SETUP",
        "completed_agents": [],
        "selected_tasks": [],
        "selection_history": [],
        "created_from": str(QUEUE_PATH),
    }
    write_json(d / "campaign_status.json", status)
    return status


def current_contract(run_id: str, agent_number: int) -> dict[str, Any]:
    path = run_dir(run_id) / "agents" / f"agent_{agent_number:02d}_contract.json"
    if not path.exists():
        raise SystemExit(f"missing controller contract for agent {agent_number:02d}")
    return load_json(path)


def select_task(run_id: str, agent_number: int) -> dict[str, Any]:
    d = run_dir(run_id)
    if not d.exists():
        raise SystemExit("campaign has not been initialized")
    expected_role = ROLE_BY_AGENT.get(agent_number)
    campaign_count = int(load_json(d / "campaign_status.json")["agent_count"])
    if expected_role is None or agent_number > campaign_count:
        raise SystemExit("agent number is outside campaign")
    tasks = load_queue()

    if agent_number > 1:
        prev = d / "agents" / f"agent_{agent_number-1:02d}.json"
        if prev.exists():
            previous_record = load_json(prev)
            for candidate in previous_record.get("candidate_tasks", []):
                validate_task(candidate)
                if not any(t["task_id"] == candidate["task_id"] for t in tasks):
                    candidate = dict(candidate)
                    candidate["status"] = "OPEN"
                    tasks.append(candidate)

            previous_task_id = previous_record.get("task_id")
            decision = previous_record.get("decision")
            for task in tasks:
                if task["task_id"] != previous_task_id:
                    continue
                if decision in {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN", "REJECT", "NO_SUBSTANTIVE_ACTION"}:
                    task["status"] = "RESOLVED" if decision != "REJECT" else "REJECTED"
                else:
                    task["status"] = "DEFERRED"
    save_queue(tasks)

    candidates = candidate_tasks_for_role(tasks, expected_role)
    if not candidates:
        raise SystemExit(f"no OPEN focused task remains for role {expected_role}")

    selected = dict(candidates[0])
    selected["status"] = "SELECTED"
    selected["agent_number"] = agent_number
    selected["selected_role"] = expected_role

    prev_record = None
    if agent_number > 1:
        prev_path = d / "agents" / f"agent_{agent_number-1:02d}.json"
        if prev_path.exists():
            prev_record = load_json(prev_path)

    contract = {
        "task_id": selected["task_id"],
        "agent_number": agent_number,
        "role": expected_role,
        "primary_question": selected["primary_question"],
        "bottleneck": selected["bottleneck"],
        "objective": selected["objective"],
        "scope": selected["scope"],
        "out_of_scope": selected["out_of_scope"],
        "deliverable": selected["deliverable"],
        "success_criterion": selected["success_criterion"],
        "stop_condition": selected["stop_condition"],
        "verification_requirement": selected["verification_requirement"],
        "evidence_basis": selected["evidence_basis"],
        "tests_preceding_process": selected["tests_preceding_process"],
        "selection_basis": {
            "qualitative_priority": [
                selected["expected_information_gain"],
                selected["downstream_leverage"],
                selected["falsification_power"],
                selected["verifiability"],
                selected["effort"],
                selected["complexity"],
                selected["execution_risk"],
            ],
            "compared_against": [x["task_id"] for x in candidates[:5]],
            "reason": "highest evidence-backed qualitative priority among currently OPEN tasks for this role",
        },
        "preceding_agent": prev_record if prev_record else None,
    }

    for task in tasks:
        if task["task_id"] == selected["task_id"]:
            task["status"] = "SELECTED"
            task["selected_agent"] = agent_number
    save_queue(tasks)

    write_json(d / "agents" / f"agent_{agent_number:02d}_contract.json", contract)
    write_json(CURRENT_TASK, contract)

    status = load_json(d / "campaign_status.json")
    status["phase"] = f"AGENT_{agent_number:02d}_SELECTED"
    status["selected_tasks"].append({"agent_number": agent_number, "task_id": selected["task_id"]})
    status["selection_history"].append(
        {
            "agent_number": agent_number,
            "role": expected_role,
            "task_id": selected["task_id"],
            "compared_against": [x["task_id"] for x in candidates[:5]],
        }
    )
    write_json(d / "campaign_status.json", status)
    return contract


def render_prompt(run_id: str, agent_number: int) -> str:
    contract = current_contract(run_id, agent_number)
    role = contract["role"]
    if role == "LEARNING_PROCESS":
        role_guidance = (
            "Improve, test, or challenge the research-learning process. A no-change "
            "RETAIN/REJECT/UNVERIFIED outcome is valid. Do not change the process without "
            "observed evidence of a bottleneck or capability gain."
        )
    else:
        role_guidance = (
            "Advance the highest-value unresolved quantitative-research frontier using the "
            "current process. Explicitly evaluate whether the preceding learning-process "
            "decision helped; record what uncertainty changed; do not optimize a local "
            "result merely because it is already underway."
        )

    return f"""You are Agent {agent_number:02d} in a controlled sequential quantitative-research campaign.
This is a genuinely fresh Kilo session. Do not rely on prior live Kilo context or on another agent's conversational state.
The repository's durable state is the only cross-agent communication medium.

HIGHER-ORDER OBJECTIVE:
Improve the system's ability to choose what is worth learning, learn it efficiently, falsify it, validate it independently, preserve the evidence, and choose what to learn next.

ROLE:
{role}

ROLE GUIDANCE:
{role_guidance}

AUTHORITATIVE FOCUSED-TASK CONTRACT:
{json.dumps(contract, indent=2, sort_keys=True)}

MANDATORY SESSION FIREWALL:
- Exactly one primary learning question.
- Exactly one bounded objective.
- Exactly one meaningful deliverable.
- Exactly one evidence gate.
- Exactly one explicit stop condition.
- Zero intentional scope expansion.
- Secondary ideas become candidate_tasks in the durable record; they are NOT current work.
- Do not redefine the campaign objective.
- Do not dispatch another workflow.
- Do not create hidden retries or another Kilo session.
- Do not modify .github/workflows/**, .github/scripts/**, .kilo/**, AGENTS.md, ENTERPRISE.md, MANUAL_SETUP.md, or PERSISTENCE_POLICY.md.
- Do not add cosmetic changes.
- Do not treat commands executed, files edited, workflow success, tokens, runtime, or confidence as learning evidence.

BEFORE SUBSTANTIVE WORK, ANSWER CONCISELY IN YOUR NOTES:
1. What is the system ultimately trying to become better at learning?
2. What currently most constrains useful learning?
3. What uncertainty prevents a better decision?
4. What single bounded action is most likely to reduce it?
5. Why is that action worth its effort, complexity, and execution risk?

EXECUTION:
- Inspect actual durable evidence before changing anything.
- Stay within the contract's scope and out_of_scope boundaries.
- Prefer existing deterministic research infrastructure.
- Use the narrowest relevant validation.
- Independently inspect resulting evidence rather than trusting your own interpretation.
- Stop when the success criterion or stop condition is reached, or when the task is no longer the highest-value use of effort.
- Preserve negative and inconclusive evidence.

DURABLE HANDOFF:
Write exactly one JSON record to:
state/campaign/runs/{run_id}/agents/agent_{agent_number:02d}.json

Required record fields:
agent_number, role, task_id, objective, bottleneck, question, action, changed, verified,
unverified, observed_effect, uncertainty_targeted, uncertainty_reduced,
process_decision, research_result, decision, next, candidate_tasks,
complexity_added, failure_class, task_selection_observation

Canonical decision values:
{sorted(DECISIONS)}

Canonical process_decision values:
{sorted(PROCESS_DECISIONS)}

Canonical task_selection_observation values:
{sorted(SELECTION_OBSERVATIONS)}

Canonical failure_class values:
{sorted(FAILURE_CLASSES)}

Rules:
- decision must reflect evidence, not effort.
- next must contain exactly ONE bounded next action.
- candidate_tasks may contain zero or more future tasks, each with the same focused-task schema as the current contract.
- changed/verified/unverified must be arrays.
- failure_class must be NONE when no failure occurred.
- task_selection_observation must state whether the selected task improved the learning process, remained unchanged, worsened it, or is unverified.
- Do not erase contradictory evidence.
- Update state/worker_progress.md only at genuine semantic milestones.

When the evidence gate is satisfied, STOP.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("--run-id", required=True)
    p.add_argument("--agent-count", type=int, required=True, choices=(2, 10))

    p = sub.add_parser("select")
    p.add_argument("--run-id", required=True)
    p.add_argument("--agent-number", type=int, required=True)

    p = sub.add_parser("render-prompt")
    p.add_argument("--run-id", required=True)
    p.add_argument("--agent-number", type=int, required=True)

    args = ap.parse_args()
    if args.command == "init":
        init_campaign(args.run_id, args.agent_count)
        print(f"CAMPAIGN_INITIALIZED={args.run_id}")
        return 0
    if args.command == "select":
        contract = select_task(args.run_id, args.agent_number)
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    if args.command == "render-prompt":
        print(render_prompt(args.run_id, args.agent_number))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
