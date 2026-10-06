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
GLOBAL_COUNTER_PATH = ROOT / "state" / "campaign" / "global_agent_counter.json"

ROLE_BY_AGENT = {n: ("LEARNING_PROCESS" if n % 2 else "HIGHER_ORDER_OBJECTIVE") for n in range(1, 11)}
AUTONOMOUS_ROLE = "AUTONOMOUS_RESEARCH"
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def allocate_global_agent_numbers(run_id: str, agent_count: int, identity_mode: str) -> dict[str, Any]:
    if identity_mode not in {"PERSISTENT", "EPHEMERAL"}:
        raise SystemExit("identity_mode must be PERSISTENT or EPHEMERAL")
    if agent_count < 1:
        raise SystemExit("agent_count must be positive")
    if identity_mode == "EPHEMERAL":
        return {str(slot): f"MINI-{slot:02d}" for slot in range(1, agent_count + 1)}

    state: dict[str, Any] = {
        "schema_version": 1,
        "next_global_agent_number": 1,
        "campaigns": [],
    }
    if GLOBAL_COUNTER_PATH.exists():
        state = load_json(GLOBAL_COUNTER_PATH)
    next_number = int(state.get("next_global_agent_number", 1))
    if next_number < 1:
        raise SystemExit("global agent counter is invalid")
    numbers = {str(slot): next_number + slot - 1 for slot in range(1, agent_count + 1)}
    state["next_global_agent_number"] = next_number + agent_count
    state.setdefault("campaigns", [])
    state["campaigns"].append(
        {
            "run_id": str(run_id),
            "global_start": next_number,
            "global_end": next_number + agent_count - 1,
            "agent_count": agent_count,
        }
    )
    write_json(GLOBAL_COUNTER_PATH, state)
    return numbers


def question_is_single(question: str) -> bool:
    q = question.strip()
    if not q or len(q) > 500:
        return False
    if q.count("?") > 1:
        return False
    banned = (
        "improve everything",
        "improve the research system",
        "improve the research process",
        "improve the repository",
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


def benefit_order(value: str) -> int:
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[value]


def cost_order(value: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[value]


def task_rank(task: dict[str, Any]) -> tuple[Any, ...]:
    # Ordinal qualitative ordering, not a fabricated score:
    # maximize information/leverage/falsification/verifiability and minimize cost.
    return (
        benefit_order(task["expected_information_gain"]),
        benefit_order(task["downstream_leverage"]),
        benefit_order(task["falsification_power"]),
        benefit_order(task["verifiability"]),
        cost_order(task["effort"]),
        cost_order(task["complexity"]),
        cost_order(task["execution_risk"]),
        task["task_id"],
    )


def candidate_tasks_for_role(tasks: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return sorted(
        [t for t in tasks if t["role"] == role and t["status"] == "OPEN"],
        key=task_rank,
    )


def init_campaign(run_id: str, agent_count: int, identity_mode: str = "PERSISTENT", campaign_mode: str = "LEGACY") -> dict[str, Any]:
    if agent_count not in {2, 10}:
        raise SystemExit("agent_count must be 2 or 10")
    if campaign_mode not in {"LEGACY", "AUTONOMOUS"}:
        raise SystemExit("campaign_mode must be LEGACY or AUTONOMOUS")
    tasks = load_queue()
    for role in sorted(set(ROLE_BY_AGENT[n] for n in range(1, agent_count + 1))):
        if not any(t["role"] == role for t in tasks):
            raise SystemExit(f"durable task queue has no tasks for role {role}")
    for task in tasks:
        if task["status"] == "SELECTED":
            task["status"] = "OPEN"
    save_queue(tasks)
    global_agent_numbers = allocate_global_agent_numbers(run_id, agent_count, identity_mode)

    d = run_dir(run_id)
    (d / "agents").mkdir(parents=True, exist_ok=True)
    write_json(d / "task_queue_initial.json", {"tasks": tasks})
    status = {
        "schema_version": 1,
        "run_id": str(run_id),
        "agent_count": agent_count,
        "identity_mode": identity_mode,
        "campaign_mode": campaign_mode,
        "global_agent_numbers": global_agent_numbers,
        "global_agent_start": list(global_agent_numbers.values())[0],
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
    campaign_status = load_json(d / "campaign_status.json")
    campaign_count = int(campaign_status["agent_count"])
    global_agent_number = campaign_status["global_agent_numbers"].get(str(agent_number))
    if agent_number < 1 or agent_number > campaign_count:
        raise SystemExit("agent number is outside campaign")
    if global_agent_number is None:
        raise SystemExit("global agent identity is missing for campaign slot")

    campaign_mode = str(campaign_status.get("campaign_mode", "LEGACY")).upper()
    if campaign_mode == "AUTONOMOUS":
        task_id = f"autonomous-{run_id}-{agent_number:02d}"
        previous = None
        if agent_number > 1:
            prev_path = d / "agents" / f"agent_{agent_number-1:02d}_contract.json"
            if prev_path.exists():
                previous = load_json(prev_path)
        contract = {
            "task_id": task_id,
            "agent_number": agent_number,
            "campaign_slot": agent_number,
            "global_agent_number": global_agent_number,
            "role": AUTONOMOUS_ROLE,
            "primary_question": "What is the highest-leverage improvement to the quantitative research process that this session can establish or validate?",
            "bottleneck": "Determine the current process bottleneck from durable repository evidence rather than from a predefined task list.",
            "objective": "Continuously improve the quantitative research process.",
            "scope": "Any repository work that materially improves research design, hypothesis generation, data quality, experimentation, validation, falsification, reproducibility, tooling, knowledge transfer, workflow, or research decision quality.",
            "out_of_scope": "Live trading, production execution, credential access, recursive workflow dispatch, fabricated evidence, and changes that weaken the immutable execution safeguards.",
            "deliverable": "A durable, verified process improvement or a high-value negative result that improves future research decisions.",
            "success_criterion": "Leave durable evidence that the chosen intervention materially improves research capability or resolves an important process uncertainty.",
            "stop_condition": "Stop when additional work has lower expected information value than preserving the current evidence and handing off the next unresolved bottleneck.",
            "verification_requirement": "Verify important claims with tests, reproduction, independent calculation, or another appropriate second check before treating them as established.",
            "evidence_basis": "Current repository state, research history, tests, experiments, and prior activation evidence.",
            "tests_preceding_process": False,
            "selection_basis": {
                "mode": "AUTONOMOUS",
                "reason": "The agent chooses the intervention; the controller supplies only the highest-order objective and safety boundary.",
                "compared_against": []
            },
            "preceding_agent": previous,
        }
        write_json(d / "agents" / f"agent_{agent_number:02d}_contract.json", contract)
        write_json(CURRENT_TASK, contract)

        status = load_json(d / "campaign_status.json")
        status["phase"] = f"AGENT_{agent_number:02d}_SELECTED"
        status["selected_tasks"].append({
            "agent_number": agent_number,
            "global_agent_number": global_agent_number,
            "task_id": task_id,
        })
        status["selection_history"].append({
            "agent_number": agent_number,
            "global_agent_number": global_agent_number,
            "role": AUTONOMOUS_ROLE,
            "task_id": task_id,
            "compared_against": [],
        })
        write_json(d / "campaign_status.json", status)
        return contract

    expected_role = ROLE_BY_AGENT.get(agent_number)
    if expected_role is None:
        raise SystemExit("agent number is outside legacy role map")
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
                if decision in {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN"}:
                    task["status"] = "RESOLVED"
                elif decision == "REJECT":
                    task["status"] = "REJECTED"
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
        "campaign_slot": agent_number,
        "global_agent_number": global_agent_number,
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
    status["selected_tasks"].append({
        "agent_number": agent_number,
        "global_agent_number": global_agent_number,
        "task_id": selected["task_id"],
    })
    status["selection_history"].append({
        "agent_number": agent_number,
        "global_agent_number": global_agent_number,
        "role": expected_role,
        "task_id": selected["task_id"],
        "compared_against": [x["task_id"] for x in candidates[:5]],
    })
    write_json(d / "campaign_status.json", status)
    return contract


def render_prompt(run_id: str, agent_number: int) -> str:
    contract = current_contract(run_id, agent_number)
    activation_environment = {
        "execution_layer": "GitHub Actions Linux runner executing a fresh Kilo Code CLI process",
        "repository_root": str(ROOT),
        "workflow": os.environ.get("GITHUB_WORKFLOW", "unknown"),
        "run_id": str(run_id),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "unknown"),
        "event_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "ref": os.environ.get("GITHUB_REF_NAME", "unknown"),
        "campaign_slot": agent_number,
        "global_agent_number": contract["global_agent_number"],
        "research_mode": "quantitative research and simulation only; no live trading or production execution",
    }

    prompt_path = ROOT / ".kilo" / "wakeup-prompt.md"
    base_prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""

    result_path = f"state/campaign/runs/{run_id}/agents/agent_{agent_number:02d}.json"
    return f"""
{base_prompt}

CURRENT ACTIVATION ENVIRONMENT:
{json.dumps(activation_environment, indent=2, sort_keys=True)}

AUTONOMOUS SESSION:
You decide the workflow, priorities, organization, experiments, validation, tooling, and process improvements.
Treat the contract below as the highest-order objective and safety boundary, not as a preselected research task.

CONTRACT:
{json.dumps(contract, indent=2, sort_keys=True)}

RESULT CONTRACT:
Before ending the session, write the machine-readable durable result to:
{result_path}

Required fields:
agent_number, global_agent_number, campaign_slot, role, task_id, objective, bottleneck, question,
action, changed, verified, unverified, observed_effect, uncertainty_targeted, uncertainty_reduced,
process_decision, research_result, decision, next, candidate_tasks, complexity_added, failure_class,
task_selection_observation.

Use only canonical status tokens already defined by the repository. Do not fabricate evidence.

IMMUTABLE SAFETY BOUNDARY:
Do not modify the supervisor/controller safeguards in .github/scripts/, the scheduled supervisor workflow .github/workflows/kilo-wakeup.yml, credential/secret controls, or recursion safeguards.
Do not dispatch another workflow or create uncontrolled self-triggering schedules.
"""

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("--run-id", required=True)
    p.add_argument("--agent-count", type=int, required=True, choices=(2, 10))
    p.add_argument("--identity-mode", choices=("PERSISTENT", "EPHEMERAL"), default="PERSISTENT")
    p.add_argument("--campaign-mode", choices=("LEGACY", "AUTONOMOUS"), default="LEGACY")

    p = sub.add_parser("select")
    p.add_argument("--run-id", required=True)
    p.add_argument("--agent-number", type=int, required=True)

    p = sub.add_parser("render-prompt")
    p.add_argument("--run-id", required=True)
    p.add_argument("--agent-number", type=int, required=True)

    args = ap.parse_args()
    if args.command == "init":
        init_campaign(args.run_id, args.agent_count, args.identity_mode, args.campaign_mode)
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
