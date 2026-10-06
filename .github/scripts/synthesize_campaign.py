#!/usr/bin/env python3
"""Controller-owned final synthesis and durable handoff for one campaign."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SEVERITY = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_section(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n" + text.strip() + "\n")


def rank(task: dict) -> tuple:
    return (
        SEVERITY[task["expected_information_gain"]],
        SEVERITY[task["downstream_leverage"]],
        SEVERITY[task["falsification_power"]],
        SEVERITY[task["verifiability"]],
        SEVERITY[task["effort"]],
        SEVERITY[task["complexity"]],
        SEVERITY[task["execution_risk"]],
        task["task_id"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--independent-verify", required=True, choices=("success", "failure"))
    args = ap.parse_args()

    root = Path("state/campaign/runs") / str(args.run_id)
    status = load(root / "campaign_status.json")
    count = int(status["agent_count"])

    records = []
    validations = []
    missing_agents = []
    contradictions = []
    for n in range(1, count + 1):
        record_path = root / "agents" / f"agent_{n:02d}.json"
        validation_path = root / "agents" / f"agent_{n:02d}_validation.json"
        if not record_path.exists() or not validation_path.exists():
            missing_agents.append(n)
            continue
        record = load(record_path)
        validation = load(validation_path)
        records.append(record)
        validations.append(validation)
        if record.get("decision") == "UNVERIFIED":
            contradictions.append(f"agent_{n:02d}:UNVERIFIED")
        if record.get("task_selection_observation") == "WORSENED":
            contradictions.append(f"agent_{n:02d}:WORSENED_SELECTION_EFFECT")

    process_eval_path = root / "process_evaluation.json"
    process_eval = load(process_eval_path) if process_eval_path.exists() else {"campaign_process_outcome": "UNVERIFIED"}

    queue_path = Path("state/campaign/task_queue.json")
    queue_doc = load(queue_path)
    queue = queue_doc["tasks"]

    if len(records) == count:
        final_record = records[-1]
        final_task_id = final_record["task_id"]
        final_decision = final_record["decision"]
        for task in queue:
            if task["task_id"] == final_task_id:
                if final_decision in {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN", "NO_SUBSTANTIVE_ACTION"}:
                    task["status"] = "RESOLVED"
                elif final_decision == "REJECT":
                    task["status"] = "REJECTED"
                else:
                    task["status"] = "DEFERRED"

    known = {t["task_id"] for t in queue}
    for record in records:
        validation = next((v for v in validations if v.get("agent_number") == record.get("agent_number")), None)
        if not validation or validation.get("status") != "VERIFIED_SESSION":
            continue
        for candidate in record.get("candidate_tasks", []):
            if candidate["task_id"] not in known:
                candidate = dict(candidate)
                candidate["status"] = "OPEN"
                queue.append(candidate)
                known.add(candidate["task_id"])

    next_candidates = sorted([t for t in queue if t["status"] == "OPEN"], key=rank)
    next_task = next_candidates[0]["task_id"] if next_candidates else "NONE"
    write(queue_path, {
        "schema_version": queue_doc.get("schema_version", 1),
        "selection_principle": queue_doc.get("selection_principle", "qualitative_evidence_backed_priority"),
        "tasks": queue,
    })

    validated_info = [
        {
            "agent_number": r["agent_number"],
            "uncertainty_targeted": r["uncertainty_targeted"],
            "uncertainty_reduced": r["uncertainty_reduced"],
        }
        for r in records
        if r.get("uncertainty_reduced") not in {"", "none", "none demonstrated", None}
        and next((v for v in validations if v.get("agent_number") == r.get("agent_number")), {}).get("status") == "VERIFIED_SESSION"
    ]
    process_improvement = [
        r["agent_number"] for r in records
        if r.get("role") == "LEARNING_PROCESS"
        and r.get("process_decision") == "IMPROVE"
        and r.get("decision") in {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN"}
    ]
    research_progress = [
        r["agent_number"] for r in records
        if r.get("role") == "HIGHER_ORDER_OBJECTIVE"
        and r.get("decision") in {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN"}
        and r.get("uncertainty_reduced") not in {"", "none", "none demonstrated", None}
    ]
    rejected = [
        r["agent_number"] for r in records
        if r.get("decision") in {"REJECT", "VERIFIED_NEGATIVE_RESULT"}
    ]
    unresolved = [r["agent_number"] for r in records if r.get("decision") == "UNVERIFIED"]
    complexity = [
        {"agent_number": r["agent_number"], "complexity_added": r.get("complexity_added")}
        for r in records
    ]

    substantive_failures = [
        v for v in validations
        if v.get("status") != "VERIFIED_SESSION" and v.get("substantive_changed")
    ]
    process_verified = process_eval.get("campaign_process_outcome") != "UNVERIFIED"
    persistence = (
        "MAIN"
        if (
            args.independent_verify == "success"
            and process_verified
            and not missing_agents
            and not substantive_failures
            and not any(v.get("protected_paths") for v in validations)
        )
        else "RECOVERY_BRANCH"
    )
    outcome = (
        "BLOCKED"
        if missing_agents
        else "COMPLETE"
        if persistence == "MAIN" and not unresolved
        else "PARTIAL"
        if persistence == "MAIN"
        else "QUARANTINED"
    )

    synthesis = {
        "run_id": str(args.run_id),
        "agent_count": count,
        "validated_information_gained": validated_info,
        "validated_process_improvement": process_improvement,
        "validated_research_progress": research_progress,
        "rejected_hypotheses_or_negative_results": rejected,
        "unresolved_uncertainty": unresolved,
        "new_complexity_introduced": complexity,
        "contradictory_or_unverified_evidence": contradictions,
        "missing_agents": missing_agents,
        "focused_task_selection_observations": {str(r["agent_number"]): r.get("task_selection_observation") for r in records},
        "odd_even_process_pairs": process_eval.get("odd_even_process_pairs", []),
        "campaign_process_outcome": process_eval.get("campaign_process_outcome", "UNVERIFIED"),
        "single_highest_value_next_task": next_task,
        "single_highest_value_task_candidates": [t["task_id"] for t in next_candidates[:5]],
        "independent_verification": args.independent_verify,
        "persistence_decision": persistence,
        "terminal_outcome": outcome,
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    write(root / "final_synthesis.json", synthesis)

    activation = {
        "activation_id": str(args.run_id),
        "status": outcome,
        "objective": "Sequential focused-task campaign to improve validated learning efficiency and higher-order quantitative-research capability.",
        "phase": "COMPLETE",
        "changed": sorted({p for r in records for p in r.get("changed", [])}),
        "verified": sorted({x for r in records for x in r.get("verified", [])})[:100],
        "unverified": sorted({x for r in records for x in r.get("unverified", [])})[:100],
        "next": next_task,
        "research_conclusion": [r.get("research_result") for r in records if r.get("role") == "HIGHER_ORDER_OBJECTIVE" and r.get("research_result")],
        "process_conclusion": [r.get("observed_effect") for r in records if r.get("role") == "LEARNING_PROCESS" and r.get("observed_effect")],
        "campaign_outcome": outcome,
        "persistence_decision": persistence,
    }
    write(Path("state/activation_status.json"), activation)

    append_section(
        Path("state/LEARNING_STATE.md"),
        f"""## Sequential campaign {args.run_id}
- **Validated process improvement:** agents {process_improvement or 'NONE'}
- **Validated research progress:** agents {research_progress or 'NONE'}
- **Validated negative/rejected evidence:** agents {rejected or 'NONE'}
- **Unresolved uncertainty:** agents {unresolved or 'NONE'}
- **Odd/even process observations:** {json.dumps(synthesis["odd_even_process_pairs"], sort_keys=True)}
- **Task-selection observation:** {json.dumps(synthesis["focused_task_selection_observations"], sort_keys=True)}
- **Highest-value unresolved next task:** {next_task}
- **New complexity introduced:** {json.dumps(complexity, sort_keys=True)}
- **Campaign persistence decision:** {persistence}
- **Contradictory/unverified evidence:** {contradictions or 'NONE'}
"""
    )

    status["status"] = "SYNTHESIZED"
    status["phase"] = "SYNTHESIS"
    status["completed_agents"] = [r["agent_number"] for r in records]
    status["terminal_outcome"] = outcome
    status["persistence_decision"] = persistence
    status["next_task"] = next_task
    write(root / "campaign_status.json", status)
    print(json.dumps(synthesis, indent=2, sort_keys=True))
    return 0 if outcome in {"COMPLETE", "PARTIAL", "QUARANTINED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
