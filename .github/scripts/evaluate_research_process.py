#!/usr/bin/env python3
"""Independent controller-side evaluation of the sequential campaign process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    root = Path("state/campaign/runs") / str(args.run_id)
    status = load(root / "campaign_status.json")
    count = int(status["agent_count"])
    records = []
    validations = []
    missing: list[int] = []

    for n in range(1, count + 1):
        rp = root / "agents" / f"agent_{n:02d}.json"
        vp = root / "agents" / f"agent_{n:02d}_validation.json"
        if rp.exists():
            records.append(load(rp))
        else:
            missing.append(n)
        if vp.exists():
            validations.append(load(vp))
        else:
            missing.append(n)

    verified_sessions = sum(v.get("status") == "VERIFIED_SESSION" for v in validations)
    process_pairs = []
    for even in range(2, count + 1, 2):
        odd = even - 1
        odd_record = next((r for r in records if r.get("agent_number") == odd), None)
        even_record = next((r for r in records if r.get("agent_number") == even), None)
        if odd_record and even_record:
            process_pairs.append({
                "odd_agent": odd,
                "odd_process_decision": odd_record.get("process_decision"),
                "odd_task": odd_record.get("task_id"),
                "even_agent": even,
                "even_task": even_record.get("task_id"),
                "even_observation": even_record.get("task_selection_observation"),
                "even_uncertainty_reduced": even_record.get("uncertainty_reduced"),
            })

    validated_process = [
        r for r in records
        if r.get("role") in {"LEARNING_PROCESS", "AUTONOMOUS_RESEARCH"}
        and r.get("decision") in {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN"}
        and r.get("process_decision") in {"IMPROVE", "RETAIN", "REJECT"}
    ]
    validated_research = [
        r for r in records
        if r.get("role") in {"HIGHER_ORDER_OBJECTIVE", "AUTONOMOUS_RESEARCH"}
        and r.get("decision") in {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN"}
        and r.get("uncertainty_reduced") not in {"", "none", "none demonstrated", None}
    ]
    negative = [r for r in records if r.get("decision") == "VERIFIED_NEGATIVE_RESULT"]
    unresolved = [r for r in records if r.get("decision") == "UNVERIFIED"]
    no_action = [r for r in records if r.get("decision") == "NO_SUBSTANTIVE_ACTION"]
    selection_observations = {
        str(r.get("agent_number")): r.get("task_selection_observation")
        for r in records
    }

    contradictions = []
    for r in records:
        if r.get("decision") == "UNVERIFIED":
            contradictions.append(f"agent_{r.get('agent_number'):02d}:UNVERIFIED")
        if r.get("task_selection_observation") == "WORSENED":
            contradictions.append(f"agent_{r.get('agent_number'):02d}:WORSENED_SELECTION_EFFECT")

    if missing or verified_sessions != count:
        outcome = "UNVERIFIED"
    elif validated_process or validated_research or negative:
        outcome = "VERIFIED_PROGRESS"
    else:
        outcome = "NO_SUBSTANTIVE_ACTION"

    report = {
        "campaign_process_outcome": outcome,
        "agent_count": count,
        "controller_verified_sessions": verified_sessions,
        "unverified_sessions": len(unresolved) + len(missing),
        "validated_process_improvement": [r["agent_number"] for r in validated_process],
        "validated_research_progress": [r["agent_number"] for r in validated_research],
        "rejected_or_negative_hypotheses": [r["agent_number"] for r in negative],
        "unresolved_uncertainty": [r["agent_number"] for r in unresolved],
        "no_substantive_action": [r["agent_number"] for r in no_action],
        "task_selection_observations": selection_observations,
        "odd_even_process_pairs": process_pairs,
        "contradictory_evidence": contradictions,
        "missing_agents": missing,
    }
    (root / "process_evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(os.environ.get("GITHUB_ENV", "/dev/null"), "a", encoding="utf-8") as env:
        env.write(f"CAMPAIGN_PROCESS_OUTCOME={outcome}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if outcome != "UNVERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
