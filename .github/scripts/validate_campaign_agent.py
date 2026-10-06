#!/usr/bin/env python3
"""Controller-side validation, quarantine, and fallback classification for one campaign agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import campaign_controller as controller  # noqa: E402


PROTECTED_PREFIXES = (".github/workflows/", ".github/scripts/", ".kilo/")
PROTECTED_FILES = {
    "AGENTS.md",
    "ENTERPRISE.md",
    "MANUAL_SETUP.md",
    "PERSISTENCE_POLICY.md",
}
DECISIONS = controller.DECISIONS
PROCESS_DECISIONS = controller.PROCESS_DECISIONS
OBS = controller.SELECTION_OBSERVATIONS
FAILURES = controller.FAILURE_CLASSES


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return p.returncode, p.stdout


def sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_changed() -> list[str]:
    _, out = run(["git", "diff", "--name-only"])
    _, out2 = run(["git", "ls-files", "--others", "--exclude-standard"])
    return sorted(set(x for x in out.splitlines() if x) | set(x for x in out2.splitlines() if x))


def fallback_record(agent: int, contract: dict, runtime: dict | None, reason: str) -> dict:
    failure = runtime.get("failure_class", "UNKNOWN") if runtime else "VALIDATION_FAILURE"
    return {
        "agent_number": agent,
        "role": contract["role"],
        "task_id": contract["task_id"],
        "objective": contract["objective"],
        "bottleneck": contract["bottleneck"],
        "question": contract["primary_question"],
        "action": "session did not produce a controller-validated durable agent record",
        "changed": [],
        "verified": [],
        "unverified": [reason],
        "observed_effect": "unverified",
        "uncertainty_targeted": contract["primary_question"],
        "uncertainty_reduced": "none demonstrated",
        "process_decision": "UNVERIFIED",
        "research_result": "UNVERIFIED",
        "decision": "UNVERIFIED",
        "next": "inspect the preserved runtime evidence and re-rank the unresolved task queue",
        "candidate_tasks": [],
        "complexity_added": "UNVERIFIED",
        "failure_class": failure if failure in FAILURES else "UNKNOWN",
        "task_selection_observation": "UNVERIFIED",
    }


def validate_record(record: dict, agent: int, contract: dict) -> list[str]:
    required = [
        "agent_number", "role", "task_id", "objective", "bottleneck", "question",
        "action", "changed", "verified", "unverified", "observed_effect",
        "uncertainty_targeted", "uncertainty_reduced", "process_decision",
        "research_result", "decision", "next", "candidate_tasks",
        "complexity_added", "failure_class", "task_selection_observation",
    ]
    failures = [x for x in required if x not in record]
    if failures:
        return [f"missing_record_fields:{failures}"]
    if record["agent_number"] != agent:
        failures.append("agent_number_mismatch")
    if record["task_id"] != contract["task_id"]:
        failures.append("task_id_mismatch")
    if record["role"] != contract["role"]:
        failures.append("role_mismatch")
    if record["decision"] not in DECISIONS:
        failures.append("invalid_decision")
    if record["process_decision"] not in PROCESS_DECISIONS:
        failures.append("invalid_process_decision")
    if record["task_selection_observation"] not in OBS:
        failures.append("invalid_task_selection_observation")
    if record["failure_class"] not in FAILURES:
        failures.append("invalid_failure_class")
    if not isinstance(record["changed"], list) or not isinstance(record["verified"], list) or not isinstance(record["unverified"], list):
        failures.append("list_fields_invalid")
    if not isinstance(record["candidate_tasks"], list):
        failures.append("candidate_tasks_not_list")
    else:
        for idx, task in enumerate(record["candidate_tasks"]):
            if not isinstance(task, dict):
                failures.append(f"candidate_task_{idx}_not_object")
                continue
            try:
                controller.validate_task(task)
            except SystemExit as exc:
                failures.append(f"candidate_task_{idx}_invalid:{exc}")
    if not isinstance(record["next"], str) or not record["next"].strip():
        failures.append("next_missing")
    if len([x for x in record["next"].splitlines() if x.strip()]) != 1:
        failures.append("next_must_be_one_bounded_action")
    return failures


def quarantine_workspace(agent: int, run_root: Path, failures: list[str]) -> dict[str, object]:
    quarantine = run_root / "quarantine" / f"agent_{agent:02d}"
    quarantine.mkdir(parents=True, exist_ok=True)
    rc, diff = run(["git", "diff", "--binary"])
    (quarantine / "tracked_worktree.diff").write_text(diff, encoding="utf-8")
    _, cached = run(["git", "diff", "--cached", "--binary"])
    (quarantine / "tracked_index.diff").write_text(cached, encoding="utf-8")
    _, status = run(["git", "status", "--porcelain=v1"])
    (quarantine / "status.txt").write_text(status, encoding="utf-8")
    (quarantine / "failure_reasons.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")

    _, untracked_text = run(["git", "ls-files", "--others", "--exclude-standard"])
    moved: list[str] = []
    for rel in [x for x in untracked_text.splitlines() if x]:
        if rel.startswith("state/campaign/") or rel == "state/worker_progress.md":
            continue
        source = Path(rel)
        if not source.exists():
            continue
        destination = quarantine / "workspace" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append(rel)
    (quarantine / "untracked_moved.json").write_text(json.dumps(moved, indent=2) + "\n", encoding="utf-8")

    run(["git", "restore", "--staged", "--worktree", "--", "."])
    return {"path": str(quarantine), "moved_untracked": moved, "diff_return_code": rc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--agent-number", required=True, type=int)
    ap.add_argument("--worker-exit-code", type=int, default=1)
    args = ap.parse_args()

    run_root = Path("state/campaign/runs") / str(args.run_id)
    agent_dir = run_root / "agents"
    contract = json.loads((agent_dir / f"agent_{args.agent_number:02d}_contract.json").read_text(encoding="utf-8"))
    record_path = agent_dir / f"agent_{args.agent_number:02d}.json"
    runtime_path = agent_dir / f"agent_{args.agent_number:02d}_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None

    before_path = agent_dir / f"agent_{args.agent_number:02d}_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {"hashes": {}}
    before_hashes = before.get("hashes", {})
    after_paths = current_changed()

    touched = [path for path in after_paths if before_hashes.get(path) != sha(Path(path))]
    protected = sorted(
        p for p in after_paths
        if p in PROTECTED_FILES or any(p.startswith(prefix) for prefix in PROTECTED_PREFIXES)
    )
    failures: list[str] = []
    warnings: list[str] = []

    rc, out = run(["git", "diff", "--check"])
    if rc != 0:
        failures.append("git_diff_check_failed")
        warnings.append(out[-4000:])

    python_paths = [p for p in touched if p.endswith(".py") and Path(p).exists()]
    if python_paths:
        rc, out = run([sys.executable, "-m", "py_compile", *python_paths])
        if rc != 0:
            failures.append("python_compile_failed")
            warnings.append(out[-4000:])

    if any(p.startswith(("research/", "tests/", "examples/")) for p in touched):
        rc, out = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"])
        if rc != 0:
            failures.append("regression_suite_failed")
            warnings.append(out[-6000:])

    raw_record = None
    if record_path.exists():
        try:
            raw_record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"record_unreadable:{type(exc).__name__}")
    if raw_record is None:
        raw_record = fallback_record(
            args.agent_number, contract, runtime, "required agent record missing or unreadable"
        )
        record_path.write_text(json.dumps(raw_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        warnings.append("controller_created_UNVERIFIED_fallback")
    else:
        record_failures = validate_record(raw_record, args.agent_number, contract)
        failures.extend(record_failures)
        if record_failures:
            invalid_path = agent_dir / f"agent_{args.agent_number:02d}_invalid_record.json"
            invalid_path.write_text(json.dumps(raw_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raw_record = fallback_record(
                args.agent_number, contract, runtime, "agent record failed controller validation"
            )
            raw_record["unverified"].append("; ".join(record_failures))
            record_path.write_text(json.dumps(raw_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            warnings.append("invalid_agent_record_replaced_with_UNVERIFIED_fallback")

    expected_record = f"state/campaign/runs/{args.run_id}/agents/agent_{args.agent_number:02d}.json"
    unexpected_campaign_paths = [
        p for p in touched
        if p.startswith("state/campaign/")
        and p not in {expected_record}
        and not p.startswith(f"state/campaign/runs/{args.run_id}/agents/agent_{args.agent_number:02d}_")
    ]
    if unexpected_campaign_paths:
        failures.append("controller_state_modified")
        warnings.extend(f"unexpected_campaign_path:{p}" for p in unexpected_campaign_paths)

    if protected:
        failures.append("protected_control_plane_modified")
    if args.worker_exit_code != 0:
        warnings.append(f"worker_exit_code={args.worker_exit_code}")

    status = "VERIFIED_SESSION" if not failures and args.worker_exit_code == 0 else "UNVERIFIED_SESSION"
    validation = {
        "agent_number": args.agent_number,
        "worker_exit_code": args.worker_exit_code,
        "touched_paths": touched,
        "protected_paths": protected,
        "status": status,
        "warnings": warnings,
        "failures": failures,
        "substantive_changed": [
            p for p in touched if not p.startswith("state/campaign/") and p != "state/worker_progress.md"
        ],
        "runtime": runtime,
        "quarantine": None,
    }

    if status != "VERIFIED_SESSION":
        validation["quarantine"] = quarantine_workspace(args.agent_number, run_root, failures)

    out_path = agent_dir / f"agent_{args.agent_number:02d}_validation.json"
    out_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if status == "VERIFIED_SESSION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
