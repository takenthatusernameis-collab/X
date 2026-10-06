#!/usr/bin/env python3
"""Independent controller-side evaluation of research-process quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TERMINAL_PROCESS = {"VERIFIED_PROGRESS", "VERIFIED_REPAIR", "NO_SUBSTANTIVE_ACTION", "UNVERIFIED"}


def sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--campaign", default="state/campaign/campaign_status.json")
    ap.add_argument("--verify-outcome", required=True)
    ap.add_argument("--worker-outcome", required=True)
    ap.add_argument("--liveness", required=True)
    ap.add_argument("--output", default="state/campaign/process_evaluation.json")
    args = ap.parse_args()

    baseline = load_json(Path(args.baseline))
    campaign_path = Path(args.campaign)
    campaign = load_json(campaign_path) if campaign_path.exists() else {}

    successes = int(campaign.get("success_count", 0) or 0)
    failures = int(campaign.get("failure_count", 0) or 0)
    agent_count = int(campaign.get("agent_count", 0) or 0)

    current = {
        "learning_state": sha(Path("state/LEARNING_STATE.md")),
        "research_state": sha(Path("state/STATE.md")),
        "activation_receipt": sha(Path("state/activation_status.json")),
        "worker_progress": sha(Path("state/worker_progress.md")),
    }

    changes = set()
    try:
        changes.update(x for x in os.popen("git diff --name-only").read().splitlines() if x)
        changes.update(x for x in os.popen("git ls-files --others --exclude-standard").read().splitlines() if x)
    except Exception:
        pass

    learning_changed = current["learning_state"] != baseline.get("learning_state")
    research_state_changed = current["research_state"] != baseline.get("research_state")
    worker_progress_changed = current["worker_progress"] != baseline.get("worker_progress")
    research_code_changed = any(
        p.startswith("research/") or p.startswith("tests/") or p.startswith("examples/")
        for p in changes
    )
    durable_state_changed = learning_changed or research_state_changed or research_code_changed

    validation_files = sorted(Path("state/campaign").glob("agent_*_validation.json"))
    validation_reports = []
    validation_failures = 0
    validation_verified = 0
    checkpoint_transitions = 0
    for path in validation_files:
        report = load_json(path)
        validation_reports.append(report)
        if report.get("status") == "VERIFIED_SESSION":
            validation_verified += 1
        else:
            validation_failures += 1
        if report.get("checkpoint_changed"):
            checkpoint_transitions += 1

    reasons: list[str] = []
    if args.verify_outcome != "success":
        process = "UNVERIFIED"
        reasons.append("independent post-worker verification did not succeed")
    elif successes <= 0:
        process = "UNVERIFIED"
        reasons.append("no campaign agent reached controller-verified session status")
    elif learning_changed or research_state_changed:
        process = "VERIFIED_PROGRESS"
        reasons.append("durable research state changed beyond the activation receipt and logs")
    elif research_code_changed and validation_verified > 0 and validation_failures == 0:
        process = "VERIFIED_REPAIR"
        reasons.append("research/software capability changed and all observed campaign sessions passed controller validation")
    else:
        process = "NO_SUBSTANTIVE_ACTION"
        reasons.append("execution and verification completed without independently evidenced durable research-state progress")

    if failures:
        reasons.append(f"campaign reported {failures} failed or timeout sessions")
    if checkpoint_transitions == 0:
        reasons.append("no post-start semantic worker checkpoint transition was independently observed")
    if worker_progress_changed:
        reasons.append("worker progress state changed during the activation")

    report = {
        "process_outcome": process,
        "agent_count": agent_count,
        "campaign_success_count": successes,
        "campaign_failure_count": failures,
        "controller_verified_agent_sessions": validation_verified,
        "controller_failed_agent_sessions": validation_failures,
        "semantic_checkpoint_transitions": checkpoint_transitions,
        "learning_state_changed": learning_changed,
        "research_state_changed": research_state_changed,
        "research_code_changed": research_code_changed,
        "worker_progress_changed": worker_progress_changed,
        "independent_verify_outcome": args.verify_outcome,
        "worker_outcome": args.worker_outcome,
        "liveness": args.liveness,
        "reasons": reasons,
        "baseline": baseline,
        "current_hashes": current,
        "validation_reports": validation_reports,
    }

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"RESEARCH_PROCESS_OUTCOME={process}")
    print(f"RESEARCH_PROCESS_REASON={' | '.join(reasons)}")

    with open(os.environ.get("GITHUB_ENV", "/dev/null"), "a", encoding="utf-8") as env:
        env.write(f"RESEARCH_PROCESS_OUTCOME={process}\n")
        env.write(f"RESEARCH_PROCESS_REASON={' | '.join(reasons)}\n")

    with open(os.environ.get("GITHUB_STEP_SUMMARY", "/dev/null"), "a", encoding="utf-8") as summary:
        summary.write("### Independent research-process evaluation\n")
        summary.write(f"- outcome: **{process}**\n")
        summary.write(f"- controller-verified agent sessions: {validation_verified}/{agent_count}\n")
        summary.write(f"- semantic checkpoint transitions: {checkpoint_transitions}\n")
        summary.write(f"- learning state changed: {learning_changed}\n")
        summary.write(f"- research state changed: {research_state_changed}\n")
        summary.write(f"- research/software paths changed: {research_code_changed}\n")
        for reason in reasons:
            summary.write(f"- reason: {reason}\n")

    return 0 if process != "UNVERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
