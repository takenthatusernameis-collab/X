#!/usr/bin/env python3
"""Deterministic terminal classifier for the sequential campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ALLOWED = {"COMPLETE", "PARTIAL", "QUARANTINED", "FAILED", "BLOCKED"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    path = Path("state/campaign/runs") / str(args.run_id) / "final_synthesis.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    outcome = report["terminal_outcome"]
    if outcome not in ALLOWED:
        raise SystemExit(f"invalid terminal outcome: {outcome}")
    intended_persistence = report["persistence_decision"]
    if intended_persistence not in {"MAIN", "RECOVERY_BRANCH"}:
        raise SystemExit(f"invalid persistence decision: {intended_persistence}")
    actual_persistence = os.environ.get("PERSISTENCE_STATUS", "")
    if actual_persistence not in {"MAIN", "RECOVERY_BRANCH"}:
        raise SystemExit(
            "persistence was not verified; "
            f"PERSISTENCE_STATUS={actual_persistence or 'unset'}"
        )
    print(f"CAMPAIGN_OUTCOME={outcome}")
    print(f"PERSISTENCE_DECISION={intended_persistence}")
    print(f"PERSISTENCE_STATUS={actual_persistence}")
    recovery_branch = os.environ.get("PERSISTENCE_RECOVERY_BRANCH", "")
    if actual_persistence == "RECOVERY_BRANCH":
        if not recovery_branch:
            raise SystemExit("recovery branch persistence is not named")
        print(f"PERSISTENCE_RECOVERY_BRANCH={recovery_branch}")
        print("::warning::Checkpoint is durable on a recovery branch, not the default branch; the next task should reconcile it.")
    print(f"HIGHEST_VALUE_NEXT_TASK={report['single_highest_value_next_task']}")
    if outcome == "QUARANTINED":
        print("::warning::Campaign was quarantined after preserving its validated evidence; this is a controlled research outcome, not a workflow infrastructure failure.")
        return 0
    return 0 if outcome in {"COMPLETE", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
