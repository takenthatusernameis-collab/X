#!/usr/bin/env python3
"""Deterministic terminal classifier for the sequential campaign."""

from __future__ import annotations

import argparse
import json
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
    persistence = report["persistence_decision"]
    if persistence not in {"MAIN", "RECOVERY_BRANCH"}:
        raise SystemExit(f"invalid persistence decision: {persistence}")
    print(f"CAMPAIGN_OUTCOME={outcome}")
    print(f"PERSISTENCE_DECISION={persistence}")
    print(f"HIGHEST_VALUE_NEXT_TASK={report['single_highest_value_next_task']}")
    if outcome == "QUARANTINED":
        print("::warning::Campaign was quarantined after preserving its validated evidence; this is a controlled research outcome, not a workflow infrastructure failure.")
        return 0
    return 0 if outcome in {"COMPLETE", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
