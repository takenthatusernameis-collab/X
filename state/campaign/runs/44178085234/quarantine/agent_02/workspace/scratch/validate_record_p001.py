#!/usr/bin/env python3
"""Read-only replication of the controller's agent-record validation for this session.

Does not write, quarantine, or invoke git.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import campaign_controller as cc  # noqa: E402
import validate_campaign_agent as vca  # noqa: E402

RUN = ROOT / "state" / "campaign" / "runs" / "44178085234"


def main() -> int:
    contract = json.loads((RUN / "agents" / "agent_01_contract.json").read_text(encoding="utf-8"))
    record = json.loads((RUN / "agents" / "agent_01.json").read_text(encoding="utf-8"))
    failures = vca.validate_record(record, 1, contract)
    print(f"record_validation_failures={failures}")
    print(f"decision={record['decision']} process_decision={record['process_decision']} "
          f"task_selection_observation={record['task_selection_observation']} "
          f"failure_class={record['failure_class']}")
    print(f"next_lines={len([x for x in record['next'].splitlines() if x.strip()])}")
    for idx, task in enumerate(record["candidate_tasks"]):
        try:
            cc.validate_task(task)
            print(f"candidate_task_{idx}_valid={task['task_id']} status={task['status']}")
        except SystemExit as exc:
            print(f"candidate_task_{idx}_invalid={exc}")
            failures.append(f"candidate_task_{idx}")
    print(f"OVERALL_OK={not failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())