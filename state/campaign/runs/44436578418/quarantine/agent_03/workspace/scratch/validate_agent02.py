#!/usr/bin/env python3
"""Scratch: validate agent_02.json with the controller's own validators."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import validate_campaign_agent as vca  # noqa: E402
import campaign_controller as controller  # noqa: E402

run_id = "44436578418"
agent = 2
agent_dir = ROOT / "state" / "campaign" / "runs" / run_id / "agents"
record = json.loads((agent_dir / f"agent_{agent:02d}.json").read_text(encoding="utf-8"))
contract = json.loads((agent_dir / f"agent_{agent:02d}_contract.json").read_text(encoding="utf-8"))

failures = vca.validate_record(record, agent, contract)
print("validate_record failures:", failures)

for i, task in enumerate(record.get("candidate_tasks", [])):
    try:
        controller.validate_task(task)
        print(f"candidate_task[{i}] {task['task_id']}: OK")
    except SystemExit as exc:
        print(f"candidate_task[{i}] {task['task_id']}: INVALID -> {exc}")

print("RESULT:", "PASS" if not failures else "FAIL")
sys.exit(1 if failures else 0)