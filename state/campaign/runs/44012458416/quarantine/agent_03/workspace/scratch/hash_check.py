"""Scratch diagnostic: validate this run's agent record with the controller's
own validation code, and report campaign-state hash integrity plus UTC time.

Run: python3 scratch/hash_check.py
"""
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/runner/work/X/X")
print("utc_now:", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

run = "44012458416"
agent = 2
agent_dir = root / "state/campaign/runs" / run / "agents"
before = json.loads((agent_dir / f"agent_{agent:02d}_before.json").read_text())
hashes = before["hashes"]
for rel in [
    "state/campaign/task_queue.json",
    "state/campaign/current_task.json",
    "state/worker_progress.md",
    f"state/campaign/runs/{run}/campaign_status.json",
]:
    p = root / rel
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    print(f"{rel}: same_as_before={h == hashes.get(rel)}")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = load_module("campaign_controller", root / ".github/scripts/campaign_controller.py")
vca = load_module("validate_campaign_agent", root / ".github/scripts/validate_campaign_agent.py")

contract = json.loads((agent_dir / f"agent_{agent:02d}_contract.json").read_text())
record = json.loads((agent_dir / f"agent_{agent:02d}.json").read_text())
failures = vca.validate_record(record, agent, contract)
print("record_failures:", failures)
for i, task in enumerate(record["candidate_tasks"]):
    try:
        cc.validate_task(task)
        print(f"candidate_{i} {task['task_id']}: OK")
    except SystemExit as exc:
        print(f"candidate_{i} {task['task_id']}: INVALID {exc}")