import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "validate_campaign_agent", root / ".github" / "scripts" / "validate_campaign_agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

contract = json.loads((root / "state/campaign/runs/45721512242/agents/agent_02_contract.json").read_text())
record = json.loads((root / "state/campaign/runs/45721512242/agents/agent_02.json").read_text())

failures = mod.validate_record(record, 2, contract)
print("record failures:", failures)

for i, t in enumerate(record["candidate_tasks"]):
    try:
        mod.controller.validate_task(t)
        print(f"candidate_task_{i}: OK")
    except SystemExit as exc:
        print(f"candidate_task_{i}: INVALID -> {exc}")

# next must be exactly one non-empty line
nlines = [x for x in record["next"].splitlines() if x.strip()]
print("next line count:", len(nlines))
print("identity:", record["agent_number"], record["campaign_slot"], record["global_agent_number"])
print("enums:", record["decision"], record["process_decision"], record["task_selection_observation"], record["failure_class"])