#!/usr/bin/env python3
import json

with open('/home/runner/work/X/X/state/campaign/runs/38057185488/agents/agent_03.json', 'r') as f:
    data = json.load(f)

print("Validating JSON structure...")
print(f"agent_number: {data.get('agent_number', 'MISSING')}")
print(f"campaign_slot: {data.get('campaign_slot', 'MISSING')}")
print(f"global_agent_number: {data.get('global_agent_number', 'MISSING')}")

print(f"\nDecision: {data.get('decision', 'MISSING')}")
print(f"Process decision: {data.get('process_decision', 'MISSING')}")
print(f"Task selection observation: {data.get('task_selection_observation', 'MISSING')}")
print(f"Failure class: {data.get('failure_class', 'MISSING')}")

print(f"\nChanged array length: {len(data.get('changed', []))}")
print(f"Verified array length: {len(data.get('verified', []))}")
print(f"Unverified array length: {len(data.get('unverified', []))}")
print(f"Candidate tasks array length: {len(data.get('candidate_tasks', []))}")

# Check if candidate_tasks is an array
if isinstance(data.get('candidate_tasks'), list):
    print("✓ candidate_tasks is an array (correct)")
else:
    print("✗ candidate_tasks is not an array (incorrect)")

print("\nJSON validation completed successfully!")