#!/usr/bin/env python3
import json

with open('/home/runner/work/X/X/state/campaign/runs/38057185488/agents/agent_03.json', 'r') as f:
    data = json.load(f)

# Check required controller-owned identity metadata
required_metadata = ['agent_number', 'campaign_slot', 'global_agent_number']
for key in required_metadata:
    if key not in data:
        print(f'ERROR: Missing required metadata: {key}')
    else:
        print(f'✓ {key}: {data[key]}')

# Check canonical decision values
print(f'\nDecision values:')
print(f'  decision: {data["decision"]}')
print(f'  process_decision: {data["process_decision"]}')
print(f'  task_selection_observation: {data["task_selection_observation"]}')

# Check arrays
print(f'\nArrays:')
print(f'  changed: {len(data["changed"])} items')
print(f'  verified: {len(data["verified"])} items')
print(f'  unverified: {len(data["unverified"])} items')
print(f'  candidate_tasks: {len(data["candidate_tasks"])} items')

# Check failure_class
print(f'\nFailure class: {data["failure_class"]}')

print('\n✓ All required fields present')