import json

# Load the agent_06.json file
with open('state/campaign/runs/38012513908/agents/agent_06.json', 'r') as f:
    data = json.load(f)

# Check required fields
required_fields = [
    'agent_number', 'role', 'task_id', 'objective', 'bottleneck', 
    'question', 'action', 'changed', 'verified', 'unverified',
    'observed_effect', 'uncertainty_targeted', 'uncertainty_reduced',
    'process_decision', 'research_result', 'decision', 'next',
    'candidate_tasks', 'complexity_added', 'failure_class'
]

print('Checking required fields:')
for field in required_fields:
    if field in data:
        print(f'  ✓ {field}: PRESENT')
    else:
        print(f'  ✗ {field}: MISSING')

print(f'\nCandidate tasks count: {len(data["candidate_tasks"])}')
print(f'Next action: {data["next"]}')
print(f'Failure class: {data["failure_class"]}')
print(f'Decision: {data["decision"]}')