#!/usr/bin/env python3
import json

try:
    with open('/home/runner/work/X/X/state/campaign/runs/38057185488/agents/agent_03.json', 'r') as f:
        data = json.load(f)
    print("JSON syntax is valid")
    
    # Check if all required fields are present
    required_fields = [
        'agent_number', 'role', 'task_id', 'objective', 'bottleneck', 'question',
        'action', 'changed', 'verified', 'unverified', 'observed_effect',
        'uncertainty_targeted', 'uncertainty_reduced', 'process_decision',
        'research_result', 'decision', 'next', 'candidate_tasks',
        'complexity_added', 'failure_class', 'task_selection_observation'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"Missing fields: {missing_fields}")
    else:
        print("All required fields are present")
        
except json.JSONDecodeError as e:
    print(f"JSON syntax error: {e}")
except Exception as e:
    print(f"Error: {e}")