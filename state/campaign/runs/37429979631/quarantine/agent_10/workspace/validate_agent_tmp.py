import json
p = '/home/runner/work/X/X/state/campaign/runs/37429979631/agents/agent_10.json'
d = json.load(open(p))
print('valid JSON, keys:', sorted(d.keys()))
print('agent_number:', d['agent_number'], '| campaign_slot:', d['campaign_slot'], '| global_agent_number:', d['global_agent_number'])
print('research_result:', d['research_result'], '| decision:', d['decision'], '| process_decision:', d['process_decision'], '| failure_class:', d['failure_class'])
print('candidate_tasks:', len(d['candidate_tasks']))
