import json
d = json.load(open('state/campaign/runs/37429979631/agents/agent_06.json'))
print('valid JSON, top keys:', list(d.keys()))
print('agent_number:', d['agent_number'], '| global_agent_number:', d['global_agent_number'], '| campaign_slot:', d['campaign_slot'], '| task_id:', d['task_id'])
print('task_selection_observation:', d['task_selection_observation'], '| research_result:', d['research_result'], '| failure_class:', d['failure_class'])
