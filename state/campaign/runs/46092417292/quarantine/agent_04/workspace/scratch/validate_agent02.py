import json
p = 'state/campaign/runs/46092417292/agents/agent_02.json'
d = json.load(open(p))
req = ['agent_number','role','task_id','objective','bottleneck','question','action','changed','verified','unverified','observed_effect','uncertainty_targeted','uncertainty_reduced','process_decision','research_result','decision','next','candidate_tasks','complexity_added','failure_class','task_selection_observation','campaign_slot','global_agent_number']
missing = [k for k in req if k not in d]
assert not missing, 'missing: %s' % missing
assert d['agent_number'] == 2 and d['campaign_slot'] == 2 and d['global_agent_number'] == 2
assert d['decision'] in ['NO_SUBSTANTIVE_ACTION','REJECT','RETAIN','UNVERIFIED','USEFUL_CHANGE','VERIFIED_NEGATIVE_RESULT']
assert d['process_decision'] in ['IMPROVE','REJECT','RETAIN','UNVERIFIED']
assert d['task_selection_observation'] in ['IMPROVED','UNCHANGED','UNVERIFIED','WORSENED']
assert d['failure_class'] in ['LIVENESS','NONE','PERSISTENCE','RESEARCH_EXECUTION','SCOPE_VIOLATION','TOOL_FAILURE','TRANSIENT_GATEWAY','UNKNOWN','VALIDATION_FAILURE']
assert isinstance(d['changed'], list) and isinstance(d['verified'], list) and isinstance(d['unverified'], list)
assert isinstance(d['candidate_tasks'], list)
print('JSON_OK fields=%d candidate_tasks=%d' % (len(req), len(d['candidate_tasks'])))