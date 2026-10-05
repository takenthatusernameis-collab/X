import json
d=json.load(open('/home/runner/work/X/X/state/activation_status.json'))
required=['activation_id','run_attempt','sha','ref_name','repository','status','objective','phase','changed','verified','unverified','next','acceptance','research_conclusion']
missing=[k for k in required if k not in d]
print('missing keys:', missing)
print('status:', d['status'])
print('activation_id:', d['activation_id'])
print('phase:', d['phase'])
print('changed count:', len(d['changed']))
print('verified count:', len(d['verified']))
print('unverified count:', len(d['unverified']))
try:
    json.dumps(d, indent=2)
    print('json serializable: OK')
except Exception as e:
    print('json error:', e)
