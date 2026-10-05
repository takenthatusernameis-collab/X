import json
d=json.load(open('/home/runner/work/X/X/state/activation_status.json'))
notes=d['notes']
i=notes.find('defect was')
print(notes[i-200:i+700])
