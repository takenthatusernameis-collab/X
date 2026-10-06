import json
from pathlib import Path
m = json.load(open('research/data/manifest.json'))
print(json.dumps(m['entries'][0], indent=1))
