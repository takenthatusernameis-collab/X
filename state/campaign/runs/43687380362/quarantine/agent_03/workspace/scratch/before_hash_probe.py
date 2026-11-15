import hashlib
import json
from pathlib import Path

before = json.load(open("state/campaign/runs/43687380362/agents/agent_02_before.json"))
hashes = before["hashes"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


changed = []
missing = []
for rel, old in hashes.items():
    p = Path(rel)
    if not p.exists():
        missing.append(rel)
        continue
    if sha(p) != old:
        changed.append(rel)
print("changed since before.json:", changed)
print("missing since before.json:", missing)