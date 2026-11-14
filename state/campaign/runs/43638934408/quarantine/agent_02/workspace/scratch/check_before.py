import hashlib
import json
import subprocess
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


before = json.loads(Path("state/campaign/runs/43638934408/agents/agent_02_before.json").read_text())["hashes"]
out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
out += subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True).stdout.split()
out += subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
changed = sorted(set(x for x in out if x and "__pycache__/" not in x and not x.endswith(".pyc")))
print("touched (hash differs from before snapshot):")
for p in changed:
    if not Path(p).exists():
        continue
    if before.get(p) != sha(Path(p)):
        print("  ", p, "(not in before)" if p not in before else "(hash changed)")