import hashlib
import json
from pathlib import Path

before = json.loads(Path("state/campaign/runs/43776172539/agents/agent_02_before.json").read_text())
hashes = before.get("hashes", {})
print("before snapshot keys:", sorted(hashes.keys()))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for rel in sorted(hashes):
    p = Path(rel)
    if not p.exists():
        print(f"  MISSING {rel}")
        continue
    cur = sha(p)
    print(f"  {'CHANGED' if cur != hashes[rel] else 'same   '} {rel}")

print("\ncontroller campaign paths present:")
for rel in [
    "state/campaign/task_queue.json",
    "state/campaign/global_agent_counter.json",
    "state/campaign/current_task.json",
]:
    p = Path(rel)
    print(f"  {rel}: exists={p.exists()} in_before={rel in hashes}")