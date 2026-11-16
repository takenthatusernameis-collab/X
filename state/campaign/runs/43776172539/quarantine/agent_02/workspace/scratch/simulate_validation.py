import hashlib
import json
import subprocess
import sys
from pathlib import Path

RUN_ID = "43776172539"
AGENT = 2
ROOT = Path(".").resolve()

PROTECTED_PREFIXES = (".github/workflows/", ".github/scripts/", ".kilo/")
PROTECTED_FILES = {
    "AGENTS.md",
    "ENTERPRISE.md",
    "MANUAL_SETUP.md",
    "PERSISTENCE_POLICY.md",
    "tests/test_campaign_architecture.py",
    "tests/test_staged_secret_scan.py",
}


def run(cmd):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, check=False)
    return p.returncode, p.stdout


def sha(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


before = json.loads(
    Path(f"state/campaign/runs/{RUN_ID}/agents/agent_{AGENT:02d}_before.json").read_text()
)
before_hashes = before.get("hashes", {})

_, out = run(["git", "diff", "--name-only"])
_, out_cached = run(["git", "diff", "--cached", "--name-only"])
_, out2 = run(["git", "ls-files", "--others", "--exclude-standard"])
paths = (
    set(x for x in out.splitlines() if x)
    | set(x for x in out_cached.splitlines() if x)
    | set(x for x in out2.splitlines() if x)
)
after_paths = sorted(
    p for p in paths if "__pycache__/" not in p and not p.endswith(".pyc")
)
touched = [p for p in after_paths if before_hashes.get(p) != sha(Path(p))]
protected = sorted(
    p for p in after_paths
    if p in PROTECTED_FILES or any(p.startswith(pre) for pre in PROTECTED_PREFIXES)
)
expected_record = f"state/campaign/runs/{RUN_ID}/agents/agent_{AGENT:02d}.json"
unexpected_campaign = [
    p for p in touched
    if p.startswith("state/campaign/")
    and p not in {expected_record}
    and not p.startswith(
        f"state/campaign/runs/{RUN_ID}/agents/agent_{AGENT:02d}_"
    )
]

print("after_paths:")
for p in after_paths:
    print("  ", p)
print("\ntouched:")
for p in touched:
    print("  ", p)
print("\nprotected:", protected)
print("unexpected_campaign_paths:", unexpected_campaign)
print("\nresearch/tests/examples touched (regression trigger):",
      [p for p in touched if p.startswith(("research/", "tests/", "examples/"))])
print("python touched (compile trigger):",
      [p for p in touched if p.endswith(".py") and Path(p).exists()])

failures = []
if protected:
    failures.append("protected_control_plane_modified")
if unexpected_campaign:
    failures.append("controller_state_modified")
print("\nsimulated structural failures:", failures)
sys.exit(1 if failures else 0)