#!/usr/bin/env python3
"""Scratch: read-only replica of validate_campaign_agent path checks for agent 01."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "44436578418"
AGENT = 1
AGENT_DIR = ROOT / "state" / "campaign" / "runs" / RUN_ID / "agents"

PROTECTED_PREFIXES = (".github/workflows/", ".github/scripts/", ".kilo/")
PROTECTED_FILES = {
    "AGENTS.md", "ENTERPRISE.md", "MANUAL_SETUP.md", "PERSISTENCE_POLICY.md",
    "tests/test_campaign_architecture.py", "tests/test_staged_secret_scan.py",
}


def run(cmd):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return p.stdout


def sha(path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


before = json.loads((AGENT_DIR / f"agent_{AGENT:02d}_before.json").read_text())
before_hashes = before.get("hashes", {})

out = run(["git", "diff", "--name-only"])
out_cached = run(["git", "diff", "--cached", "--name-only"])
out2 = run(["git", "ls-files", "--others", "--exclude-standard"])
paths = set(x for x in out.splitlines() if x) | set(x for x in out_cached.splitlines() if x) | set(x for x in out2.splitlines() if x)
after_paths = sorted(p for p in paths if "__pycache__/" not in p and not p.endswith(".pyc"))

touched = [p for p in after_paths if before_hashes.get(p) != sha(ROOT / p)]
protected = sorted(p for p in after_paths if p in PROTECTED_FILES or any(p.startswith(x) for x in PROTECTED_PREFIXES))

expected_record = f"state/campaign/runs/{RUN_ID}/agents/agent_{AGENT:02d}.json"
unexpected_campaign = [
    p for p in touched
    if p.startswith("state/campaign/")
    and p not in {expected_record}
    and not p.startswith(f"state/campaign/runs/{RUN_ID}/agents/agent_{AGENT:02d}_")
]

print("after_paths:", after_paths)
print("touched:", touched)
print("protected:", protected)
print("unexpected_campaign_paths:", unexpected_campaign)
print("regression_trigger_paths:", [p for p in touched if p.startswith(("research/", "tests/", "examples/"))])
ok = not protected and not unexpected_campaign
print("PATH_CHECK:", "OK" if ok else "WOULD_FAIL")