#!/usr/bin/env python3
"""Read-only equivalent of the controller's post-session agent gate.

It imports the controller's own validation functions instead of running
`validate_campaign_agent.py` as a subprocess, because that script's failure branch
quarantines the workspace and restores tracked files. Nothing is written except
this script's own stdout.

Checks:
  1. record schema/semantic validation (controller.validate_record);
  2. protected-path check recomputed from the agent's before-hashes;
  3. `next` is exactly one bounded action;
  4. markdown hygiene of the edited tracked files (no trailing whitespace, no tabs).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "43548892244"
AGENT = 1

spec = importlib.util.spec_from_file_location(
    "validate_campaign_agent", ROOT / ".github" / "scripts" / "validate_campaign_agent.py"
)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

PROTECTED_PREFIXES = (".github/workflows/", ".github/scripts/", ".kilo/")
PROTECTED_FILES = {
    "AGENTS.md",
    "ENTERPRISE.md",
    "MANUAL_SETUP.md",
    "PERSISTENCE_POLICY.md",
    "tests/test_campaign_architecture.py",
    "tests/test_staged_secret_scan.py",
}


def sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    agent_dir = ROOT / "state" / "campaign" / "runs" / RUN_ID / "agents"
    contract = json.loads((agent_dir / f"agent_{AGENT:02d}_contract.json").read_text(encoding="utf-8"))
    record = json.loads((agent_dir / f"agent_{AGENT:02d}.json").read_text(encoding="utf-8"))
    before = json.loads((agent_dir / f"agent_{AGENT:02d}_before.json").read_text(encoding="utf-8"))
    before_hashes = before.get("hashes", {})

    print("[1] controller record validation")
    failures = validator.validate_record(record, AGENT, contract)
    print(f"  failures={failures}")

    print("\n[2] protected-path check (recomputed from before-hashes)")
    changed = []
    for rel, old in before_hashes.items():
        new = sha(ROOT / rel)
        if new != old:
            changed.append(rel)
    protected = sorted(
        p for p in changed if p in PROTECTED_FILES or any(p.startswith(x) for x in PROTECTED_PREFIXES)
    )
    print(f"  changed_tracked_paths={changed}")
    print(f"  protected_touched={protected}")

    print("\n[3] `next` is one bounded action")
    next_lines = [x for x in record["next"].splitlines() if x.strip()]
    print(f"  non_empty_lines={len(next_lines)} ok={len(next_lines) == 1}")

    print("\n[4] markdown hygiene of edited tracked files")
    for rel in ("state/LEARNING_STATE.md",):
        text = (ROOT / rel).read_text(encoding="utf-8")
        trailing = [i + 1 for i, line in enumerate(text.split("\n")) if line != line.rstrip()]
        tabs = [i + 1 for i, line in enumerate(text.split("\n")) if "\t" in line]
        print(f"  {rel}: trailing_whitespace_lines={trailing} tab_lines={tabs}")

    ok = not failures and not protected and len(next_lines) == 1
    print(f"\nRESULT={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())