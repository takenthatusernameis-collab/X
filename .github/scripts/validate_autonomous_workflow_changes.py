#!/usr/bin/env python3
"""Validate agent-authored GitHub workflow changes against the immutable scheduler boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SUPERVISOR = ".github/workflows/kilo-wakeup.yml"
FORBIDDEN_TRIGGERS = ("schedule:", "workflow_run:", "repository_dispatch:", "workflow_call:")
FORBIDDEN_PERMISSIONS = ("contents: write", "actions: write", "id-token: write")
FORBIDDEN_COMMANDS = ("gh workflow run", "repository_dispatch", "workflow_dispatch", "/dispatches", "secrets: inherit")
FORBIDDEN_RUNNERS = ("runs-on: self-hosted", "runs-on: [self-hosted")

def staged_paths() -> list[str]:
    p = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return [x for x in p.stdout.decode("utf-8", "replace").split("\x00") if x]

def staged_text(path: str) -> str:
    p = subprocess.run(
        ["git", "show", f":{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return p.stdout.decode("utf-8", "replace")

def main() -> int:
    failures: list[str] = []
    for path in staged_paths():
        if not path.startswith(".github/workflows/"):
            continue
        if path == SUPERVISOR:
            failures.append("supervisor_workflow_modified")
            continue
        text = staged_text(path).lower()
        for marker in FORBIDDEN_TRIGGERS:
            if marker in text:
                failures.append(f"{path}:forbidden_trigger:{marker}")
        for marker in FORBIDDEN_PERMISSIONS:
            if marker in text:
                failures.append(f"{path}:forbidden_permission:{marker}")
        for marker in FORBIDDEN_COMMANDS:
            if marker in text:
                failures.append(f"{path}:forbidden_recursive_dispatch:{marker}")
        for marker in FORBIDDEN_RUNNERS:
            if marker in text:
                failures.append(f"{path}:forbidden_runner:{marker}")
    if failures:
        print("\n".join(failures))
        return 1
    print("AUTONOMOUS_WORKFLOW_CHANGES_VALID=1")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
