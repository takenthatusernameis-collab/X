#!/usr/bin/env python3
"""Controller-owned activation receipt identity stamping."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Mapping


def stamp_receipt(
    path: Path,
    *,
    env: Mapping[str, str] | None = None,
    head_sha: str | None = None,
) -> dict:
    if env is None:
        env = os.environ
    run_id = env.get("GITHUB_RUN_ID")
    ref_name = env.get("GITHUB_REF_NAME")
    repository = env.get("GITHUB_REPOSITORY")
    if not run_id or not ref_name or not repository:
        raise SystemExit("missing required GitHub activation identity environment")

    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["activation_id"] = run_id
    receipt["run_attempt"] = env.get("GITHUB_RUN_ATTEMPT", "1")
    receipt["sha"] = head_sha or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    receipt["ref_name"] = ref_name
    receipt["repository"] = repository

    path.write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    path = Path("state/activation_status.json")
    receipt = stamp_receipt(path)
    print(
        "Controller-stamped activation identity: "
        f"id={receipt['activation_id']} "
        f"attempt={receipt['run_attempt']} "
        f"sha={receipt['sha']} "
        f"ref={receipt['ref_name']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
