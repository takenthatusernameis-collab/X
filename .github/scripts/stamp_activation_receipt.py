#!/usr/bin/env python3
"""Controller-owned activation receipt identity stamping and validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Mapping


REQUIRED_FIELDS = {
    "activation_id",
    "status",
    "objective",
    "phase",
    "changed",
    "verified",
    "unverified",
    "next",
}
ALLOWED_STATUSES = {
    "COMPLETE",
    "PARTIAL",
    "FAILED",
    "BLOCKED",
    "QUARANTINED",
    "NO_SUBSTANTIVE_ACTION",
}
LIST_FIELDS = ("changed", "verified", "unverified")
STRING_FIELDS = ("objective", "phase", "next")


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


def validate_receipt(
    receipt: Mapping[str, object],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """Independently validate a controller-stamped activation receipt.

    Uses statement-form raises. Ternary `raise X if cond else None` is invalid
    because Python parses it as `raise (X if cond else None)` and a valid
    receipt then raises None.
    """
    if env is None:
        env = os.environ
    missing = REQUIRED_FIELDS - set(receipt)
    if missing:
        raise SystemExit(f"missing receipt fields: {sorted(missing)}")
    if receipt["status"] not in ALLOWED_STATUSES:
        raise SystemExit("invalid activation status")
    if str(receipt["activation_id"]) != str(env.get("GITHUB_RUN_ID")):
        raise SystemExit("receipt activation_id does not match current run")
    if not all(isinstance(receipt[k], list) for k in LIST_FIELDS):
        raise SystemExit("receipt list field has invalid type")
    if not all(isinstance(receipt[k], str) and receipt[k].strip() for k in STRING_FIELDS):
        raise SystemExit("receipt objective/phase/next must be non-empty strings")


def validate_receipt_path(
    path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, env=env)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing receipt without restamping identity.",
    )
    args = parser.parse_args(argv)
    path = Path("state/activation_status.json")
    if args.validate_only:
        validate_receipt_path(path)
        print("Activation receipt schema and run identity independently validated.")
        return 0

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
