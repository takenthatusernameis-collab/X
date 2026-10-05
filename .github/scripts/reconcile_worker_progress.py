#!/usr/bin/env python3
"""Reconcile the worker liveness receipt with the authoritative activation receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TERMINAL = {"COMPLETE", "PARTIAL", "FAILED", "BLOCKED", "QUARANTINED", "NO_SUBSTANTIVE_ACTION"}


def reconcile(activation_path: Path, progress_path: Path, run_id: str) -> bool:
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    activation_id = str(activation.get("activation_id", ""))
    status = activation.get("status", "")
    if activation_id != str(run_id):
        raise SystemExit("activation receipt identity does not match current run")
    if status not in TERMINAL:
        return False

    existing = progress_path.read_text(encoding="utf-8") if progress_path.exists() else ""
    current_id = ""
    if existing:
        for line in existing.splitlines():
            if line.startswith("activation_id:"):
                current_id = line.split(":", 1)[1].strip()
                break
    if current_id not in {"", activation_id}:
        raise SystemExit("worker progress belongs to a different activation")

    objective = activation.get("objective", "").strip()
    verified = activation.get("verified", [])
    next_action = activation.get("next", "").strip()
    milestone = verified[-1] if verified else "authoritative activation receipt reached terminal state"

    progress_path.write_text(
        "\n".join(
            [
                "# Worker Progress",
                "",
                f"activation_id: {activation_id}",
                "phase: VERIFY",
                f"status: {status}",
                "",
                "## objective",
                objective,
                "",
                "## last verified milestone",
                milestone,
                "",
                "## next bounded action",
                next_action,
                "",
                "## reconciliation",
                "Controller reconciliation synchronized worker_progress.md with the authoritative activation receipt.",
                "",
                "This file is a liveness contract. Update it only after a real research-state transition or verified milestone.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return True


def self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        activation = root / "activation.json"
        progress = root / "worker_progress.md"
        activation.write_text(
            json.dumps(
                {
                    "activation_id": "123",
                    "status": "COMPLETE",
                    "objective": "test objective",
                    "verified": ["test milestone"],
                    "next": "test next",
                }
            ),
            encoding="utf-8",
        )
        progress.write_text(
            "activation_id: 123\nphase: SMOKE / DEEP\nstatus: IN_PROGRESS\n",
            encoding="utf-8",
        )
        assert reconcile(activation, progress, "123") is True
        text = progress.read_text(encoding="utf-8")
        assert "status: COMPLETE" in text
        assert "test milestone" in text

    print("worker progress reconciliation self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--activation", default="state/activation_status.json")
    parser.add_argument("--progress", default="state/worker_progress.md")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.run_id:
        parser.error("--run-id or GITHUB_RUN_ID is required")
    reconcile(Path(args.activation), Path(args.progress), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
