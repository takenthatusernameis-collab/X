#!/usr/bin/env python3
"""Controller-side validation of one fresh Kilo campaign agent session."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROTECTED_PREFIXES = (".github/workflows/", ".kilo/")
PROTECTED_FILES = {
    "AGENTS.md",
    "ENTERPRISE.md",
    "MANUAL_SETUP.md",
    "PERSISTENCE_POLICY.md",
}


def run(cmd: list[str], *, timeout: int | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return 124, out + "\nTIMEOUT"
    return p.returncode, p.stdout


def sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_changed() -> list[str]:
    rc, out = run(["git", "diff", "--name-only"])
    if rc != 0:
        raise SystemExit(out)
    rc2, out2 = run(["git", "ls-files", "--others", "--exclude-standard"])
    if rc2 != 0:
        raise SystemExit(out2)
    return sorted(set([x for x in out.splitlines() if x] + [x for x in out2.splitlines() if x]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--exit-code", required=True, type=int)
    ap.add_argument("--before")
    ap.add_argument("--before-worker-progress-sha", default="")
    args = ap.parse_args()

    before = json.loads(Path(args.before).read_text(encoding="utf-8")) if args.before else {}
    before_hashes = before.get("hashes", {})
    after_paths = current_changed()

    touched: list[str] = []
    for path in after_paths:
        before_hash = before_hashes.get(path)
        after_hash = sha(Path(path))
        if before_hash != after_hash:
            touched.append(path)

    protected = sorted(
        p for p in after_paths
        if p in PROTECTED_FILES or any(p.startswith(prefix) for prefix in PROTECTED_PREFIXES)
    )

    failures: list[str] = []
    warnings: list[str] = []

    rc, out = run(["git", "diff", "--check"])
    if rc != 0:
        failures.append("git_diff_check_failed")
        warnings.append(out[-4000:])

    python_paths = [p for p in touched if p.endswith(".py") and Path(p).exists()]
    if python_paths:
        rc, out = run([sys.executable, "-m", "py_compile", *python_paths], timeout=120)
        if rc != 0:
            failures.append("python_compile_failed")
            warnings.append(out[-4000:])

    research_python_touched = any(
        p.startswith("research/") or p.startswith("tests/") or p.startswith("examples/")
        for p in touched
    )
    tests_run = False
    if research_python_touched:
        tests_run = True
        rc, out = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], timeout=120)
        if rc != 0:
            failures.append("regression_suite_failed")
            warnings.append(out[-6000:])

    receipt_checked = False
    receipt_path = Path("state/activation_status.json")
    if "state/activation_status.json" in touched and receipt_path.exists():
        receipt_checked = True
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            required = {"activation_id", "status", "objective", "phase", "changed", "verified", "unverified", "next"}
            if not required.issubset(receipt):
                failures.append("activation_receipt_schema_invalid")
        except Exception as exc:
            failures.append(f"activation_receipt_unreadable:{type(exc).__name__}")

    if protected:
        failures.append("protected_control_plane_modified:" + ",".join(protected))

    if args.exit_code != 0:
        warnings.append(f"worker_exit_code={args.exit_code}")

    checkpoint_path = Path("state/worker_progress.md")
    checkpoint_present = checkpoint_path.exists() and checkpoint_path.stat().st_size > 0
    before_checkpoint = args.before_worker_progress_sha or before.get("worker_progress_sha256")
    after_checkpoint = sha(checkpoint_path)
    checkpoint_changed = before_checkpoint != after_checkpoint

    if not checkpoint_present:
        failures.append("worker_progress_missing")
    elif not checkpoint_changed:
        warnings.append("no_post_start_worker_progress_transition_observed")

    if failures:
        status = "FAILED_VALIDATION"
        rc = 1
    elif args.exit_code == 0:
        status = "VERIFIED_SESSION"
        rc = 0
    else:
        status = "WORKER_FAILED_BUT_WORKTREE_VALID"
        rc = 1

    report = {
        "agent_id": args.agent_id,
        "worker_exit_code": args.exit_code,
        "status": status,
        "touched_paths": touched,
        "protected_paths": protected,
        "checkpoint_changed": checkpoint_changed,
        "python_paths_checked": python_paths,
        "regression_suite_run": tests_run,
        "activation_receipt_checked": receipt_checked,
        "warnings": warnings,
        "failures": failures,
    }

    output = Path(f"state/campaign/agent_{args.agent_id}_validation.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
