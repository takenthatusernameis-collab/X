#!/usr/bin/env python3
"""Deep per-step audit for the X research enterprise workflow.

The audit is deliberately non-blocking at the workflow level: callers use
continue-on-error so the audit can diagnose failures without hiding them.
It records a durable-for-the-job JSONL ledger, workspace integrity, workflow
shell syntax, credential-like changes, and research-program invariants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
RUNNER_TEMP = Path(os.environ.get("RUNNER_TEMP", "/tmp"))
LEDGER = RUNNER_TEMP / "x-workflow-deep-audit.jsonl"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
ACTIVE_STATUSES = {
    "ACTIVE", "ACTIVE_HIGH_INTENSITY", "ACTIVE_LOW_INTENSITY",
    "DEPRIORITIZED", "READY_FOR_REVIEW", "SUBSTANTIAL_EFFORT",
    "EXHAUSTED_FOR_NOW", "REOPENED", "VERIFIED", "NEGATED",
}
SECRET_FILE_RE = re.compile(
    r"(^|/)(\.env(?:\..*)?|id_rsa|id_dsa|id_ecdsa|id_ed25519|"
    r"credentials\.(?:json|ya?ml|toml)|secrets\.(?:json|ya?ml|toml))$"
    r"|\.(?:pem|key)$"
)
SECRET_TEXT_RE = re.compile(
    r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|"
    r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
    r"sk-[A-Za-z0-9_-]{20,}"
)


def sh(*args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:
        return 99, f"{type(exc).__name__}: {exc}"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def workflow_shell_audit() -> list[str]:
    issues: list[str] = []
    workflows = sorted((ROOT / ".github/workflows").glob("*.yml"))
    if not workflows:
        return ["No workflow YAML files found."]
    cache_dir = RUNNER_TEMP / "workflow-shell-audit"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for wf in workflows:
        raw = wf.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        cache = cache_dir / f"{wf.name}.{digest}.ok"
        if cache.exists():
            continue

        lines = raw.splitlines()
        i = 0
        block_no = 0
        while i < len(lines):
            line = lines[i]
            m = re.match(r"^(\s*)run:\s*\|\s*$", line)
            if not m:
                i += 1
                continue
            base_indent = len(m.group(1))
            i += 1
            block: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= base_indent:
                    break
                block.append(nxt)
                i += 1
            block_no += 1
            dedented = []
            for b in block:
                if not b.strip():
                    dedented.append("")
                else:
                    dedented.append(b[base_indent + 2:] if len(b) >= base_indent + 2 else "")
            script = "\n".join(dedented) + "\n"
            # GitHub expressions are interpolated before Bash sees the script.
            # Replace them with inert literals for syntax-only validation.
            script = re.sub(r"\$\{\{.*?\}\}", "0", script, flags=re.S)
            with tempfile.NamedTemporaryFile(
                "w", suffix=".sh", prefix="workflow-audit-", delete=False, encoding="utf-8"
            ) as handle:
                handle.write(script)
                temp_path = handle.name
            rc, out = sh("bash", "-n", temp_path)
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
            if rc != 0:
                issues.append(f"{wf.name}: run block {block_no} fails bash -n: {out[-800:]}")
        cache.write_text("ok\n", encoding="utf-8")
    return issues


def audit_program() -> list[str]:
    issues: list[str] = []
    path = ROOT / "state/research/PROGRAM.json"
    if not path.is_file():
        return issues
    try:
        program = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"PROGRAM.json unreadable: {type(exc).__name__}: {exc}"]

    surfaces = program.get("surfaces")
    families = program.get("evolution_families")
    if not isinstance(surfaces, list) or not isinstance(families, list):
        return ["PROGRAM.json missing list-valued surfaces/evolution_families"]

    family_ids = {f.get("family_id") for f in families if isinstance(f, dict)}
    family_by_surface: dict[str, list[dict[str, Any]]] = {}
    for f in families:
        if not isinstance(f, dict):
            issues.append("evolution_families contains non-object entry")
            continue
        sid = f.get("surface_id")
        family_by_surface.setdefault(sid, []).append(f)
        status = f.get("status")
        if status != "ARCHIVED" and sid not in {s.get("surface_id") for s in surfaces if isinstance(s, dict)}:
            issues.append(f"family {f.get('family_id')} points to unknown surface {sid}")
        for field in ("seed_requests", "best_candidates", "lineage"):
            if field in f and not isinstance(f[field], list):
                issues.append(f"family {f.get('family_id')} field {field} is not a list")

        for idx, seed in enumerate(f.get("seed_requests") or []):
            if not isinstance(seed, dict):
                issues.append(f"family {f.get('family_id')} seed {idx} is not an object")
                continue
            if str(seed.get("method", "GET")).upper() not in SAFE_METHODS:
                issues.append(f"family {f.get('family_id')} seed {idx} uses unsafe method")
            if not isinstance(seed.get("path"), str) or not seed["path"].startswith("/"):
                issues.append(f"family {f.get('family_id')} seed {idx} has invalid path")
            if not isinstance(seed.get("query", {}), dict) or not isinstance(seed.get("headers", {}), dict):
                issues.append(f"family {f.get('family_id')} seed {idx} has invalid query/headers")

        for idx, cand in enumerate(f.get("best_candidates") or []):
            if not isinstance(cand, dict):
                issues.append(f"family {f.get('family_id')} candidate {idx} is not an object")
                continue
            req = cand.get("request")
            if not isinstance(req, dict):
                issues.append(f"family {f.get('family_id')} candidate {idx} lacks executable request")
                continue
            if str(req.get("method", "GET")).upper() not in SAFE_METHODS:
                issues.append(f"family {f.get('family_id')} candidate {idx} uses unsafe method")
            if not isinstance(req.get("path"), str) or not req["path"].startswith("/"):
                issues.append(f"family {f.get('family_id')} candidate {idx} has invalid path")
            if not isinstance(req.get("query", {}), dict) or not isinstance(req.get("headers", {}), dict):
                issues.append(f"family {f.get('family_id')} candidate {idx} has invalid query/headers")

    for s in surfaces:
        if not isinstance(s, dict):
            issues.append("surfaces contains non-object entry")
            continue
        sid = s.get("surface_id")
        status = s.get("status", "CANDIDATE")
        non_archived = [f for f in family_by_surface.get(sid, []) if f.get("status") != "ARCHIVED"]
        if status in ACTIVE_STATUSES and not non_archived:
            issues.append(
                f"active surface {sid} has no durable non-archived family; "
                "keep it CANDIDATE until a family exists"
            )
        declared = s.get("evolutionary_families")
        if isinstance(declared, list):
            for family_id in declared:
                if family_id not in family_ids:
                    issues.append(f"surface {sid} declares unknown evolutionary family {family_id}")

    return issues


def audit_workspace() -> list[str]:
    issues: list[str] = []
    rc, status = sh("git", "status", "--short")
    if rc not in (0,):
        issues.append(f"git status failed: {status}")
    tracked = status.splitlines()
    changed_paths = []
    for line in tracked:
        if len(line) >= 3:
            changed_paths.append(line[3:])
    for path in changed_paths:
        if SECRET_FILE_RE.search(path):
            issues.append(f"credential-like path changed: {path}")

    rc, diff = sh("git", "diff", "--cached", "--binary")
    if rc == 0 and SECRET_TEXT_RE.search(diff):
        issues.append("staged diff contains credential-like secret material")
    return issues


def final_rollup() -> int:
    if not LEDGER.exists():
        print("No per-step audit ledger exists.")
        return 0
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    findings = sum(len(r.get("issues", [])) for r in rows)
    print(f"Deep audit rollup: {len(rows)} checkpoints, {findings} findings.")
    for row in rows:
        if row.get("issues"):
            print(f"[{row.get('step')}]")
            for issue in row["issues"]:
                print(f"  - {issue}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="")
    ap.add_argument("--profile", default="desktop")
    ap.add_argument("--prev-outcome", default=os.environ.get("AUDIT_PREV_OUTCOME", "unknown"))
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    if args.final:
        return final_rollup()

    issues = []
    if args.step:
        issues.extend(workflow_shell_audit())
        issues.extend(audit_program())
        issues.extend(audit_workspace())

    payload = {
        "timestamp_utc": now(),
        "step": args.step,
        "profile": args.profile,
        "prev_outcome": args.prev_outcome,
        "head": sh("git", "rev-parse", "HEAD")[1],
        "status": sh("git", "status", "--short")[1],
        "issues": issues,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

    print(f"DEEP_AUDIT step={args.step!r} prev_outcome={args.prev_outcome!r} findings={len(issues)}")
    for issue in issues:
        print(f"::warning::DEEP_AUDIT {issue}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"\n**Deep audit — {args.step}**: {len(issues)} finding(s).\n")
            for issue in issues[:20]:
                handle.write(f"- {issue}\n")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
