#!/usr/bin/env python3
"""Run fresh Kilo sessions; only the enclosing workflow imposes an execution deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


MAX_TRANSIENT_GATEWAY_RETRIES = 3


def may_retry_transient_gateway(failure_class: str, retry_count: int) -> bool:
    """Allow a small recovery budget for transient gateway failures only."""
    return failure_class == "TRANSIENT_GATEWAY" and retry_count < MAX_TRANSIENT_GATEWAY_RETRIES


def sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_workspace(path: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    hashes: dict[str, str] = {}
    for rel in result.stdout.splitlines():
        if not rel:
            continue
        digest = sha(path / rel)
        if digest:
            hashes[rel] = digest
    return hashes


def classify_failure(output: str, return_code: int, liveness: str) -> str:
    lower = output.lower()
    transient_markers = (
        "invalid request error trace_id:",
        "upstream idle timeout exceeded",
        "idle timeout exceeded",
        "econnreset",
        "econnrefused",
        "etimedout",
        "eai_again",
        "enetunreach",
        "ehostunreach",
        "socket hang up",
        "fetch failed",
        "temporary failure in name resolution",
        "gateway timeout",
        "bad gateway",
        "service unavailable",
        "too many requests",
        "rate limit",
        "resource_exhausted",
        "temporarily unavailable",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    )
    if any(marker in lower for marker in transient_markers):
        return "TRANSIENT_GATEWAY"
    if liveness in {"STALLED", "TIMED_OUT"}:
        return "LIVENESS"
    if return_code != 0:
        if "permission" in lower or "not allowed" in lower:
            return "TOOL_FAILURE"
        if "traceback" in lower or "syntaxerror" in lower or "typeerror" in lower:
            return "RESEARCH_EXECUTION"
        return "UNKNOWN"
    return "NONE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--agent-number", required=True, type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    args = ap.parse_args()

    run_root = Path("state/campaign/runs") / str(args.run_id)
    agent_dir = run_root / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    progress = Path("state/worker_progress.md")
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    contract = json.loads(
        (agent_dir / f"agent_{args.agent_number:02d}_contract.json").read_text(encoding="utf-8")
    )

    before = snapshot_workspace(Path("."))
    (agent_dir / f"agent_{args.agent_number:02d}_before.json").write_text(
        json.dumps({"hashes": before, "captured_utc": utc_now()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    progress.write_text(
        "\n".join(
            [
                "# Worker Progress",
                "",
                f"activation_id: {args.run_id}",
                f"agent_number: {args.agent_number:02d}",
                f"task_id: {contract['task_id']}",
                f"phase: AGENT_{args.agent_number:02d}_START",
                "status: RUNNING",
                "last_verified_milestone: controller-started fresh Kilo session",
                "next_bounded_action: execute the declared focused task",
                "",
            ]
        ),
        encoding="utf-8",
    )

    log_path = agent_dir / f"agent_{args.agent_number:02d}_kilo.log"
    session_started_utc = utc_now()
    # No per-agent deadline or inactivity kill is applied here.
    # The enclosing GitHub Actions job/workflow timeout is the sole execution cutoff.
    kilo_bin = os.environ.get("KILO_BIN", "kilo")
    retry_count = 0
    retry_delay = 15
    attempt_number = 0
    return_code = 1
    liveness = "RUNNING"
    semantic_progress_seen = False
    failure_class = "UNKNOWN"
    output_chunks: list[str] = []

    while True:
        attempt_number += 1
        last_progress_mtime = progress.stat().st_mtime if progress.exists() else time.time()
        attempt_semantic_progress = False
        attempt_chunks: list[str] = []
        proc = subprocess.Popen(
            [kilo_bin, "run", "--model", args.model, "--auto", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        selector = selectors.DefaultSelector()
        if proc.stdout is not None:
            selector.register(proc.stdout, selectors.EVENT_READ)

        attempt_liveness = "RUNNING"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                if attempt_number == 1:
                    log.write(f"--- Fresh Kilo session attempt {attempt_number} ---\n")
                else:
                    log.write(
                        f"\n--- Fresh Kilo session retry attempt {attempt_number} "
                        f"(transient upstream failure; retry #{retry_count}) ---\n"
                    )
                log.flush()

                while True:
                    events = selector.select(timeout=1.0)
                    if events:
                        for key, _ in events:
                            line = key.fileobj.readline()
                            if line:
                                print(line, end="")
                                log.write(line)
                                log.flush()
                                attempt_chunks.append(line)
                                output_chunks.append(line)

                    current_progress_mtime = (
                        progress.stat().st_mtime if progress.exists() else last_progress_mtime
                    )
                    if current_progress_mtime > last_progress_mtime:
                        attempt_semantic_progress = True
                        last_progress_mtime = current_progress_mtime

                    if proc.poll() is not None:
                        break
        finally:
            selector.close()

        return_code = proc.returncode if proc.returncode is not None else 1
        attempt_liveness = (
            "COMPLETED_WITH_SEMANTIC_CHECKPOINTS"
            if attempt_semantic_progress
            else "COMPLETED_WITHOUT_SEMANTIC_CHECKPOINT"
        )
        liveness = attempt_liveness
        semantic_progress_seen = semantic_progress_seen or attempt_semantic_progress
        failure_class = classify_failure(
            "".join(attempt_chunks), return_code, attempt_liveness
        )

        if return_code == 0:
            break
        if not may_retry_transient_gateway(failure_class, retry_count):
            if failure_class == "TRANSIENT_GATEWAY":
                print(
                    "::error::Transient Kilo upstream/network failure persisted after "
                    f"{retry_count} retries; ending this agent session so the campaign "
                    "controller can proceed."
                )
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"Transient gateway retry budget exhausted after {retry_count} "
                        f"retries at attempt {attempt_number}; ending this agent session.\n"
                    )
            break

        retry_count += 1
        print(
            f"::warning::Transient Kilo upstream/network failure; launching a fresh "
            f"process after {retry_delay}s (retry #{retry_count}/{MAX_TRANSIENT_GATEWAY_RETRIES}; "
            "the enclosing workflow remains the only execution deadline)."
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"Transient upstream failure classified at attempt {attempt_number}; "
                f"retry #{retry_count}/{MAX_TRANSIENT_GATEWAY_RETRIES} in {retry_delay}s.\n"
            )
        time.sleep(retry_delay)
        retry_delay = min(300, retry_delay * 2)

    runtime_output = "".join(output_chunks)
    runtime = {
        "agent_number": args.agent_number,
        "campaign_slot": args.agent_number,
        "global_agent_number": contract["global_agent_number"],
        "run_id": str(args.run_id),
        "task_id": contract["task_id"],
        "fresh_session": True,
        "started_utc": session_started_utc,
        "finished_utc": utc_now(),
        "return_code": return_code,
        "liveness": liveness,
        "failure_class": failure_class,
        "log": str(log_path),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "worker_progress_sha256": sha(progress),
        "automatic_retry": retry_count > 0,
        "retry_count": retry_count,
        "max_transient_gateway_retries": MAX_TRANSIENT_GATEWAY_RETRIES,
        "retry_budget_exhausted": failure_class == "TRANSIENT_GATEWAY" and retry_count >= MAX_TRANSIENT_GATEWAY_RETRIES,
        "retry_policy": "recognized transient upstream/network errors only; at most 3 fresh-process retries (4 total attempts) with capped exponential backoff; no per-agent execution deadline; the enclosing workflow cutoff remains authoritative",
        "previous_session_context_reused": False,
    }
    (agent_dir / f"agent_{args.agent_number:02d}_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"AGENT_RUNTIME={json.dumps(runtime, sort_keys=True)}")

    return 0 if return_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
