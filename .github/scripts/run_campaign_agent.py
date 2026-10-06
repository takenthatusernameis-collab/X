#!/usr/bin/env python3
"""Run one genuinely fresh Kilo campaign session with bounded liveness and no retry."""

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
    if "invalid request error trace_id:" in lower:
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


def terminate_process(proc: subprocess.Popen[str]) -> int:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    return proc.returncode if proc.returncode is not None else 124


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--agent-number", required=True, type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--timeout-seconds", type=int, default=1760)
    ap.add_argument("--stall-seconds", type=int, default=900)
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
    start_mono = time.monotonic()
    last_progress_mtime = progress.stat().st_mtime
    last_signal_time = start_mono
    semantic_progress_seen = False
    liveness = "RUNNING"
    output_chunks: list[str] = []

    kilo_bin = os.environ.get("KILO_BIN", "kilo")
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

    try:
        with log_path.open("w", encoding="utf-8") as log:
            while True:
                now = time.monotonic()
                if now - start_mono >= args.timeout_seconds:
                    liveness = "TIMED_OUT"
                    terminate_process(proc)
                    break

                events = selector.select(timeout=1.0)
                if events:
                    for key, _ in events:
                        line = key.fileobj.readline()
                        if line:
                            print(line, end="")
                            log.write(line)
                            log.flush()
                            output_chunks.append(line)
                            last_signal_time = time.monotonic()

                current_progress_mtime = progress.stat().st_mtime if progress.exists() else last_progress_mtime
                if current_progress_mtime > last_progress_mtime:
                    semantic_progress_seen = True
                    last_progress_mtime = current_progress_mtime
                    last_signal_time = time.monotonic()

                if proc.poll() is not None:
                    break

                if (
                    time.monotonic() - start_mono >= args.stall_seconds
                    and time.monotonic() - last_signal_time >= args.stall_seconds
                ):
                    liveness = "STALLED"
                    terminate_process(proc)
                    break
    finally:
        selector.close()

    return_code = proc.returncode if proc.returncode is not None else 124
    if liveness == "RUNNING":
        liveness = (
            "COMPLETED_WITH_SEMANTIC_CHECKPOINTS"
            if semantic_progress_seen
            else "COMPLETED_WITHOUT_SEMANTIC_CHECKPOINT"
        )

    output = "".join(output_chunks)
    failure_class = classify_failure(output, return_code, liveness)
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
        "automatic_retry": False,
        "previous_session_context_reused": False,
    }
    (agent_dir / f"agent_{args.agent_number:02d}_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"AGENT_RUNTIME={json.dumps(runtime, sort_keys=True)}")

    return 0 if return_code == 0 and liveness not in {"STALLED", "TIMED_OUT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
