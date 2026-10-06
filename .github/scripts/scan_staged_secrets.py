#!/usr/bin/env python3
"""Scan staged file contents for credential-like material without trusting filenames."""

from __future__ import annotations

import re
import subprocess
from typing import Iterable


PATTERNS = (
    ("private_key", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    ("github_classic_token", re.compile(r"ghp_[A-Za-z0-9]{30,}")),
    ("github_fine_grained_token", re.compile(r"github_pat_[A-Za-z0-9_]{40,}")),
    # Kilo documents Gateway API keys as JWTs. Match them only when they
    # appear in an authentication context instead of guessing a "kilo_" prefix.
    (
        "kilo_jwt_key",
        re.compile(
            r"(?i)(?:KILO_API_KEY\s*[:=]\s*|Authorization\s*:\s*Bearer\s+)"
            r"(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"
        ),
    ),
    ("openai_style_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
)


def staged_paths() -> list[str]:
    raw = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
    )
    return [p.decode("utf-8") for p in raw.split(b"\0") if p]


def staged_text(path: str) -> str:
    raw = subprocess.check_output(["git", "cat-file", "blob", f":{path}"])
    return raw.decode("utf-8", errors="ignore")


def scan_text(text: str) -> list[str]:
    return [name for name, pattern in PATTERNS if pattern.search(text)]


def scan_staged(paths: Iterable[str] | None = None) -> list[tuple[str, list[str]]]:
    findings: list[tuple[str, list[str]]] = []
    for path in paths if paths is not None else staged_paths():
        matches = scan_text(staged_text(path))
        if matches:
            findings.append((path, matches))
    return findings


def main() -> int:
    findings = scan_staged()
    if findings:
        for path, matches in findings:
            print(f"credential-like content detected in staged file: {path} [{', '.join(matches)}]")
        return 1
    print("Staged-content credential scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
