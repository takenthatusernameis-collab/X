#!/usr/bin/env python3
"""Restore the longer-horizon momentum frontier row that a prior repair step dropped.

Source of truth: the committed baseline copy of state/LEARNING_STATE.md (HEAD), which
matches the controller-captured pre-activation hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "state" / "LEARNING_STATE.md"
BASELINE = Path("/tmp/kilo/orig_learning_state.md")
EXPECTED_BASELINE_SHA = "5b2d37b5c78e45eb58818ef7de2fd2db13ee1ee3da7b3c82dc8017cb8f37b04f"


def main() -> int:
    raw = BASELINE.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    print(f"baseline_sha={sha} expected={EXPECTED_BASELINE_SHA} match={sha == EXPECTED_BASELINE_SHA}")
    if sha != EXPECTED_BASELINE_SHA:
        raise SystemExit("baseline mismatch; refusing to restore")
    base_lines = raw.decode("utf-8").split("\n")
    row = next(
        (line for line in base_lines if line.startswith("| Momentum: longer-horizon generalization")),
        None,
    )
    if row is None:
        raise SystemExit("longer-horizon row not found in baseline")
    print(f"baseline row cells={len(row.strip().strip('|').split('|'))} chars={len(row)}")

    lines = TARGET.read_text(encoding="utf-8").split("\n")
    if any(line.startswith("| Momentum: longer-horizon generalization") for line in lines):
        raise SystemExit("row already present; nothing to do")
    idx = next(i for i, line in enumerate(lines) if line.startswith("| Momentum (long the previous 5-day return"))
    lines.insert(idx + 1, row)
    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"restored longer-horizon row after line {idx + 1}")
    for j in range(idx, idx + 5):
        line = lines[j]
        n = len(line.strip().strip("|").split("|")) if line.startswith("|") else "n/a"
        print(f"  {j + 1}: cells={n} :: {line[:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())