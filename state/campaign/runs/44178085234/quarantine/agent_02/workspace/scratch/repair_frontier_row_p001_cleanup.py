#!/usr/bin/env python3
"""Remove the stale 1-cell duplicate left by the previous repair step.

Precondition: two consecutive lines share the same momentum cell text; the first has
1 cell (stale), the second has 4 cells (merged). Deletes the stale line only.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "state" / "LEARNING_STATE.md"


def cells(line: str) -> list[str]:
    return line.strip().strip("|").split("|")


def main() -> int:
    lines = TARGET.read_text(encoding="utf-8").split("\n")
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if not a.startswith("| Momentum (long the previous 5-day return"):
            continue
        if len(cells(a)) != 1 or len(cells(b)) != 4:
            raise SystemExit(f"unexpected structure at lines {i + 1}/{i + 2}")
        if cells(a)[0] != cells(b)[0]:
            raise SystemExit("unexpected: duplicate rows differ in first cell")
        del lines[i]
        TARGET.write_text("\n".join(lines), encoding="utf-8")
        print(f"removed stale 1-cell duplicate at line {i + 1}")
        for j in range(max(i - 2, 0), i + 4):
            line = lines[j]
            print(f"  {j + 1}: cells={len(cells(line)) if line.startswith('|') else 'n/a'} :: {line[:80]!r}")
        return 0
    raise SystemExit("stale duplicate not found; nothing to do")


if __name__ == "__main__":
    raise SystemExit(main())