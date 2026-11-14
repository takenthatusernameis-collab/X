#!/usr/bin/env python3
"""One-shot markdown repair: drop the stray blank line that separates the new
learning-history row from the preceding table row in state/LEARNING_STATE.md.

The row was appended via a heading anchor, which left one empty line between the
last table row and the new row and therefore broke table continuity. This script
removes exactly that one blank line and nothing else; it is idempotent.
"""

from __future__ import annotations

from pathlib import Path

PATH = Path("state/LEARNING_STATE.md")
ROW_PREFIX = "| 43548892244 agent 01 (2026-11-14) |"


def main() -> int:
    lines = PATH.read_text(encoding="utf-8").split("\n")
    out: list[str] = []
    removed = 0
    for idx, line in enumerate(lines):
        if (
            line.startswith(ROW_PREFIX)
            and out
            and out[-1] == ""
            and len(out) >= 2
            and out[-2].startswith("| 37414038601 (2026-10-06) |")
        ):
            out.pop()
            removed += 1
        out.append(line)
    if removed:
        PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"removed_blank_lines={removed} total_lines={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())