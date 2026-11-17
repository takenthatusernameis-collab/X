#!/usr/bin/env python3
"""Verify the frontier-table repair preserved all durable content.

Checks:
  1. baseline and current files have the same non-empty content lines except the
     merged momentum row (i.e. every baseline content line still present, and the
     only removed line is the 1-cell truncated momentum row);
  2. the frontier table is one contiguous block with 4 cells per row;
  3. every frontier cell (hypothesis) present in the baseline is still present.
"""
from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "state" / "LEARNING_STATE.md"
BASELINE = Path("/tmp/kilo/orig_learning_state.md")
BASELINE_REF = "HEAD:state/LEARNING_STATE.md"


def load_baseline(force_git: bool = False) -> str:
    """Prefer the captured pre-activation copy; fall back to the committed baseline.

    Both must be the same pre-activation revision: the file was clean at HEAD before
    this activation's edit.
    """
    if BASELINE.exists() and not force_git:
        return BASELINE.read_text(encoding="utf-8")
    proc = subprocess.run(
        ["git", "show", BASELINE_REF],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"baseline unavailable: {proc.stderr.strip()}")
    print(f"baseline_source=git show {BASELINE_REF}")
    return proc.stdout


def cells(line: str) -> list[str]:
    return line.strip().strip("|").split("|")


def main() -> int:
    base = load_baseline(force_git="--from-git" in sys.argv).split("\n")
    cur = TARGET.read_text(encoding="utf-8").split("\n")
    print(f"baseline_lines={len(base)} current_lines={len(cur)}")

    removed, added = [], []
    for line in difflib.unified_diff(base, cur, lineterm="", n=0):
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        (removed if line.startswith("-") else added).append(line[1:])
    print(f"diff_removed_lines={len(removed)} diff_added_lines={len(added)}")
    for line in removed:
        print(f"  REMOVED: cells={len(cells(line))} :: {line[:100]!r}")
    for line in added:
        print(f"  ADDED:   cells={len(cells(line))} :: {line[:100]!r}")

    stale = [line for line in removed if len(cells(line)) == 1 and line.startswith("| Momentum (long the previous 5-day")]
    merged = [line for line in added if line.startswith("| Momentum (long the previous 5-day")]
    print(f"stale_single_cell_rows_removed={len(stale)} merged_rows_added={len(merged)}")
    other_removed = [
        line
        for line in removed
        if line not in stale and line.strip() and not any(line.strip() in merged_line for merged_line in merged)
    ]
    other_added = [line for line in added if line not in merged and line.strip()]
    print(f"orphan_lines_absorbed_into_merged_row={len([x for x in removed if x.strip() and x not in stale and x not in other_removed])}")
    print(f"other_content_removed={len(other_removed)} other_content_added={len(other_added)}")
    for line in other_removed + other_added:
        print(f"  UNEXPECTED: {line[:120]!r}")

    header = next(i for i, line in enumerate(cur) if line.startswith("| Area / hypothesis |"))
    rows = 0
    malformed = []
    i = header
    while i < len(cur) and cur[i].startswith("|"):
        rows += 1
        if len(cells(cur[i])) != 4:
            malformed.append((i + 1, len(cells(cur[i]))))
        i += 1
    print(f"frontier_table: header_line={header + 1} rows={rows} malformed={malformed}")
    print(f"line_after_table={cur[i + 1][:60]!r} blank={not cur[i].strip() if i < len(cur) else 'EOF'}")

    base_cells = {cells(line)[0] for line in base if line.startswith("|")}
    cur_cells = {cells(line)[0] for line in cur if line.startswith("|")}
    lost = sorted(base_cells - cur_cells)
    print(f"baseline_table_cells_missing_now={lost}")
    return 0 if not malformed and not lost and not other_removed else 1


if __name__ == "__main__":
    raise SystemExit(main())