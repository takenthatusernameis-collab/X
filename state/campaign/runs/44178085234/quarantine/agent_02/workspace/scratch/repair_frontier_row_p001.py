#!/usr/bin/env python3
"""Inspect (and optionally repair) the malformed momentum row in the LEARNING_STATE frontier table.

Structure found:
  line 30: '| Momentum (long the previous 5-day return ...) |'      <- 1-cell truncated row
  line 31: '| Momentum: longer-horizon generalization ... |'        <- intact 4-cell row
  line 32: ' ADMITTED AS CANDIDATE POSITIVE EVIDENCE (...) | ... |' <- orphaned remaining cells of line 30
  line 33: ''                                                        <- blank line that splits the table
  line 34+: intact rows for cross-sectional momentum and 20-day breakout

Usage:
  python3 scratch/repair_frontier_row_p001.py inspect
  python3 scratch/repair_frontier_row_p001.py repair
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "state" / "LEARNING_STATE.md"


def cells(line: str) -> list[str]:
    return line.strip().strip("|").split("|")


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "inspect"
    lines = TARGET.read_text(encoding="utf-8").split("\n")
    idx = next(i for i, line in enumerate(lines) if line.startswith("| Momentum (long the previous 5-day return"))
    orphan_idx = idx + 2
    print(f"momentum_row_line={idx + 1} cells={len(cells(lines[idx]))}")
    print(f"orphan_line={orphan_idx + 1} cells={len(cells(lines[orphan_idx]))} starts={lines[orphan_idx][:40]!r}")
    print(f"blank_after_orphan={not lines[orphan_idx + 1].strip()}")
    print(f"line_after_blank={lines[orphan_idx + 2][:70]!r}")
    if mode != "repair":
        return 0
    if len(cells(lines[idx])) != 1:
        raise SystemExit("unexpected: momentum row already has multiple cells")
    if not lines[orphan_idx].startswith(" ADMITTED AS CANDIDATE POSITIVE EVIDENCE"):
        raise SystemExit("unexpected: orphan continuation not found")
    if lines[orphan_idx + 1].strip():
        raise SystemExit("unexpected: expected blank line after orphan")
    merged = lines[idx] + lines[orphan_idx]
    if len(cells(merged)) != 4:
        raise SystemExit(f"unexpected merged cell count {len(cells(merged))}")
    if not lines[orphan_idx + 2].startswith("| Cross-sectional momentum"):
        raise SystemExit("unexpected: expected cross-sectional momentum row after blank")
    new_lines = lines[: idx + 1] + [merged] + lines[idx + 2 : orphan_idx] + lines[orphan_idx + 2 :]
    TARGET.write_text("\n".join(new_lines), encoding="utf-8")
    print("repaired: merged orphan cells into the momentum row; removed splitting blank line")
    for i in range(idx, idx + 5):
        line = new_lines[i]
        print(f"  {i + 1}: cells={len(cells(line)) if line.startswith('|') else 'n/a'} :: {line[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))