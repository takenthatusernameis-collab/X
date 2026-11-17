#!/usr/bin/env python3
"""Read-only audit for activation P-001: selector reproducibility + frontier record structure.

Does not modify repository state. Prints findings only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import campaign_controller as cc  # noqa: E402

RUN = ROOT / "state" / "campaign" / "runs" / "44178085234"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)


def main() -> int:
    initial = json.loads((RUN / "task_queue_initial.json").read_text(encoding="utf-8"))["tasks"]
    current = json.loads((ROOT / "state" / "campaign" / "task_queue.json").read_text(encoding="utf-8"))["tasks"]
    contract = json.loads((RUN / "agents" / "agent_01_contract.json").read_text(encoding="utf-8"))

    section("1. Selector reproducibility (pre-selection snapshot)")
    candidates = cc.candidate_tasks_for_role(initial, "LEARNING_PROCESS")
    print(f"  ranked OPEN LEARNING_PROCESS tasks at selection time:")
    for idx, task in enumerate(candidates):
        print(f"    {idx}: {task['task_id']} rank={cc.task_rank(task)}")
    top = candidates[0]
    print(f"  recomputed_top={top['task_id']} contract_task={contract['task_id']} MATCH={top['task_id'] == contract['task_id']}")
    print(f"  compared_against_recomputed={[t['task_id'] for t in candidates[:5]]}")
    print(f"  compared_against_recorded={contract['selection_basis']['compared_against']}")
    print(f"  MATCH={[t['task_id'] for t in candidates[:5]] == contract['selection_basis']['compared_against']}")
    priority = [
        top["expected_information_gain"],
        top["downstream_leverage"],
        top["falsification_power"],
        top["verifiability"],
        top["effort"],
        top["complexity"],
        top["execution_risk"],
    ]
    print(f"  priority_recomputed={priority}")
    print(f"  priority_recorded={contract['selection_basis']['qualitative_priority']}")
    print(f"  MATCH={priority == contract['selection_basis']['qualitative_priority']}")
    print(f"  selected_task_status_in_initial_queue={[t['status'] for t in initial if t['task_id'] == top['task_id']]}")
    print(f"  selected_task_status_in_current_queue={[t['status'] for t in current if t['task_id'] == top['task_id']]}")
    print(f"  contract_matches_current_task_file={contract == json.loads((ROOT / 'state' / 'campaign' / 'current_task.json').read_text(encoding='utf-8'))}")

    section("2. Frontier table structure in state/LEARNING_STATE.md")
    text = (ROOT / "state" / "LEARNING_STATE.md").read_text(encoding="utf-8")
    lines = text.split("\n")
    header = next(i for i, line in enumerate(lines) if line.startswith("| Area / hypothesis |"))
    print(f"  header_line={header + 1}")
    malformed = []
    row_lines = []
    for idx in range(header, len(lines)):
        line = lines[idx]
        if not line.startswith("|"):
            break
        row_lines.append(idx)
        cells = line.strip().strip("|").split("|")
        if len(cells) != 4:
            malformed.append((idx + 1, len(cells), line.strip()[:70]))
    print(f"  contiguous_table_lines={len(row_lines)} malformed_rows={len(malformed)}")
    for lineno, count, preview in malformed:
        print(f"    line {lineno}: {count} cells :: {preview}")
    print("  lines after first contiguous block:")
    for idx in range(row_lines[-1] + 1, row_lines[-1] + 6):
        print(f"    {idx + 1}: {lines[idx][:100]!r}")

    section("3. Frontier rows that mention momentum")
    for idx in range(header, len(lines)):
        if "momentum" in lines[idx].lower() and lines[idx].startswith("|"):
            cells = lines[idx].strip().strip("|").split("|")
            print(f"  line {idx + 1}: cells={len(cells)} status={cells[1][:60] if len(cells) > 1 else 'MISSING'}")

    section("4. Activation log coverage and topic mix (dated logs only)")
    for log in sorted((ROOT / "logs").glob("ACTIVATION-*.md")):
        body = log.read_text(encoding="utf-8")
        blocks = re.split(r"^##+ ", body, flags=re.M)[1:]
        print(f"  {log.name}: blocks={len(blocks)}")
        for block in blocks:
            title = block.split("\n", 1)[0][:60]
            low = block.lower()
            tags = []
            if "ma_crossover" in low or "ma(20/60)" in low or "ma crossover" in low:
                tags.append("MA")
            if "momentum" in low:
                tags.append("MOM")
            if "breakout" in low:
                tags.append("BREAKOUT")
            if "mean_reversion" in low or "mean-reversion" in low or "reversal" in low:
                tags.append("REV")
            if "cross-sectional" in low or "relative_strength" in low:
                tags.append("CSRS")
            if "volatility" in low and "targeting" in low:
                tags.append("VOL")
            print(f"      - {title.strip()} :: {','.join(tags) if tags else 'OTHER'}")

    section("5. Frontier-first selection markers in durable records")
    for name in ("logs/ACTIVATION-2026-10-05.md", "logs/ACTIVATION-2026-10-06.md", "state/STATE.md", "state/LEARNING_STATE.md"):
        body = (ROOT / name).read_text(encoding="utf-8")
        print(f"  {name}: frontier-first mentions={len(re.findall(r'frontier-first', body))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())