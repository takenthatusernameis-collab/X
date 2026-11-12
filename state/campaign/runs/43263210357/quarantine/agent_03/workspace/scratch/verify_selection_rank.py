"""Read-only audit: reproduce the controller's qualitative task ranking.

Verifies that the campaign controller selected the top-ranked OPEN
LEARNING_PROCESS task and that the contract's compared_against list matches
the controller's own candidate ordering. Does not mutate any state.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import campaign_controller as c  # noqa: E402


def main() -> int:
    tasks = c.load_queue()
    # The queue has P-001 marked SELECTED for the in-flight agent; restore the
    # pre-selection view in memory only so the ordering is reproducible.
    for t in tasks:
        if t["status"] == "SELECTED":
            t["status"] = "OPEN"

    candidates = c.candidate_tasks_for_role(tasks, "LEARNING_PROCESS")
    order = [t["task_id"] for t in candidates]

    contract = json.loads((ROOT / "state" / "campaign" / "current_task.json").read_text())
    reported = contract["selection_basis"]["compared_against"]

    print("LEARNING_PROCESS candidate order:", order)
    print("controller top-5:", order[:5])
    print("contract compared_against:", reported)
    print("TOP5_MATCH:", order[:5] == reported)
    print("SELECTED_TASK:", contract["task_id"])
    print("SELECTED_IS_RANK1:", order[0] == contract["task_id"])
    for t in candidates:
        print(t["task_id"], c.task_rank(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())