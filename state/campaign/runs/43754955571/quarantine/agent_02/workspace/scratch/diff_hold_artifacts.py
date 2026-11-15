"""Diff the artifact produced before and after the hold_profiles repair.

Confirms that the repair changed only the hold_profiles field and left every
computed number, verdict, and classification byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

A = Path("state/campaign/runs/43754955571/agent_02_artifact_before_holdprofiles_fix.json")
B = Path("state/check_artifacts/momentum_hold_period_sensitivity_results.json")


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, "{}.{}".format(prefix, k)))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True)
    else:
        out[prefix] = obj
    return out


def main() -> int:
    a = flatten(json.loads(A.read_text()))
    b = flatten(json.loads(B.read_text()))
    keys = sorted(set(a) | set(b))
    diffs = [k for k in keys if a.get(k) != b.get(k)]
    print("leaf fields compared: {}".format(len(keys)))
    print("differing fields: {}".format(len(diffs)))
    for k in diffs[:40]:
        print("  {}: {} -> {}".format(k, a.get(k), b.get(k)))
    tops = sorted({k.split(".")[2] for k in diffs})
    print("top-level sections touched: {}".format(tops))
    return 0 if tops == ["hold_profiles"] else 1


if __name__ == "__main__":
    sys.exit(main())