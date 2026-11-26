"""Debug the determinism comparison for the cost-sensitivity check."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt
import importlib.util

spec = importlib.util.spec_from_file_location(
    "mcs", "research/checks/momentum_cost_sensitivity.py")
mcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcs)

bars, _ = bt.load_ticker("AMZN")
a = {str(lb): {c: mcs.segment_result("AMZN", bars, lb, c) for c in mcs.COST_ORDER}
     for lb in mcs.LOOKBACKS}
b = {str(lb): {c: mcs.segment_result("AMZN", bars, lb, c) for c in mcs.COST_ORDER}
     for lb in mcs.LOOKBACKS}
ja = json.dumps(a, sort_keys=True)
jb = json.dumps(b, sort_keys=True)
print("two fresh runs identical:", ja == jb)
if ja != jb:
    for lb in mcs.LOOKBACKS:
        for c in mcs.COST_ORDER:
            if a[str(lb)][c] != b[str(lb)][c]:
                print("DIFF", lb, c)
                print("  a", a[str(lb)][c])
                print("  b", b[str(lb)][c])
print("DONE")