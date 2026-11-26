"""Replicate the exact determinism comparison from the check to find the diff."""
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

all_tickers = {"AMZN": bt.load_ticker("AMZN")[0]}
per_asset = {"AMZN": {"lookback": {}}}
for lb in mcs.LOOKBACKS:
    per_asset["AMZN"]["lookback"][str(lb)] = {
        c: mcs.segment_result("AMZN", all_tickers["AMZN"], lb, c) for c in mcs.COST_ORDER}

res2 = {lb: {c: mcs.segment_result("AMZN", all_tickers["AMZN"], lb, c) for c in mcs.COST_ORDER}
        for lb in mcs.LOOKBACKS}
out1 = json.dumps({lb: per_asset["AMZN"]["lookback"][str(lb)] for lb in mcs.LOOKBACKS},
                  sort_keys=True)
out2 = json.dumps({str(lb): res2[lb] for lb in mcs.LOOKBACKS}, sort_keys=True)
print("out1 == out2:", out1 == out2)
a = json.loads(out1)
b = json.loads(out2)
for lb in mcs.LOOKBACKS:
    for c in mcs.COST_ORDER:
        if a[str(lb)][c] != b[str(lb)][c]:
            print("DIFF", lb, c)
            print("  out1", a[str(lb)][c])
            print("  out2", b[str(lb)][c])
print("DONE")