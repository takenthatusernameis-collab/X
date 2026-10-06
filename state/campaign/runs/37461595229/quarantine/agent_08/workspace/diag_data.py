import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt
from research.data.preflight import load_manifest

manifest = load_manifest()
e = next(x for x in manifest["entries"] if x["ticker"] == "AAPL")
print("AAPL location:", e["location"], "first date:", e["first_available_date"], "expected bars:", e["expected_bars"])

# read raw CSV directly
import csv
with open(e["location"]) as f:
    r = csv.reader(f)
    hdr = next(r)
    print("CSV headers:", hdr)
    rows = list(r)
print("CSV rows:", len(rows))
adj = [float(row[5]) for row in rows]
print("adjclose first 3:", adj[:3])
print("adjclose last 3:", adj[-3:])
logs = np.diff(np.log(adj))
print("log returns: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(np.mean(logs), np.std(logs), np.min(logs), np.max(logs)))
# 5-day mean log returns
s5 = np.mean(logs[4:], axis=1)
print("5-day mean log ret: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(np.mean(s5), np.std(s5), np.min(s5), np.max(s5)))
print("neg 5-day means:", int(np.sum(s5 < 0)), "of", len(s5))
