from datetime import date, datetime, timedelta

import numpy as np

# Read AAPL dates
with open("research/data/raw/AAPL_daily.csv") as fh:
    next(fh)
    dates = [line.split(",")[0] for line in fh if line.strip()]

expected = []
d = "2009-01-02"
end = "2026-10-03"
while d <= end:
    dy = datetime.strptime(d, "%Y-%m-%d")
    if dy.weekday() < 5:
        expected.append(d)
    d = (dy + timedelta(days=1)).strftime("%Y-%m-%d")

actual = set(dates)
gaps = sorted(set(expected) - actual)
print(f"expected weekdays: {len(expected)}, actual bars: {len(dates)}, gaps: {len(gaps)}")
print("GAP DATES:")
for g in gaps:
    print(g, date.fromisoformat(g).strftime("%a"))
