with open("research/data/raw/AAPL_daily.csv") as fh:
    next(fh)
    rows = [line.strip().split(",") for line in fh if line.strip()]

def find(date_str):
    for r in rows:
        if r[0] == date_str:
            return r
    return None

for d in ["2018-12-04", "2018-12-05", "2018-12-06", "2025-01-08", "2025-01-09", "2025-01-10"]:
    r = find(d)
    print(d, "->", r[0] if r else "MISSING", r[4] if r else "")
