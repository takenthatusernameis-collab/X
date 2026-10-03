from datetime import date, timedelta

import numpy as np

# Read all ticker gaps (they should be identical)
with open("research/data/raw/AAPL_daily.csv") as fh:
    next(fh)
    dates = set(line.split(",")[0] for line in fh if line.strip())

# Full holiday set from check_gaps: all weekdays not in data
expected_all = []
d = "2009-01-02"
end = "2026-10-03"
while d <= end:
    dy = __import__("datetime").datetime.strptime(d, "%Y-%m-%d")
    if dy.weekday() < 5:
        expected_all.append(d)
    d = (dy + __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
all_gaps = sorted(set(expected_all) - dates)
print(f"all gap dates (weekdays not in data): {len(all_gaps)}")

# Replicate the preflight holiday calendar
def easter_sunday(y):
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    return date(y, month, day)

def observed_fixed(y, m, d):
    h = date(y, m, d)
    wd = h.weekday()
    if wd == 6:
        return h - timedelta(days=1)
    if wd == 5:
        return h + timedelta(days=1)
    return h

holidays = set()
start_y, end_y = 2009, 2026
for y in range(start_y, end_y + 1):
    holidays.add(observed_fixed(y, 1, 1))
    first_mon = date(y, 1, 1).replace(day=1)
    holidays.add(first_mon + timedelta(days=14))
    first_mon_feb = date(y, 2, 1).replace(day=1)
    holidays.add(first_mon_feb + timedelta(days=14))
    holidays.add(easter_sunday(y) - timedelta(days=2))
    import calendar
    last_day = calendar.monthrange(y, 5)[1]
    last_mon = date(y, 5, last_day)
    holidays.add(last_mon - timedelta(days=last_mon.weekday()))
    if y >= 2022:
        holidays.add(observed_fixed(y, 6, 19))
    holidays.add(observed_fixed(y, 7, 4))
    first_mon_sep = date(y, 9, 1).replace(day=1)
    holidays.add(first_mon_sep - timedelta(days=first_mon_sep.weekday()))
    first_thu = date(y, 11, 1)
    first_thu = first_thu + timedelta(days=(3 - first_thu.weekday()) % 7)
    holidays.add(first_thu + timedelta(days=21))
    holidays.add(observed_fixed(y, 12, 25))

special = [
    date(2012, 10, 29), date(2012, 10, 30),
    date(2018, 1, 19), date(2018, 1, 20),
    date(2020, 3, 16), date(2020, 3, 17), date(2020, 3, 18),
    date(2020, 3, 19), date(2020, 3, 20), date(2020, 3, 23),
    date(2020, 8, 3),
    date(2020, 10, 9),
    date(2021, 2, 3), date(2021, 2, 4),
]
holidays.update(special)

missing = [g for g in all_gaps if date.fromisoformat(g) not in holidays]
print(f"gaps NOT covered by my holiday calendar: {len(missing)}")
print("These dates are NOT in my calendar:")
for g in missing:
    print(g, date.fromisoformat(g).strftime("%a"))
print()
print("Gaps covered by my calendar:", len(all_gaps) - len(missing))
