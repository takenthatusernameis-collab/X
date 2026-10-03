"""Check the date span of Yahoo's daily response."""
import json
import time
import urllib.request

url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=max&interval=1d"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    chart = json.loads(resp.read().decode("utf-8"))
r = chart["chart"]["result"][0]
ts = r["timestamp"]
import numpy as np
dates = np.array([np.datetime64(t // 86400, "D") for t in ts])
print("first bar:", dates[0], "last bar:", dates[-1], "n:", len(ts))
print("UTC start:", ts[0], "UTC end:", ts[-1])

# Try date-bounded fetch: 2005-01-01 -> 2010-01-01 (5y chunk)
def fetch_range(start_d, end_d):
    s = int(start_d.timestamp()); e = int(end_d.timestamp())
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range={e-s}&interval=1d&period1={s}&period2={e}"
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        c = json.loads(resp.read().decode("utf-8"))
    rr = c["chart"]["result"][0]
    d = [np.datetime64(t // 86400, "D") for t in rr["timestamp"]]
    return len(rr["timestamp"]), d[0] if d else None, d[-1] if d else None

for span in (("2005-01-01", "2010-01-01"), ("2010-01-01", "2015-01-01"), ("2015-01-01", "2020-01-01")):
    from datetime import datetime
    s, e = datetime.strptime(span[0], "%Y-%m-%d"), datetime.strptime(span[1], "%Y-%m-%d")
    n, first, last = fetch_range(s, e)
    print(span, "-> n:", n, "first:", first, "last:", last)
