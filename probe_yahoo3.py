import urllib.request
import json
from datetime import datetime, timezone

def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

p1, p2 = ts("2024-01-01"), ts("2026-10-03")
url = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1={p1}&period2={p2}&interval=1d"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    d = resp.read().decode("utf-8")
    r = json.loads(d)["chart"]["result"][0]
    ts_ = r["timestamp"]
    print(f"2024-01-01..2026-10-03: {len(ts_)} bars")
