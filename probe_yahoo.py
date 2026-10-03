import urllib.request
import json
import time
from datetime import datetime, timezone

def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

# Fixed historical windows for AAPL
windows = [("2009-01-01", "2014-01-01"), ("2014-01-01", "2019-01-01"), ("2019-01-01", "2024-01-01")]
for start, end in windows:
    p1, p2 = ts(start), ts(end)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1={p1}&period2={p2}&interval=1d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = resp.read().decode("utf-8")
            r = json.loads(d)["chart"]["result"][0]
            ts_ = r["timestamp"]
            print(f"{start}..{end}: {len(ts_)} bars")
    except Exception as e:
        print(f"{start}..{end}: FAILED {type(e).__name__}: {str(e)[:140]}")
    time.sleep(5)
