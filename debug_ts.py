import urllib.request
import json

url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1=1230768000&period2=1388534400&interval=1d"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    chart = json.loads(resp.read().decode("utf-8"))
r = chart["chart"]["result"][0]
ts_ = r["timestamp"]
print("type:", type(ts_), "first 3:", ts_[:3])
print("sample types:", [type(x) for x in ts_[:3]])
print("first value repr:", repr(ts_[0]))
