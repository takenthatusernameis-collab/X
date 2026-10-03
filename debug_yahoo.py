"""Debug Yahoo Finance API response shape."""
import json
import urllib.request

url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=max&interval=1d"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    chart = json.loads(resp.read().decode("utf-8"))

r = chart["chart"]["result"][0]
print("meta:", json.dumps(r.get("meta", {}), indent=1)[:500])
print("timestamp type:", type(r.get("timestamp")), "len:", len(r.get("timestamp", [])))
print("timestamp[:3]:", r.get("timestamp", [])[:3])
q = r.get("indicators", {}).get("quote", [{}])[0]
print("quote keys:", list(q.keys()))
print("quote open type:", type(q.get("open", [])), "len:", len(q.get("open", [])))
