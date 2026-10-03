"""Probe correct CSV download URLs on stooq.com."""
import urllib.request
import time

urls = [
    ("stooq dl", "https://stooq.com/q/d/l.csv?s=aapl.us&f=dl2h5t1"),
    ("stooq dl bulk", "https://stooq.com/q/d/l.csv?s=aapl.us,msft.us,googl.us&f=dl2h5t1"),
    ("stooq dl1h5", "https://stooq.com/q/d/l.csv?s=aapl.us&f=dl1h5t1"),
    ("yahoo 5y", "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1577880000&interval=1d"),
    ("yahoo max retry", "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=max&interval=1d"),
]
for name, url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = resp.read()
            txt = d[:200].replace("\n", " ")
            print(f"{name}: HTTP {resp.status} ({len(d)} bytes) -> {txt}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:140]}")
    time.sleep(3)
