"""Probe public no-key data hosts (research in-scope) via CSV downloads."""
import urllib.request
import time

hosts = [
    ("stooq root", "https://stooq.com/"),
    ("stooq bulk", "https://stooq.com/q/d/l.csv?s=aapl.us,msft.us,googl.us&f=dl&f=ohlcv"),
    ("CodeAnts AAPL", "https://raw.githubusercontent.com/CodeAntsAI/Stock-Market-Data/main/data/AAPL.csv"),
    ("grrNIFTY50", "https://raw.githubusercontent.com/grrSandeep/daily-stock-data/master/NIFTY50.csv"),
    ("World Bank GDP", "https://api.worldbank.org/v2/country/us/indicator/NY.GDP.MKTP.CD?format=json"),
]
for name, url in hosts:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            d = resp.read()
            print(f"{name}: HTTP {resp.status} ({len(d)} bytes) -> {d[:120]}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:140]}")
    time.sleep(2)
