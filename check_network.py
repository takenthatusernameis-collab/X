"""Check connectivity to research-in-scope data sources (no credentials needed)."""
import urllib.request
import json
import time

targets = {
    "yahoo_finance": "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d&interval=1d",
    "fred_series": "https://api.fred.stlouisfed.org/series?file_type=json",
    "french_factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/fam_F_Daily_P1_Monthly.txt",
}

for name, url in targets.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if name == "fred_series":
                head = data.decode("utf-8", "replace")[:100].replace("\n", " ")
                print(f"{name}: HTTP {resp.status} ({len(data)} bytes) -> {head}")
            else:
                print(f"{name}: HTTP {resp.status} ({len(data)} bytes) -> {data[:80]}")
    except Exception as e:
        print(f"{name}: FAILED -> {type(e).__name__}: {str(e)[:120]}")
    time.sleep(1)
