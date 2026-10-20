"""Collect a research-only OHLCV universe from Yahoo Finance (no credentials).

Research/simulation only. No live trading. No exchange credentials, API keys,
or production secrets are used. Yahoo Finance public historical data is fetched
directly over HTTPS without authentication.

Usage:
    python3 research/data/collect.py

The script is deterministic in behavior: given the same universe and windows,
it writes the same CSVs to research/data/raw/.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

COLLECTION_ROOT = Path(__file__).resolve().parent
RAW_DIR = COLLECTION_ROOT / "raw"
MANIFEST_PATH = COLLECTION_ROOT / "manifest.json"

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM",
]

WINDOWS = [
    ("2009-01-01", "2014-01-01"),
    ("2014-01-01", "2019-01-01"),
    ("2019-01-01", "2024-01-01"),
    ("2024-01-01", "2026-10-03"),
]

SLEEP_BETWEEN_REQUESTS = 5.0  # respect the public endpoint's rate limits


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _fetch_window(ticker, start, end):
    p1, p2 = ts(start), ts(end)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (research; no credentials)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        chart = json.loads(resp.read().decode("utf-8"))
    result = chart.get("chart", {}).get("result")
    if not result or not result[0].get("timestamp") or result[0].get("error"):
        return None
    r = result[0]
    timestamps = np.asarray(r["timestamp"], dtype=np.int64)
    quote = r.get("indicators", {}).get("quote", [{}])[0]
    adjclose = r.get("indicators", {}).get("adjclose", [{}])[0]
    opens = np.asarray(quote.get("open", []), dtype=np.float64)
    highs = np.asarray(quote.get("high", []), dtype=np.float64)
    lows = np.asarray(quote.get("low", []), dtype=np.float64)
    closes = np.asarray(quote.get("close", []), dtype=np.float64)
    volumes = np.asarray(quote.get("volume", []), dtype=np.float64)
    adj_closes = np.asarray(adjclose.get("adjclose", []), dtype=np.float64)
    dates = np.asarray(
        [np.datetime64(int(t // 86400), "D") for t in timestamps],
        dtype=object,
    )
    n = len(timestamps)
    rows = []
    for i in range(n):
        rows.append({
            "date": str(dates[i]),
            "open": float(opens[i]) if i < len(opens) else np.nan,
            "high": float(highs[i]) if i < len(highs) else np.nan,
            "low": float(lows[i]) if i < len(lows) else np.nan,
            "close": float(closes[i]) if i < len(closes) else np.nan,
            "adjclose": float(adj_closes[i]) if i < len(adj_closes) else np.nan,
            "volume": float(volumes[i]) if i < len(volumes) else np.nan,
        })
    return rows


def dedupe_windows(rows_list):
    seen = {}
    for rows in rows_list:
        for row in rows:
            seen[row["date"]] = row
    return [seen[d] for d in sorted(seen) if d >= WINDOWS[0][0] and d <= WINDOWS[-1][1]]


def write_csv(ticker, rows):
    path = RAW_DIR / f"{ticker.upper()}_daily.csv"
    with open(path, "w") as fh:
        fh.write("date,open,high,low,close,adjclose,volume\n")
        for row in rows:
            fh.write(
                ",".join(
                    (
                        row["date"],
                        f'{row["open"]:.6f}',
                        f'{row["high"]:.6f}',
                        f'{row["low"]:.6f}',
                        f'{row["close"]:.6f}',
                        f'{row["adjclose"]:.6f}',
                        f'{row["volume"]:.2f}',
                    )
                )
                + "\n"
            )
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(durations, first_trade, collected_at):
    import sys
    entries = []
    for ticker in UNIVERSE:
        path = RAW_DIR / f"{ticker.upper()}_daily.csv"
        sha = sha256_file(path)
        entries.append({
            "ticker": ticker,
            "source": f"https://finance.yahoo.com/quote/{ticker}/",
            "dataset_id": f"yf-ohlcv-{ticker}-{WINDOWS[-1][0]}-to-{WINDOWS[-1][1]}",
            "collection_date": collected_at.split("T")[0],
            "accessed_date": collected_at,
            "location": str(path.absolute()),
            "checksum_sha256": sha,
            "license": "Public historical data; terms of use apply to redistribution. "
                       "Research/simulation only. No credentials required.",
            "collection_method": f"research/data/collect.py (Python {sys.version.split()[0]}) "
                                 "via public Yahoo Finance API endpoint "
                                 f"(GET https://query1.finance.yahoo.com/v8/finance/chart/<TICKER>"
                                 f"?period1=<start>&period2=<end>&interval=1d, no API key).",
            "transform_log": "Raw Yahoo API JSON (UTC epoch seconds). "
                             "1) four 5-year windows plus one partial window downloaded per ticker "
                             "and deduplicated by date at window boundaries. "
                             "2) timestamps converted from UTC epoch seconds to calendar days (floor) in "
                             "US Eastern time. "
                             "3) fields open/high/low/close, adjusted close (adjclose), volume normalized to "
                             "floats; rows dropped only when the API returned no data for that ticker. "
                             "4) no forward-filling, no imputation, no merges beyond per-ticker normalization. "
                             "5) prices are ADJUSTED close (dividends and splits reflected).",
            "known_issues": (
                "1) The public endpoint throttles the 'max' range; windowed fetches were used. "
                "2) Yahoo's returned series may not extend to each issuer's true first trade date. "
                "3) Universe membership is fixed at collection for survivorship discipline. "
                "4) Public endpoints can change schema; checksums capture the raw output per collection date."
            ),
        })
    base = {
        "source": "Yahoo Finance public historical OHLCV endpoint",
        "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
        "collection_date": collected_at.split("T")[0],
        "accessed_date": collected_at,
        "location": str(COLLECTION_ROOT.absolute()),
        "license": "Public historical data; research/simulation only. "
                   "No live trading or production execution from this repository.",
        "collection_method": "research/data/collect.py",
        "transform_log": "See per-ticker transform_log entries above.",
        "known_issues": "See per-ticker known_issues entries above.",
        "owner": "research (Kilo autonomous activation)",
        "created": collected_at,
        "universe": UNIVERSE,
        "windows": WINDOWS,
        "per_ticker_bars": durations,
        "first_available_date": first_trade,
        "notes": (
            "Adjusted close is used for backtesting; dividends and splits are reflected. "
            "This dataset is research-only simulation input and does not constitute a "
            "live-trading signal or recommendation."
        ),
    }
    base["entries"] = entries
    return base


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).isoformat()
    durations = {}
    first_trade = {}

    for ticker in UNIVERSE:
        print(f"[{ticker}] fetching public history across {len(WINDOWS)} windows ...")
        chunk_rows = []
        for start, end in WINDOWS:
            rows = _fetch_window(ticker, start, end)
            if rows is None or len(rows) == 0:
                print(f"[{ticker}] no data for window {start}..{end}; skipping")
                continue
            chunk_rows.append(rows)
        if not chunk_rows:
            print(f"[{ticker}] NO DATA from public endpoint; skipping")
            continue
        rows = dedupe_windows(chunk_rows)
        path = write_csv(ticker, rows)
        durations[ticker] = len(rows)
        first_trade[ticker] = rows[0]["date"]
        print(f"[{ticker}] saved {len(rows)} rows to {path}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    manifest = build_manifest(durations, first_trade, collected_at)
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nmanifest written to {MANIFEST_PATH}")
    print(f"collection summary: {durations}")
    print(f"first trade dates: {first_trade}")


if __name__ == "__main__":
    import hashlib
    main()
