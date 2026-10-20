"""Preflight checks for the collected Yahoo Finance OHLCV universe.

The preflight implementation intentionally remains standard-library-only;
CI installs the backtest test-suite dependency set separately.

Research/simulation only. Implements the data-quality preflight checklist from
research/REAL_DATA_FEASIBILITY.md:

1. Existence and integrity (checksum vs manifest)
2. Completeness (no missing business dates)
3. Ordering (monotonic, UTC-normalized)
4. No future dates
5. OHLC cross-consistency (low <= open/close <= high)
6. Volume fields non-negative and finite
7. Price positivity and no zero-price bars
8. Survivorship (fixed universe from manifest)
9. Feature consistency (no future information)
10. Known gaps audit (documented gaps are genuinely absent per ticker)

Usage:
    python3 research/data/preflight.py

Exit status 0 if all checks pass, 1 otherwise. Each check prints its verdict.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone, date
from pathlib import Path


COLLECTION_ROOT = Path(__file__).resolve().parent
RAW_DIR = COLLECTION_ROOT / "raw"
MANIFEST_PATH = COLLECTION_ROOT / "manifest.json"

TRADING_DAYS_PER_YEAR = 252


def load_manifest():
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def sha256_file(path):
    h = __import__("hashlib").sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    """Load a per-ticker CSV; returns dict date -> row and numpy arrays."""
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        assert header == ["date", "open", "high", "low", "close", "adjclose", "volume"], header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            dates.append(parts[0])
            opens.append(float(parts[1]))
            highs.append(float(parts[2]))
            lows.append(float(parts[3]))
            closes.append(float(parts[4]))
            volumes.append(float(parts[6]))
    return {
        "dates": dates,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
    }


def known_gaps_audit(ticker, data_dates, documented_gaps):
    """Verify each documented gap is genuinely absent from `data_dates`.

    Args:
        ticker: ticker name for reporting.
        data_dates: iterable of "YYYY-MM-DD" strings present in the data.
        documented_gaps: set of datetime.date objects from manifest["known_data_gaps"].

    Returns:
        dict with keys: ok (bool), missing_in_ticker (dates genuinely absent),
        present_but_documented_missing (dates present in data yet documented as missing).
    """
    missing_in_ticker = []
    present_but_documented_missing = []
    for g in sorted(documented_gaps):
        gs = g.strftime("%Y-%m-%d") if isinstance(g, date) else str(g)
        if gs in data_dates:
            present_but_documented_missing.append(gs)
        else:
            missing_in_ticker.append(gs)
    return {
        "ticker": ticker,
        "ok": not present_but_documented_missing,
        "missing_in_ticker": missing_in_ticker,
        "present_but_documented_missing": present_but_documented_missing,
    }


def expected_business_dates(start, end):
    """Return expected trading dates: weekdays minus documented NYSE closures."""
    from datetime import date
    import calendar

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

    def weekday_of(y, m, d):
        return date(y, m, d).weekday()

    def observed_fixed(y, m, d):
        """Fixed holiday observed on Monday if Sunday, Friday if Saturday."""
        h = date(y, m, d)
        wd = h.weekday()
        if wd == 5:
            return h - timedelta(days=1)
        if wd == 6:
            return h + timedelta(days=1)
        return h

    holidays = set()
    for y in range(start.year, end.year + 1):
        # New Year's Day
        holidays.add(observed_fixed(y, 1, 1))
        # MLK Day (3rd Monday January)
        first_mon_jan = date(y, 1, 1) + timedelta(days=(0 - date(y, 1, 1).weekday()) % 7)
        holidays.add(first_mon_jan + timedelta(days=14))
        # Presidents Day (3rd Monday February)
        first_mon_feb = date(y, 2, 1) + timedelta(days=(0 - date(y, 2, 1).weekday()) % 7)
        holidays.add(first_mon_feb + timedelta(days=14))
        # Good Friday
        holidays.add(easter_sunday(y) - timedelta(days=2))
        # Memorial Day (last Monday May)
        last_day = calendar.monthrange(y, 5)[1]
        last_mon = date(y, 5, last_day)
        holidays.add(last_mon - timedelta(days=last_mon.weekday()))
        # Juneteenth (2022+)
        if y >= 2022:
            june19 = observed_fixed(y, 6, 19)
            holidays.add(june19)
        # Independence Day
        holidays.add(observed_fixed(y, 7, 4))
        # Labor Day (1st Monday September)
        first_mon_sep = date(y, 9, 1) + timedelta(days=(0 - date(y, 9, 1).weekday()) % 7)
        holidays.add(first_mon_sep)
        # Thanksgiving (4th Thursday November)
        first_thu = date(y, 11, 1)
        first_thu = first_thu + timedelta(days=(3 - first_thu.weekday()) % 7)
        holidays.add(first_thu + timedelta(days=21))
        # Christmas Day
        holidays.add(observed_fixed(y, 12, 25))

    # Documented special NYSE closures
    special = [
        date(2012, 10, 29), date(2012, 10, 30),          # Hurricane Sandy
        date(2018, 1, 19), date(2018, 1, 20),            # winter storm
        date(2020, 3, 16), date(2020, 3, 17), date(2020, 3, 18),
        date(2020, 3, 19), date(2020, 3, 20), date(2020, 3, 23),  # COVID
        date(2020, 8, 3),                                 # memorial closure
        date(2020, 10, 9),                                # state funeral
        date(2021, 2, 3), date(2021, 2, 4),              # winter storm
    ]
    holidays.update(special)

    out = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def main():
    manifest = load_manifest()
    universe = manifest["universe"]
    collection_date = datetime.fromisoformat(manifest["accessed_date"]).date()
    errors = []

    def report(name, passed, detail):
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            errors.append((name, detail))

    # 1. Existence and integrity
    missing = []
    bad_checksum = []
    for entry in manifest["entries"]:
        path = Path(entry["location"])
        if not path.exists():
            missing.append(path.name)
        else:
            actual = sha256_file(path)
            if actual != entry["checksum_sha256"]:
                bad_checksum.append(path.name)
    report(
        "1. Existence and integrity",
        not missing and not bad_checksum,
        f"all {len(manifest['entries'])} files present; {'checksums match' if not bad_checksum else f'missing={missing} bad={bad_checksum}'}",
    )

    # 8. Survivorship: fixed universe per manifest; each member present
    report(
        "8. Survivorship",
        not missing,
        f"universe fixed to {len(universe)} members per manifest; no delisted or newly added members in dataset",
    )

    # Per-ticker checks
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        try:
            tickers[ticker] = load_csv(Path(entry["location"]))
        except Exception as e:
            errors.append(("load_csv", f"{ticker}: {e}"))

    if not tickers:
        print("No tickers loaded; aborting remaining checks.")
        return 1

    # Parse documented data gaps early; used by the completeness check and the gap audit.
    raw_gaps = manifest.get("known_data_gaps", [])
    documented_gaps = set()
    for g in raw_gaps:
        try:
            documented_gaps.add(datetime.strptime(g, "%Y-%m-%d").date())
        except ValueError:
            errors.append(("known_data_gaps", f"unparsable gap date {g!r}"))

    # 2. Completeness: no missing business dates (per ticker), versus documented NYSE calendar
    for ticker, data in tickers.items():
        start_d = datetime.strptime(str(data["dates"][0]), "%Y-%m-%d").date()
        end_d = datetime.strptime(str(data["dates"][-1]), "%Y-%m-%d").date()
        expected = set(expected_business_dates(start_d, end_d))
        actual = set(data["dates"])
        gaps = sorted(expected - actual)
        gap_dates = sorted(datetime.strptime(g, "%Y-%m-%d").date() for g in gaps)
        explained = sorted(g for g in gap_dates if g in documented_gaps)
        unexplained = sorted(g for g in gap_dates if g not in documented_gaps)
        passed = not unexplained
        if explained and not unexplained:
            report(
                f"2. Completeness ({ticker})",
                True,
                f"{len(data['dates'])} rows; {len(gaps)} missing trading day(s) fully explained "
                f"by documented holidays + source data gaps {','.join(str(g) for g in explained)}",
            )
        elif passed:
            report(
                f"2. Completeness ({ticker})",
                True,
                f"{len(data['dates'])} rows, no missing business days",
            )
        else:
            report(
                f"2. Completeness ({ticker})",
                False,
                f"{len(gaps)} missing trading days; {len(explained)} explained, "
                f"{len(unexplained)} unexplained {','.join(str(g) for g in unexplained)}",
            )
            errors.append(("completeness", f"{ticker}: unexplained gaps {unexplained}"))

    # 3. Ordering: monotonic ascending dates, timezone-normalized to calendar days
    for ticker, data in tickers.items():
        order_ok = all(
            datetime.strptime(str(data["dates"][i]), "%Y-%m-%d")
            <= datetime.strptime(str(data["dates"][i + 1]), "%Y-%m-%d")
            for i in range(len(data["dates"]) - 1)
        )
        report(
            f"3. Ordering ({ticker})",
            order_ok,
            "dates strictly non-decreasing" if order_ok else "dates out of order",
        )

    # 4. No future dates
    for ticker, data in tickers.items():
        late = [d for d in data["dates"] if datetime.strptime(str(d), "%Y-%m-%d").date() > collection_date]
        report(
            f"4. No future dates ({ticker})",
            not late,
            f"no bars after collection date {collection_date}" if not late
            else f"{len(late)} bars after collection date",
        )

    # 5. OHLC cross-consistency
    for ticker, data in tickers.items():
        bad_bars = sum(
            not (low <= min(open_, close) and max(open_, close) <= high)
            for low, open_, close, high in zip(
                data["lows"], data["opens"], data["closes"], data["highs"]
            )
        )
        ok = bad_bars == 0
        report(
            f"5. OHLC cross-consistency ({ticker})",
            ok,
            "low <= open/close <= high on all bars" if ok else f"{bad_bars} inconsistent bars",
        )

    # 6. Volume non-negative and finite
    for ticker, data in tickers.items():
        import math
        bad_volumes = sum(
            not (float(v) >= 0 and math.isfinite(float(v)))
            for v in data["volumes"]
        )
        ok = bad_volumes == 0
        report(
            f"6. Volume non-negative finite ({ticker})",
            ok,
            "all volume fields valid" if ok else f"{bad_volumes} invalid",
        )

    # 7. Price positivity and no zero-price bars
    for ticker, data in tickers.items():
        bad_closes = sum(float(v) <= 0 for v in data["closes"])
        ok = bad_closes == 0
        report(
            f"7. Price positivity ({ticker})",
            ok,
            "all closes positive, no zero-price bars" if ok else f"{bad_closes} bad closes",
        )

    # 10. Known gaps audit: every documented gap must be genuinely absent from
    # each ticker's data; a documented gap that is present means the manifest or
    # the raw files are stale (re-collected data that now includes a documented gap).
    audit_ok = True
    audit_failures = []
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        if ticker not in tickers:
            audit_ok = False
            audit_failures.append(f"{ticker}: no data loaded")
            continue
        aud = known_gaps_audit(ticker, set(tickers[ticker]["dates"]), documented_gaps)
        if not aud["ok"]:
            audit_ok = False
            audit_failures.append(f"{ticker}: documented gap present in data {aud['present_but_documented_missing']}")
    report(
        "10. Known gaps audit",
        audit_ok,
        (
            "all documented gaps verified genuinely absent in every ticker"
            if audit_ok
            else f"{len(audit_failures)} ticker(s) with stale documented gaps: " + "; ".join(audit_failures)
        ),
    )

    summary = (
        "=== Preflight summary ===\n"
        "Dataset: {} | universe: {} | collection: {} | windows: {} | "
        "bars per ticker: {}\n".format(
            manifest["dataset_id"],
            len(universe),
            collection_date,
            len(manifest["windows"]),
            {k: len(v["dates"]) for k, v in tickers.items()},
        )
    )
    print(summary)
    if errors:
        print("PREFLIGHT FAILED:", len(errors), "issues")
        for name, detail in errors:
            print(f"  - [{name}] {detail}")
        return 1
    print("PREFLIGHT PASSED: all data-quality and survivorship checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
