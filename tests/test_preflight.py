"""Unit tests for the real-data preflight gate.

Deterministic and runnable with `python -m unittest`. Tests the documented-gaps
audit (check 10), the interaction with the completeness check, and determinism;
no network access is required.
"""
import json
import sys
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.data import preflight


class TestKnownGapsAudit(unittest.TestCase):
    """The known-gaps audit verifies each documented source gap is genuinely absent."""

    def test_documented_gaps_absent_in_aapl(self):
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        data = preflight.load_csv(preflight.RAW_DIR / "AAPL_daily.csv")
        doc = manifest.get("known_data_gaps", [])
        doc_dates = set(preflight.datetime.strptime(d, "%Y-%m-%d").date() for d in doc)
        aud = preflight.known_gaps_audit("AAPL", set(data["dates"]), doc_dates)
        assert aud["ok"]
        assert aud["present_but_documented_missing"] == []
        assert aud["missing_in_ticker"] == sorted(doc)

    def test_all_tickers_audit_ok(self):
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        doc = manifest.get("known_data_gaps", [])
        doc_dates = set(preflight.datetime.strptime(d, "%Y-%m-%d").date() for d in doc)
        for ticker in manifest["universe"]:
            aud = preflight.known_gaps_audit(
                ticker,
                set(preflight.load_csv(preflight.RAW_DIR / f"{ticker}_daily.csv")["dates"]),
                doc_dates,
            )
            assert aud["ok"], ticker
            assert aud["present_but_documented_missing"] == []

    def test_documented_gap_present_fails_audit(self):
        # Fabricate a manifest where a documented gap is actually present in the data.
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        data = preflight.load_csv(preflight.RAW_DIR / "AAPL_daily.csv")
        data_dates = set(data["dates"])
        fake_present = next(iter(data_dates))
        fake_dates = {preflight.datetime.strptime(fake_present, "%Y-%m-%d").date()}
        aud = preflight.known_gaps_audit("AAPL", data_dates, fake_dates)
        assert not aud["ok"]
        assert aud["present_but_documented_missing"] == [fake_present]

    def test_audit_deterministic(self):
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        data = preflight.load_csv(preflight.RAW_DIR / "AAPL_daily.csv")
        doc = manifest.get("known_data_gaps", [])
        doc_dates = set(preflight.datetime.strptime(d, "%Y-%m-%d").date() for d in doc)
        aud1 = preflight.known_gaps_audit("AAPL", set(data["dates"]), doc_dates)
        aud2 = preflight.known_gaps_audit("AAPL", set(data["dates"]), doc_dates)
        assert aud1 == aud2

    def test_known_gap_absent_is_reported_as_missing(self):
        # Every genuinely missing documented gap is listed in missing_in_ticker.
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        data = preflight.load_csv(preflight.RAW_DIR / "AAPL_daily.csv")
        doc = manifest.get("known_data_gaps", [])
        doc_dates = set(preflight.datetime.strptime(d, "%Y-%m-%d").date() for d in doc)
        aud = preflight.known_gaps_audit("AAPL", set(data["dates"]), doc_dates)
        assert sorted(aud["missing_in_ticker"]) == sorted(doc)


class TestPreflightIntegration(unittest.TestCase):
    """End-to-end preflight behavior with modified manifests; no network."""

    def test_preflight_passes_as_is(self):
        buf = StringIO()
        sys.stdout = buf
        try:
            rc = preflight.main()
        finally:
            sys.stdout = sys.__stdout__
        assert rc == 0
        assert "PREFLIGHT PASSED" in buf.getvalue()

    def test_manifest_staleness_detected(self):
        # A manifest that lists a date present in the data as a documented gap
        # must fail on check 10.
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        data = preflight.load_csv(preflight.RAW_DIR / "AAPL_daily.csv")
        present_date = set(data["dates"]).pop()
        manifest["known_data_gaps"].append(present_date)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(manifest, fh)
            tmp = fh.name
        old_path = preflight.MANIFEST_PATH
        preflight.MANIFEST_PATH = Path(tmp)
        buf = StringIO()
        sys.stdout = buf
        try:
            rc = preflight.main()
        finally:
            preflight.MANIFEST_PATH = old_path
            Path(tmp).unlink()
            sys.stdout = sys.__stdout__
        assert rc == 1
        out = buf.getvalue()
        assert "[FAIL] 10. Known gaps audit" in out, out
        assert present_date in out

    def test_unexplained_gap_fails_preflight(self):
        # A genuinely missing gap that is NOT in known_data_gaps must be reported
        # as unexplained and fail check 2 (completeness).
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        removed = manifest["known_data_gaps"].pop()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(manifest, fh)
            tmp = fh.name
        old_path = preflight.MANIFEST_PATH
        preflight.MANIFEST_PATH = Path(tmp)
        buf = StringIO()
        sys.stdout = buf
        try:
            rc = preflight.main()
        finally:
            preflight.MANIFEST_PATH = old_path
            Path(tmp).unlink()
            sys.stdout = sys.__stdout__
        assert rc == 1
        out = buf.getvalue()
        assert "[FAIL] 2. Completeness" in out, out
        assert removed in out

    def test_unparsable_gap_date_errors(self):
        # Corrupt known_data_gaps must be caught rather than swallowed.
        with open(preflight.MANIFEST_PATH) as fh:
            manifest = preflight.json.load(fh)
        manifest["known_data_gaps"].append("not-a-date")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(manifest, fh)
            tmp = fh.name
        old_path = preflight.MANIFEST_PATH
        preflight.MANIFEST_PATH = Path(tmp)
        buf = StringIO()
        sys.stdout = buf
        try:
            rc = preflight.main()
        finally:
            preflight.MANIFEST_PATH = old_path
            Path(tmp).unlink()
            sys.stdout = sys.__stdout__
        assert rc == 1
        out = buf.getvalue()
        assert "unparsable gap date" in out, out

    def test_preflight_summary_reports_bars(self):
        buf = StringIO()
        sys.stdout = buf
        try:
            rc = preflight.main()
        finally:
            sys.stdout = sys.__stdout__
        assert rc == 0
        assert "=== Preflight summary ===" in buf.getvalue()
        assert "bars per ticker: {'AAPL': 4465" in buf.getvalue()
