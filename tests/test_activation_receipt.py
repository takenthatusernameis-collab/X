#!/usr/bin/env python3
"""Regression tests for controller activation receipt stamping."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "stamp_activation_receipt.py"
)
SPEC = importlib.util.spec_from_file_location("stamp_activation_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ActivationReceiptStampTests(unittest.TestCase):
    def test_stamps_identity_without_overwriting_research_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activation_status.json"
            original = {
                "activation_id": "old",
                "status": "PARTIAL",
                "objective": "research",
                "phase": "DEEP",
                "changed": ["a.py"],
                "verified": ["check"],
                "unverified": ["execution"],
                "next": "run verifier",
            }
            path.write_text(json.dumps(original) + "\n", encoding="utf-8")

            stamped = MODULE.stamp_receipt(
                path,
                env={
                    "GITHUB_RUN_ID": "12345",
                    "GITHUB_RUN_ATTEMPT": "2",
                    "GITHUB_REF_NAME": "main",
                    "GITHUB_REPOSITORY": "owner/repo",
                },
                head_sha="abc123",
            )

            assert stamped["activation_id"] == "12345"
            assert stamped["run_attempt"] == "2"
            assert stamped["sha"] == "abc123"
            assert stamped["ref_name"] == "main"
            assert stamped["repository"] == "owner/repo"
            assert stamped["status"] == "PARTIAL"
            assert stamped["objective"] == "research"
            assert stamped["changed"] == ["a.py"]

            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded, stamped)

    def test_requires_controller_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activation_status.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                MODULE.stamp_receipt(path, env={}, head_sha="abc123")


if __name__ == "__main__":
    unittest.main()
