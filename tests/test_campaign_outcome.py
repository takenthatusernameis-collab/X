#!/usr/bin/env python3
"""Regression tests for deterministic campaign terminal outcome reporting."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "report_campaign_outcome.py"
SPEC = importlib.util.spec_from_file_location("report_campaign_outcome", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CampaignOutcomeReportTests(unittest.TestCase):
    def run_report(self, outcome: str) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "state" / "campaign" / "runs" / "123"
            report_dir.mkdir(parents=True)
            (report_dir / "final_synthesis.json").write_text(
                json.dumps(
                    {
                        "terminal_outcome": outcome,
                        "persistence_decision": "RECOVERY_BRANCH",
                        "single_highest_value_next_task": "P-001",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(MODULE, "Path", Path), mock.patch(
                "pathlib.Path.cwd", return_value=root
            ):
                with mock.patch(
                    "sys.argv", ["report_campaign_outcome.py", "--run-id", "123"]
                ):
                    with mock.patch("builtins.print"):
                        return MODULE.main()

    def test_quarantined_campaign_is_controlled_success(self):
        self.assertEqual(self.run_report("QUARANTINED"), 0)

    def test_partial_campaign_is_success(self):
        self.assertEqual(self.run_report("PARTIAL"), 0)

    def test_failed_campaign_remains_failure(self):
        self.assertEqual(self.run_report("FAILED"), 1)


if __name__ == "__main__":
    unittest.main()
