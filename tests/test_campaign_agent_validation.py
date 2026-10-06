#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".github" / "scripts" / "validate_campaign_agent.py"

spec = importlib.util.spec_from_file_location("validate_campaign_agent", TARGET)
assert spec is not None and spec.loader is not None
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


class CampaignAgentValidationTests(unittest.TestCase):
    def test_controller_fills_missing_identity_metadata(self):
        contract = {"global_agent_number": "MINI-01"}
        record = {"role": "LEARNING_PROCESS", "task_id": "P-001", "objective": "bounded"}
        normalized, filled = MODULE.bind_controller_identity(record, 1, contract)
        self.assertEqual(normalized["agent_number"], 1)
        self.assertEqual(normalized["campaign_slot"], 1)
        self.assertEqual(normalized["global_agent_number"], "MINI-01")
        self.assertEqual(filled, ["agent_number", "campaign_slot", "global_agent_number"])

    def test_controller_does_not_overwrite_identity_metadata(self):
        contract = {"global_agent_number": "MINI-01"}
        record = {"agent_number": 9, "campaign_slot": 9, "global_agent_number": "WRONG"}
        normalized, filled = MODULE.bind_controller_identity(record, 1, contract)
        self.assertEqual(normalized, record)
        self.assertEqual(filled, [])


if __name__ == "__main__":
    unittest.main()
