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
WAKEUP = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "kilo-wakeup.yml"
)
ONESHOT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "kilo-prompt-execution-one-shot.yml"
)
SPEC = importlib.util.spec_from_file_location("stamp_activation_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _complete_receipt(**overrides):
    receipt = {
        "activation_id": "12345",
        "status": "PARTIAL",
        "objective": "research",
        "phase": "DEEP",
        "changed": ["a.py"],
        "verified": ["check"],
        "unverified": ["execution"],
        "next": "run verifier",
    }
    receipt.update(overrides)
    return receipt


class ActivationReceiptStampTests(unittest.TestCase):
    def test_stamps_identity_without_overwriting_research_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activation_status.json"
            original = _complete_receipt(activation_id="old")
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

    def test_empty_env_mapping_does_not_fall_back_to_process_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activation_status.json"
            path.write_text(json.dumps(_complete_receipt()) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                MODULE.stamp_receipt(
                    path,
                    env={},
                    head_sha="abc123",
                )
            self.assertIn("missing required GitHub activation identity", str(caught.exception))

    def test_validate_accepts_complete_controller_identity(self):
        receipt = _complete_receipt()
        MODULE.validate_receipt(
            receipt,
            env={"GITHUB_RUN_ID": "12345"},
        )

    def test_validate_rejects_missing_fields_with_systemexit(self):
        with self.assertRaises(SystemExit) as caught:
            MODULE.validate_receipt({}, env={"GITHUB_RUN_ID": "12345"})
        self.assertIn("missing receipt fields", str(caught.exception))

    def test_validate_rejects_identity_mismatch(self):
        with self.assertRaises(SystemExit) as caught:
            MODULE.validate_receipt(
                _complete_receipt(activation_id="other"),
                env={"GITHUB_RUN_ID": "12345"},
            )
        self.assertIn("does not match current run", str(caught.exception))

    def test_validate_rejects_invalid_status(self):
        with self.assertRaises(SystemExit) as caught:
            MODULE.validate_receipt(
                _complete_receipt(status="SUCCESS"),
                env={"GITHUB_RUN_ID": "12345"},
            )
        self.assertEqual(str(caught.exception), "invalid activation status")

    def test_validate_path_round_trip_after_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activation_status.json"
            path.write_text(json.dumps(_complete_receipt(activation_id="old")) + "\n", encoding="utf-8")
            env = {
                "GITHUB_RUN_ID": "999",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_REF_NAME": "main",
                "GITHUB_REPOSITORY": "owner/repo",
            }
            stamped = MODULE.stamp_receipt(path, env=env, head_sha="deadbeef")
            validated = MODULE.validate_receipt_path(path, env=env)
            self.assertEqual(validated["activation_id"], "999")
            self.assertEqual(validated, stamped)

    def test_wakeup_validates_receipt_after_controller_synthesis(self):
        wakeup = WAKEUP.read_text(encoding="utf-8")
        verification_start = wakeup.index("      - name: Final independent research verification")
        synthesis_start = wakeup.index("      - name: Controller-owned final synthesis")
        first_validator = wakeup.index("stamp_activation_receipt.py --validate-only")
        self.assertGreater(first_validator, synthesis_start)
        self.assertGreater(first_validator, verification_start)

    def test_wakeup_persist_uses_tested_validator_not_raise_expression(self):
        wakeup = WAKEUP.read_text(encoding="utf-8")
        oneshot = ONESHOT.read_text(encoding="utf-8")
        self.assertIn("stamp_activation_receipt.py --validate-only", wakeup)
        self.assertIn("stamp_activation_receipt.py --validate-only", oneshot)
        self.assertNotIn("if missing else None", wakeup)
        self.assertNotIn("if missing else None", oneshot)


if __name__ == "__main__":
    unittest.main()
