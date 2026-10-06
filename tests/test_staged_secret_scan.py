#!/usr/bin/env python3
"""Behavioral tests for staged credential scanning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".github" / "scripts" / "scan_staged_secrets.py"


def load_scanner():
    spec = importlib.util.spec_from_file_location("scan_staged_secrets", TARGET)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StagedSecretScanTests(unittest.TestCase):
    def setUp(self):
        self.scanner = load_scanner()

    def test_benign_kilo_filename_reference_is_not_a_secret(self):
        text = (
            "state/smoke/kilo_campaign_dummy_trigger.txt: "
            "37d532d659d099080772de5a0ca27e1a6f0f6e63c6f5a06dc58039db5a617b38"
        )
        self.assertEqual(self.scanner.scan_text(text), [])

    def test_kilo_jwt_key_is_detected_only_in_auth_context(self):
        token = "eyJ" + "a" * 18 + "." + "b" * 18 + "." + "c" * 18
        self.assertIn(
            "kilo_jwt_key",
            self.scanner.scan_text(f"KILO_API_KEY={token}"),
        )


    def test_kilo_bearer_jwt_is_detected(self):
        token = "eyJ" + "a" * 18 + "." + "b" * 18 + "." + "c" * 18
        self.assertIn(
            "kilo_jwt_key",
            self.scanner.scan_text(f"Authorization: Bearer {token}"),
        )

    def test_github_classic_token_is_detected(self):
        self.assertIn(
            "github_classic_token",
            self.scanner.scan_text("TOKEN=ghp_" + "A" * 30),
        )

    def test_private_key_marker_is_detected(self):
        self.assertIn(
            "private_key",
            self.scanner.scan_text("-----BEGIN OPENSSH PRIVATE KEY-----"),
        )


if __name__ == "__main__":
    unittest.main()
