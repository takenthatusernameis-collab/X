#!/usr/bin/env python3
"""Regression tests for the controller-owned Kilo outcome classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "report_kilo_outcome.py"
SPEC = importlib.util.spec_from_file_location("report_kilo_outcome", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KiloOutcomeClassifierTests(unittest.TestCase):
    def test_verified_worker_completion_is_complete(self):
        self.assertEqual(
            MODULE.classify(
                preflight="success",
                smoke="success",
                worker="success",
                verify="success",
                liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS",
                persistence="MAIN",
            ),
            (0, "COMPLETE"),
        )

    def test_postworker_verification_is_complete(self):
        self.assertEqual(
            MODULE.classify(
                preflight="success",
                smoke="success",
                worker="success",
                verify="success",
                liveness="COMPLETED_WITH_POSTWORKER_VERIFICATION",
                persistence="MAIN",
            ),
            (0, "COMPLETE"),
        )

    def test_missing_live_checkpoint_can_only_be_upgraded_by_independent_verification(self):
        common = dict(
            preflight="success",
            smoke="success",
            worker="success",
            liveness="COMPLETED_WITHOUT_SEMANTIC_CHECKPOINT",
        )
        self.assertEqual(MODULE.classify(**common, verify="success", persistence="MAIN"), (0, "PARTIAL"))
        self.assertEqual(MODULE.classify(**common, verify="failure", persistence="MAIN"), (1, "FAILED"))
        self.assertEqual(MODULE.classify(**common, verify="success", persistence="RECOVERY_BRANCH"), (1, "FAILED"))

    def test_worker_failure_with_verified_persistence_is_partial(self):
        self.assertEqual(
            MODULE.classify(
                preflight="success",
                smoke="success",
                worker="failure",
                verify="success",
                liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS",
                persistence="MAIN",
            ),
            (0, "PARTIAL"),
        )

    def test_controller_preflight_failure_blocks_activation(self):
        self.assertEqual(
            MODULE.classify(
                preflight="failure",
                smoke="success",
                worker="success",
                verify="success",
                liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS",
                persistence="MAIN",
            ),
            (1, "FAILED"),
        )


if __name__ == "__main__":
    unittest.main()
