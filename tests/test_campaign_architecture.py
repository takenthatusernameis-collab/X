#!/usr/bin/env python3
"""Static regression tests for the sequential focused-task campaign architecture."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "kilo-wakeup.yml"
ONE_SHOT = ROOT / ".github" / "workflows" / "kilo-prompt-execution-one-shot.yml"
CONTROLLER = ROOT / ".github" / "scripts" / "campaign_controller.py"


def load_controller():
    spec = importlib.util.spec_from_file_location("campaign_controller", CONTROLLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CampaignArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.workflow = WORKFLOW.read_text(encoding="utf-8")
        self.oneshot = ONE_SHOT.read_text(encoding="utf-8")
        self.controller = load_controller()

    def test_single_job_ten_sequential_agent_steps(self):
        self.assertEqual(self.workflow.count("jobs:"), 1)
        self.assertNotIn("strategy:", self.workflow)
        self.assertNotIn("matrix:", self.workflow)
        self.assertNotIn("needs:", self.workflow)
        for n in range(1, 11):
            self.assertIn(
                f"run_campaign_agent.py \
            --run-id "$GITHUB_RUN_ID" \
            --agent-number {n:02d}",
                self.workflow,
            )
            self.assertIn(
                f"campaign_controller.py select --run-id "$GITHUB_RUN_ID" --agent-number {n:02d}",
                self.workflow,
            )

    def test_odd_even_roles_are_explicit(self):
        for n in (1, 3, 5, 7, 9):
            self.assertIn(f"Agent {n:02d} — Learning Process", self.workflow)
        for n in (2, 4, 6, 8, 10):
            self.assertIn(f"Agent {n:02d} — Higher-Order Objective", self.workflow)

    def test_no_recursive_dispatch_or_retry_once_architecture(self):
        for content in (self.workflow, self.oneshot):
            self.assertNotIn("retrying the same worker workspace once", content.lower())
            self.assertNotIn("TRANSIENT_GATEWAY_RETRY_ONCE", content)
            self.assertNotIn("rerun_workflow", content)
        self.assertNotIn("report_kilo_outcome.py", self.workflow)

    def test_task_firewall_rejects_broad_tasks(self):
        for phrase in (
            "Improve the research system.",
            "continue research",
            "find something interesting",
            "advance the repository",
        ):
            self.assertFalse(self.controller.question_is_single(phrase))

    def test_task_firewall_accepts_one_bounded_question(self):
        self.assertTrue(
            self.controller.question_is_single(
                "Does the current focused-task selector improve learning efficiency relative to prior task choice?"
            )
        )

    def test_role_mapping(self):
        for n in range(1, 11):
            expected = "LEARNING_PROCESS" if n % 2 else "HIGHER_ORDER_OBJECTIVE"
            self.assertEqual(self.controller.ROLE_BY_AGENT[n], expected)

    def test_canonical_status_tokens_have_no_annotations(self):
        tokens = (
            self.controller.TASK_STATUSES
            | self.controller.DECISIONS
            | self.controller.PROCESS_DECISIONS
            | self.controller.SELECTION_OBSERVATIONS
        )
        for token in tokens:
            self.assertNotIn(" ", token)
            self.assertEqual(token, token.upper())


if __name__ == "__main__":
    unittest.main()
