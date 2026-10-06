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
RUNNER = ROOT / ".github" / "scripts" / "run_campaign_agent.py"


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
        self.runner = RUNNER.read_text(encoding="utf-8")

    def test_single_job_ten_sequential_agent_steps(self):
        self.assertEqual(self.workflow.count("jobs:"), 1)
        self.assertNotIn("strategy:", self.workflow)
        self.assertNotIn("matrix:", self.workflow)
        self.assertNotIn("needs:", self.workflow)
        self.assertEqual(self.workflow.count("run_campaign_agent.py \\"), 10)
        for n in range(1, 11):
            self.assertIn(f"--agent-number {n:02d}", self.workflow)
            self.assertIn(
                f"campaign_controller.py select --run-id \"$GITHUB_RUN_ID\" --agent-number {n:02d}",
                self.workflow,
            )

    def test_odd_even_roles_are_explicit(self):
        for n in (1, 3, 5, 7, 9):
            self.assertIn(f"Campaign Slot {n:02d} — Learning Process — Fresh Kilo session", self.workflow)
        for n in (2, 4, 6, 8, 10):
            self.assertIn(f"Campaign Slot {n:02d} — Higher-Order Objective — Fresh Kilo session", self.workflow)

    def test_no_recursive_dispatch_or_retry_once_architecture(self):
        for value in (self.workflow, self.oneshot):
            self.assertNotIn("retrying the same worker workspace once", value.lower())
            self.assertNotIn("TRANSIENT_GATEWAY_RETRY_ONCE", value)
            self.assertNotIn("rerun_workflow", value)
            self.assertNotIn("report_kilo_outcome.py", value)

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

    def test_global_identity_counter_is_durable(self):
        import json
        counter = json.loads((ROOT / "state" / "campaign" / "global_agent_counter.json").read_text(encoding="utf-8"))
        self.assertEqual(counter["schema_version"], 1)
        self.assertGreaterEqual(counter["next_global_agent_number"], 1)

        original = self.controller.GLOBAL_COUNTER_PATH
        try:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmp:
                self.controller.GLOBAL_COUNTER_PATH = Path(tmp) / "counter.json"
                first = self.controller.allocate_global_agent_numbers("run-a", 10, "PERSISTENT")
                second = self.controller.allocate_global_agent_numbers("run-b", 10, "PERSISTENT")
                self.assertEqual(first["1"], 1)
                self.assertEqual(first["10"], 10)
                self.assertEqual(second["1"], 11)
                self.assertEqual(second["10"], 20)
                mini = self.controller.allocate_global_agent_numbers("mini", 2, "EPHEMERAL")
                self.assertEqual(mini["1"], "MINI-01")
                self.assertEqual(mini["2"], "MINI-02")
                saved = json.loads(self.controller.GLOBAL_COUNTER_PATH.read_text(encoding="utf-8"))
                self.assertEqual(saved["next_global_agent_number"], 21)
        finally:
            self.controller.GLOBAL_COUNTER_PATH = original

    def test_global_identity_is_used_in_prompt_contract(self):
        self.assertIn("Global Agent", self.controller.render_prompt.__doc__ or "")

    def test_role_mapping(self):
        for n in range(1, 11):
            expected = "LEARNING_PROCESS" if n % 2 else "HIGHER_ORDER_OBJECTIVE"
            self.assertEqual(self.controller.ROLE_BY_AGENT[n], expected)

    def test_fresh_session_runner_has_no_retry(self):
        self.assertIn("[kilo_bin, \"run\", \"--model\", args.model, \"--auto\", prompt]", self.runner)
        self.assertIn('\"automatic_retry\": False', self.runner)
        self.assertIn('\"previous_session_context_reused\": False', self.runner)
        self.assertNotIn("kilo retry", self.runner.lower())
        self.assertNotIn("retry_pid", self.runner)

    def test_task_queue_has_bounded_contracts(self):
        import json
        queue = json.loads((ROOT / "state" / "campaign" / "task_queue.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(queue["tasks"]), 10)
        for task in queue["tasks"]:
            self.controller.validate_task(task)

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
