#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".github" / "scripts" / "validate_campaign_agent.py"
spec = importlib.util.spec_from_file_location("validate_campaign_agent_quarantine_test", TARGET)
assert spec is not None and spec.loader is not None
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)
CONTROLLER = MODULE.controller


class QuarantineControllerStateTests(unittest.TestCase):
    def test_quarantine_restores_worker_changes_but_keeps_selected_queue_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)

            files = {
                "state/campaign/task_queue.json": '{"tasks":[{"task_id":"R-001","status":"OPEN"}]}\n',
                "state/campaign/current_task.json": '{"task_id":"R-001"}\n',
                "state/campaign/global_agent_counter.json": '{"next_global_agent_number":11}\n',
                "worker_output.txt": "baseline\n",
            }
            for rel, content in files.items():
                path = repo / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

            # The controller marks R-001 selected before the agent starts. The
            # pre-agent snapshot must preserve this state even though it is not
            # yet committed to the branch.
            selected_state = {
                "state/campaign/task_queue.json": '{"tasks":[{"task_id":"R-001","status":"SELECTED"}]}\n',
                "state/campaign/current_task.json": '{"task_id":"R-001","agent_number":1}\n',
                "state/campaign/global_agent_counter.json": '{"next_global_agent_number":11}\n',
            }
            for rel, content in selected_state.items():
                (repo / rel).write_text(content, encoding="utf-8")
            (repo / "worker_output.txt").write_text("untrusted worker edit\n", encoding="utf-8")

            run_root = repo / "state/campaign/runs/run-1"
            agent_dir = run_root / "agents"
            agent_dir.mkdir(parents=True)
            before = {
                "hashes": {
                    rel: MODULE.sha(repo / rel)
                    for rel in [*selected_state, "worker_output.txt"]
                },
                "controller_state": selected_state,
            }
            (agent_dir / "agent_01_before.json").write_text(json.dumps(before), encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(repo)
                result = MODULE.quarantine_workspace(
                    1, Path("state/campaign/runs/run-1"), ["invalid_agent_record"]
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(result["controller_state_status"], "restored_from_snapshot")
            for rel, expected in selected_state.items():
                self.assertEqual((repo / rel).read_text(encoding="utf-8"), expected)
            self.assertEqual((repo / "worker_output.txt").read_text(encoding="utf-8"), "baseline\n")
            self.assertTrue((run_root / "quarantine/agent_01/tracked_worktree.diff").is_file())

    def test_next_agent_does_not_reselect_task_after_invalid_record_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)

            def make_task(task_id: str, role: str) -> dict:
                return {
                    "task_id": task_id,
                    "role": role,
                    "status": "OPEN",
                    "primary_question": "What bounded evidence distinguishes the competing explanations?",
                    "bottleneck": "The controller needs one validated result.",
                    "objective": "Complete one focused controller-approved task.",
                    "scope": "Local test fixture only.",
                    "out_of_scope": "No network, external targets, or live actions.",
                    "deliverable": "One independently checked result.",
                    "success_criterion": "A deterministic result is preserved.",
                    "stop_condition": "Stop after the result is verified.",
                    "verification_requirement": "Inspect the persisted state and verify its identity.",
                    "evidence_basis": "Synthetic test fixture.",
                    "expected_information_gain": "HIGH",
                    "downstream_leverage": "HIGH",
                    "falsification_power": "HIGH",
                    "verifiability": "HIGH",
                    "effort": "LOW",
                    "complexity": "LOW",
                    "execution_risk": "LOW",
                    "tests_preceding_process": False,
                }

            queue_path = repo / "state/campaign/task_queue.json"
            current_path = repo / "state/campaign/current_task.json"
            counter_path = repo / "state/campaign/global_agent_counter.json"
            run_root = repo / "state/campaign/runs/run-1"
            agent_dir = run_root / "agents"
            agent_dir.mkdir(parents=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps({"tasks": [
                make_task("R-001", "LEARNING_PROCESS"),
                make_task("P-001", "HIGHER_ORDER_OBJECTIVE"),
            ]}), encoding="utf-8")
            current_path.write_text("{}\n", encoding="utf-8")
            counter_path.write_text('{"next_global_agent_number": 3}\n', encoding="utf-8")
            (repo / "worker_output.txt").write_text("baseline\n", encoding="utf-8")
            campaign_status = {
                "schema_version": 1,
                "run_id": "run-1",
                "agent_count": 2,
                "global_agent_numbers": {"1": 1, "2": 2},
                "status": "IN_PROGRESS",
                "phase": "SETUP",
                "completed_agents": [],
                "selected_tasks": [],
                "selection_history": [],
            }
            (run_root / "campaign_status.json").write_text(
                json.dumps(campaign_status), encoding="utf-8"
            )
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

            old_paths = (
                CONTROLLER.RUNS_ROOT,
                CONTROLLER.QUEUE_PATH,
                CONTROLLER.CURRENT_TASK,
                CONTROLLER.GLOBAL_COUNTER_PATH,
            )
            CONTROLLER.RUNS_ROOT = repo / "state/campaign/runs"
            CONTROLLER.QUEUE_PATH = queue_path
            CONTROLLER.CURRENT_TASK = current_path
            CONTROLLER.GLOBAL_COUNTER_PATH = counter_path
            previous_cwd = Path.cwd()
            try:
                os.chdir(repo)
                selected = CONTROLLER.select_task("run-1", 1)
                queue = json.loads(queue_path.read_text(encoding="utf-8"))
                selected_task = next(task for task in queue["tasks"] if task["task_id"] == selected["task_id"])
                self.assertEqual(selected_task["status"], "SELECTED")

                controller_state = {
                    str(path.relative_to(repo)): path.read_text(encoding="utf-8")
                    for path in (queue_path, current_path, counter_path)
                }
                hashes = {
                    str(path.relative_to(repo)): MODULE.sha(path)
                    for path in (
                        queue_path,
                        current_path,
                        counter_path,
                        repo / "worker_output.txt",
                        run_root / "campaign_status.json",
                        agent_dir / "agent_01_contract.json",
                    )
                }
                (agent_dir / "agent_01_before.json").write_text(
                    json.dumps({"hashes": hashes, "controller_state": controller_state}),
                    encoding="utf-8",
                )
                (repo / "worker_output.txt").write_text("untrusted worker edit\n", encoding="utf-8")
                result = MODULE.quarantine_workspace(
                    1, Path("state/campaign/runs/run-1"), ["invalid_agent_record"]
                )
                self.assertEqual(result["controller_state_status"], "restored_from_snapshot")
                queue = json.loads(queue_path.read_text(encoding="utf-8"))
                selected_task = next(task for task in queue["tasks"] if task["task_id"] == selected["task_id"])
                self.assertEqual(selected_task["status"], "SELECTED")

                # This is the controller's invalid/unverified fallback for the
                # quarantined session. The next slot should defer, not reselect it.
                (agent_dir / "agent_01.json").write_text(
                    json.dumps({
                        "task_id": selected["task_id"],
                        "decision": "UNVERIFIED",
                        "candidate_tasks": [],
                    }),
                    encoding="utf-8",
                )
                next_selected = CONTROLLER.select_task("run-1", 2)
                final_queue = json.loads(queue_path.read_text(encoding="utf-8"))
                first_task = next(task for task in final_queue["tasks"] if task["task_id"] == selected["task_id"])
                self.assertEqual(first_task["status"], "DEFERRED")
                self.assertNotEqual(next_selected["task_id"], selected["task_id"])
            finally:
                os.chdir(previous_cwd)
                (
                    CONTROLLER.RUNS_ROOT,
                    CONTROLLER.QUEUE_PATH,
                    CONTROLLER.CURRENT_TASK,
                    CONTROLLER.GLOBAL_COUNTER_PATH,
                ) = old_paths

    def test_legacy_snapshot_preserves_only_unchanged_controller_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
            rel = "state/campaign/task_queue.json"
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"tasks":[{"task_id":"P-001","status":"SELECTED"}]}\n', encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            (repo / "worker_output.txt").write_text("worker edit\n", encoding="utf-8")
            run_root = repo / "state/campaign/runs/run-2"
            agent_dir = run_root / "agents"
            agent_dir.mkdir(parents=True)
            before = {"hashes": {rel: MODULE.sha(path)}}
            (agent_dir / "agent_02_before.json").write_text(json.dumps(before), encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(repo)
                result = MODULE.quarantine_workspace(2, Path("state/campaign/runs/run-2"), ["invalid"])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(result["controller_state_status"], "legacy_unchanged_only")
            self.assertEqual(path.read_text(encoding="utf-8"), '{"tasks":[{"task_id":"P-001","status":"SELECTED"}]}\n')


if __name__ == "__main__":
    unittest.main()
