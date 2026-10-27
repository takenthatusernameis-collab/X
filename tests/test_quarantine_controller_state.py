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
