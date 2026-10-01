from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.environment import RepositoryEnvironment
from coding_agent_policy_opt.harness.policy import AdaptiveHarnessPolicy
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.harness.smoke import smoke_decisions
from coding_agent_policy_opt.models import ActionType, AgentAction, ErrorCode, PolicyOutput
from tests.helpers import SmokeRepository


class PatchConflictTest(unittest.TestCase):
    def setUp(self):
        self.fixture = SmokeRepository()
        self.env = self.fixture.environment
        self.file = self.fixture.root / "calculator.py"
        self.read = AgentAction.from_dict(smoke_decisions()[1]["action"])
        self.edit = AgentAction.from_dict(smoke_decisions()[2]["action"])

    def tearDown(self):
        self.fixture.close()

    def test_requires_read_and_blocks_success_until_read(self):
        result = self.env.execute(self.edit)
        self.assertEqual(result.error_code, ErrorCode.FILE_NOT_READ)
        finish = AgentAction(ActionType.FINISH, {"status": "success", "summary": "done"})
        self.assertFalse(self.env.execute(finish).ok)
        self.assertTrue(self.env.execute(AgentAction(ActionType.FINISH, {"status": "give_up"})).ok)
        read = self.env.execute(self.read)
        self.assertEqual(read.data["content_hash"], hashlib.sha256(self.file.read_bytes()).hexdigest())
        self.assertTrue(self.env.execute(self.edit).ok)

    def test_external_edit_blocks_even_when_git_would_accept_patch(self):
        self.file.write_bytes(b"# owner: agent\n\n" + self.file.read_bytes())
        self.edit = AgentAction(ActionType.APPLY_PATCH, {"patch": self.edit.args["patch"].replace("@@ -1,3 +1,3 @@", "@@ -3,3 +3,3 @@")})
        self.env.execute(self.read)
        self.file.write_bytes(self.file.read_bytes().replace(b"# owner: agent", b"# human addition"))
        human = self.file.read_bytes()
        baseline = subprocess.run(["git", "apply", "--check", "-"], cwd=self.fixture.root,
                                  input=self.edit.args["patch"].encode(), capture_output=True)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        result = self.env.execute(self.edit)
        self.assertEqual(result.error_code, ErrorCode.PATCH_CONFLICT)
        self.assertEqual(self.file.read_bytes(), human)
        self.assertEqual(result.data["recovery_action"], "read_then_replan")
        self.assertNotIn("apply_patch", AdaptiveHarnessPolicy().decide({
            "environment_state": self.env.state_payload(),
        }).allowed_tools)
        self.env.execute(self.read)
        self.assertTrue(self.env.execute(self.edit).ok)
        self.assertIn(b"# human addition", self.file.read_bytes())
        self.assertIn(b"return a + b", self.file.read_bytes())

    def test_one_conflict_blocks_entire_multifile_patch(self):
        other = self.fixture.root / "settings.txt"
        other.write_bytes(b"enabled=false\n")
        self.env.execute(self.read)
        self.env.execute(AgentAction(ActionType.READ_FILE, {
            "path": "settings.txt", "start_line": 1, "end_line": 10,
        }))
        other.write_bytes(b"enabled=human\n")
        before = self.file.read_bytes()
        combined = self.edit.args["patch"] + (
            "diff --git a/settings.txt b/settings.txt\n--- a/settings.txt\n+++ b/settings.txt\n"
            "@@ -1 +1 @@\n-enabled=false\n+enabled=true\n"
        )
        self.assertEqual(self.env.execute(AgentAction(ActionType.APPLY_PATCH, {
            "patch": combined,
        })).error_code, ErrorCode.PATCH_CONFLICT)
        self.assertEqual(self.file.read_bytes(), before)
        self.assertEqual(other.read_bytes(), b"enabled=human\n")

    def test_edit_during_preflight_is_detected(self):
        self.env.execute(self.read)
        real_run = subprocess.run

        def racing_run(command, **kwargs):
            result = real_run(command, **kwargs)
            if command[:3] == ["git", "apply", "--check"]:
                self.file.write_bytes(self.file.read_bytes() + b"# saved during preflight\n")
            return result

        with patch("coding_agent_policy_opt.harness.environment.subprocess.run", side_effect=racing_run):
            result = self.env.execute(self.edit)
        self.assertEqual(result.error_code, ErrorCode.PATCH_CONFLICT)
        self.assertIn(b"return a - b", self.file.read_bytes())

    def test_restart_does_not_trust_previous_reads(self):
        self.env.execute(self.read)
        restarted = RepositoryEnvironment(self.fixture.root, self.fixture.profile)
        self.assertEqual(restarted.execute(self.edit).error_code, ErrorCode.FILE_NOT_READ)
        self.env.reset_read_evidence()
        self.assertEqual(self.env.execute(self.edit).error_code, ErrorCode.FILE_NOT_READ)

    @staticmethod
    def new_file_patch():
        return AgentAction(ActionType.APPLY_PATCH, {"patch": (
            "diff --git a/new.txt b/new.txt\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+agent\n"
        )})

    def test_new_file_never_overwrites_an_existing_file(self):
        target = self.fixture.root / "new.txt"
        target.write_bytes(b"human\n")
        result = self.env.execute(self.new_file_patch())
        self.assertEqual(result.error_code, ErrorCode.PATCH_CONFLICT)
        self.assertEqual(result.data["conflicts"][0]["reason"], "created")
        self.assertEqual(target.read_bytes(), b"human\n")

    def test_byte_level_change_is_detected_even_with_same_text_lines(self):
        self.env.execute(self.read)
        original = self.file.read_bytes()
        self.file.write_bytes(original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertEqual(self.env.execute(self.edit).error_code, ErrorCode.PATCH_CONFLICT)

    def test_index_failure_reports_applied_patch_without_reversing_human_edits(self):
        original_git = self.env._git
        target = self.fixture.root / "new.txt"

        def fail_index(*args, **kwargs):
            if args[:2] == ("add", "--intent-to-add"):
                target.write_bytes(b"agent\nhuman\n")
                return subprocess.CompletedProcess(args, 1, "", "simulated index failure")
            return original_git(*args, **kwargs)

        with patch.object(self.env, "_git", side_effect=fail_index):
            result = self.env.execute(self.new_file_patch())
        self.assertFalse(result.ok)
        self.assertTrue(result.state_changed)
        self.assertTrue(result.data["patch_applied"])
        self.assertEqual(target.read_bytes(), b"agent\nhuman\n")

    def test_metadata_only_section_cannot_bypass_guard(self):
        self.env.execute(self.read)
        malicious = self.edit.args["patch"] + (
            "diff --git a/calculator.py b/stolen.py\n"
            "similarity index 100%\nrename from calculator.py\nrename to stolen.py\n"
        )
        self.assertEqual(self.env.execute(AgentAction(ActionType.APPLY_PATCH, {
            "patch": malicious,
        })).error_code, ErrorCode.INVALID_ARGUMENT)
        self.assertTrue(self.file.exists())
        self.assertIn(b"return a - b", self.file.read_bytes())

    def test_deleted_file_conflicts_and_can_be_observed(self):
        self.env.execute(self.read)
        self.file.unlink()
        self.assertEqual(self.env.execute(self.edit).error_code, ErrorCode.PATCH_CONFLICT)
        result = self.env.execute(self.read)
        self.assertTrue(result.ok)
        self.assertFalse(result.data["exists"])
        self.assertEqual(self.env.state_payload()["patch_conflicts"], [])

    def test_harness_passes_conflict_to_policy_and_records_recovery(self):
        fixture_file = self.file
        fixture_file.write_bytes(b"# owner: agent\n\n" + fixture_file.read_bytes())
        decisions = smoke_decisions()
        decisions[2]["action"]["args"]["patch"] = decisions[2]["action"]["args"]["patch"].replace("@@ -1,3 +1,3 @@", "@@ -3,3 +3,3 @@")

        class ReplanningPolicy:
            calls = 0

            def decide(self, context):
                sequence = [1, 2, 1, 2, 3, 4, 5]
                if self.calls == 1:
                    fixture_file.write_bytes(fixture_file.read_bytes().replace(b"# owner: agent", b"# human edit"))
                if self.calls == 2:
                    assert context["history"][-1]["observation"]["error_code"] == "PATCH_CONFLICT"
                    assert context["harness_policy"]["context_strategy"] == "conflict_replanning"
                decision = decisions[sequence[self.calls]]
                self.calls += 1
                return PolicyOutput(payload=decision)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            episode = AgentHarness(self.fixture.config, self.env, RunArtifactStore(run_dir),
                                   harness_policy=AdaptiveHarnessPolicy()).run(
                                       self.fixture.task, ReplanningPolicy())
            self.assertTrue(episode.resolved)
            self.assertEqual(episode.duplicate_actions, 0)
            records = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(records[1]["tool_result"]["error_code"], "PATCH_CONFLICT")
            self.assertIn(b"# human edit", self.file.read_bytes())


if __name__ == "__main__":
    unittest.main()
