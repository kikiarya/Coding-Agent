from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace

from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.environment import RepositoryEnvironment
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.harness.smoke import smoke_decisions
from coding_agent_policy_opt.harness.tools import dispatch_action
from coding_agent_policy_opt.models import AgentAction, ErrorCode
from tests.helpers import SmokeRepository


class EvidenceRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.fixture = SmokeRepository()
        self.env = self.fixture.environment

    def tearDown(self):
        self.fixture.close()

    def action(self, name, **args):
        return dispatch_action(self.env, AgentAction.from_dict({"type": name, "args": args}))

    def read(self):
        return self.action("read_file", path="calculator.py", start_line=1, end_line=20)

    def fix(self):
        self.read()
        return self.action("apply_patch", patch=smoke_decisions()[2]["action"]["args"]["patch"])

    def finish(self):
        return self.action("finish", status="success", summary="Done")

    def test_read_authorization_is_not_durable_and_new_files_are(self):
        before = self.env.durable_state_payload()
        self.read()
        self.assertEqual(before, self.env.durable_state_payload())
        fresh = RepositoryEnvironment(self.fixture.root, self.fixture.profile)
        self.assertEqual(before, fresh.durable_state_payload())
        refused = dispatch_action(fresh, AgentAction.from_dict(smoke_decisions()[2]["action"]))
        self.assertEqual(refused.error_code, ErrorCode.FILE_NOT_READ)
        (self.fixture.root / "new.txt").write_text("human change\n")
        self.assertNotEqual(before, self.env.durable_state_payload())

    def test_stale_evidence_blocks_finish_after_external_edit(self):
        self.assertTrue(self.fix().ok)
        self.assertTrue(self.env.evaluate().ok)
        self.assertTrue(self.finish().ok)
        with (self.fixture.root / "calculator.py").open("a") as stream:
            stream.write("# human edit\n")
        self.assertEqual(self.finish().error_code, ErrorCode.VERIFICATION_REQUIRED)
        self.assertEqual(self.env.state_payload()["recovery"]["evidence"]["validity"], "stale")
        self.assertTrue(self.env.evaluate().ok)
        self.assertTrue(self.finish().ok)

    def test_environment_change_invalidates_evidence(self):
        self.fix()
        self.env.evaluate()
        with patch.dict("os.environ", {"CAPO_TEST_NEW_ENV": "changed"}):
            self.assertEqual(self.finish().error_code, ErrorCode.VERIFICATION_REQUIRED)

    def test_retesting_external_change_does_not_authorize_old_candidate(self):
        self.env.evaluate()
        old = self.env.recovery.roles["current"]
        self.fix()
        path = self.fixture.root / "calculator.py"
        path.write_text(path.read_text() + "# external addition\n")
        self.env.evaluate()
        self.assertNotIn(old, self.env.recovery.candidates)
        self.read()
        result = self.action("apply_patch", candidate_id=old, workspace_version=self.env.workspace_version())
        self.assertFalse(result.ok)
        self.assertIn("# external addition", path.read_text())

    def test_workspace_changes_during_test_are_unknown(self):
        original = subprocess.run
        def execute(command, *args, **kwargs):
            if command[0] != "git":
                (self.fixture.root / "changed.txt").write_text("changed during tests\n")
                return subprocess.CompletedProcess(command, 0, "passed", "")
            return original(command, *args, **kwargs)
        with patch("subprocess.run", side_effect=execute):
            result = self.env.evaluate()
        self.assertFalse(result.ok)
        self.assertEqual(result.data["evidence"]["validity"], "unknown")
        self.assertFalse(self.finish().ok)

    def test_regression_switch_requires_fresh_read_and_retest(self):
        self.env.evaluate()
        self.fix()
        self.env.evaluate()
        best = self.env.recovery.roles["best_observed"]
        self.read()
        bad = smoke_decisions()[2]["action"]["args"]["patch"].replace("-    return a - b", "-    return a + b").replace("+    return a + b", "+    return a * b")
        self.assertTrue(self.action("apply_patch", patch=bad).ok)
        self.env.evaluate()
        self.assertTrue(self.env.recovery.regression)
        args = {"candidate_id": best, "workspace_version": self.env.workspace_version()}
        self.assertEqual(self.action("apply_patch", **args).error_code, ErrorCode.FILE_NOT_READ)
        self.read()
        self.assertTrue(self.action("apply_patch", **args).ok)
        self.assertEqual(self.finish().error_code, ErrorCode.VERIFICATION_REQUIRED)
        self.env.evaluate()
        self.assertTrue(self.finish().ok)
        self.assertLessEqual(len(self.env.recovery.candidates), 3)

    def test_switch_refuses_external_edits_even_with_current_hash(self):
        self.env.evaluate()
        original = self.env.recovery.roles["current"]
        self.fix()
        path = self.fixture.root / "calculator.py"
        path.write_text(path.read_text() + "# preserved\n")
        self.read()
        result = self.action("apply_patch", candidate_id=original, workspace_version=self.env.workspace_version())
        self.assertEqual(result.error_code, ErrorCode.PATCH_CONFLICT)
        self.assertIn("# preserved", path.read_text())

    def test_permission_denied_before_candidate_write(self):
        result = dispatch_action(self.env, AgentAction.from_dict(smoke_decisions()[2]["action"]), granted_permissions={"repo.read"})
        self.assertEqual(result.error_code, ErrorCode.PERMISSION_DENIED)

    def test_candidate_pool_is_bounded_across_multiple_edits(self):
        from coding_agent_policy_opt.harness.recovery import RecoveryState
        state = RecoveryState()
        for index in range(10):
            state.capture(str(index), {"a.py": f"x = {index}\n"})
            self.assertLessEqual(len(state.candidates), 3)
        self.assertEqual(state.roles["current"], "9")
        self.assertEqual(state.roles["backup"], "8")

    def test_snapshot_roundtrip_does_not_restore_write_authorization(self):
        self.env.evaluate()
        self.read()
        stored = self.env.export_recovery_state()
        fresh = RepositoryEnvironment(self.fixture.root, self.fixture.profile)
        fresh.restore_recovery_state(stored)
        self.assertEqual(fresh.recovery.roles, self.env.recovery.roles)
        self.assertEqual(fresh._read_versions, {})

    def test_recovery_artifact_is_deduplicated_and_checksum_checked(self):
        from coding_agent_policy_opt.harness.checkpoint import read_recovery_state
        with tempfile.TemporaryDirectory() as temp:
            store = RunArtifactStore(Path(temp) / "run")
            payload = self.env.export_recovery_state()
            reference = store.write_recovery_state(payload)
            self.assertEqual(reference, store.write_recovery_state(payload))
            self.assertEqual(len(list((store.run_dir / "recovery").glob("*.json"))), 1)
            checkpoint = {"recovery_state_ref": reference}
            self.assertEqual(read_recovery_state(store.run_dir, checkpoint), payload)
            path = next((store.run_dir / "recovery").glob("*.json"))
            path.write_text('{"version": 123}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_recovery_state(store.run_dir, checkpoint)

    def test_record_profile_does_not_allow_switching(self):
        self.env.evaluate()
        candidate = self.env.recovery.roles["current"]
        self.fix()
        self.env.candidate_switching = False
        self.env.evidence_guidance = False
        self.assertTrue(self.env.recovery.candidates)
        self.assertEqual(self.env.state_payload()["recovery"]["candidates"], [])
        self.assertIsNone(self.env.state_payload()["recovery"]["roles"]["best_observed"])
        self.assertEqual(self.action("apply_patch", candidate_id=candidate,
                                    workspace_version=self.env.workspace_version()).error_code,
                         ErrorCode.INVALID_ARGUMENT)

    def test_resuming_multiple_attempts_keeps_step_budget_and_history(self):
        from coding_agent_policy_opt.web.service import RunManager
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            config = replace(self.fixture.config, harness=replace(self.fixture.config.harness, max_tool_steps=3))
            config.raw["harness"]["max_tool_steps"] = 3
            manager = RunManager.__new__(RunManager)
            run = SimpleNamespace(run_dir=root, repo_path=self.fixture.root)
            resume = None
            for attempt in range(3):
                env = RepositoryEnvironment(self.fixture.root, self.fixture.profile)
                events = []
                episode = AgentHarness(config, env, RunArtifactStore(root, allow_existing=attempt > 0)).run(
                    self.fixture.task, ScriptedPolicy([smoke_decisions()[1]]), resume_state=resume,
                    event_sink=lambda kind, payload: events.append(kind),
                    should_cancel=lambda: events.count("step_completed") >= 1)
                resume = manager._resume_state(run)
                self.assertEqual(resume["steps_used"], attempt + 1)
            env = RepositoryEnvironment(self.fixture.root, self.fixture.profile)
            episode = AgentHarness(config, env, RunArtifactStore(root, allow_existing=True)).run(
                self.fixture.task, ScriptedPolicy(smoke_decisions()), resume_state=resume)
            self.assertEqual(episode.total_steps, 0)
            self.assertEqual(episode.termination_reason, "STEP_BUDGET_EXHAUSTED")
            self.assertEqual(len((root / "steps.jsonl").read_text().splitlines()), 3)
            from coding_agent_policy_opt.harness.checkpoint import replay_latest_checkpoint
            from tests.test_checkpoint import FreshCopyMaterializer
            report = replay_latest_checkpoint(root, FreshCopyMaterializer(self.fixture, Path(temp)))
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["replayed_steps"], 3)

    def test_web_resume_after_read_uses_durable_checkpoint_and_keeps_budget(self):
        from coding_agent_policy_opt.web.service import RunManager
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = []
            AgentHarness(self.fixture.config, self.env, RunArtifactStore(root / "run")).run(
                self.fixture.task, ScriptedPolicy(smoke_decisions()[:2]),
                event_sink=lambda kind, payload: events.append(kind),
                should_cancel=lambda: events.count("step_completed") >= 2,
            )
            manager = RunManager.__new__(RunManager)
            state = manager._resume_state(SimpleNamespace(run_dir=root / "run", repo_path=self.fixture.root))
            self.assertEqual(state["steps_used"], 2)
            fresh = RepositoryEnvironment(self.fixture.root, self.fixture.profile)
            fresh.restore_recovery_state(state["recovery_state"])
            self.assertEqual(fresh._read_versions, {})
            self.assertEqual(dispatch_action(fresh, AgentAction.from_dict(smoke_decisions()[2]["action"])).error_code,
                             ErrorCode.FILE_NOT_READ)
            (self.fixture.root / "new.txt").write_text("outside checkpoint\n")
            with self.assertRaisesRegex(ValueError, "workspace state"):
                manager._resume_state(SimpleNamespace(run_dir=root / "run", repo_path=self.fixture.root))


if __name__ == "__main__":
    unittest.main()
