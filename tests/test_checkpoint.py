from __future__ import annotations

import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.checkpoint import replay_latest_checkpoint
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import SmokeRepository


class FreshCopyMaterializer:
    def __init__(self, fixture: SmokeRepository, root: Path) -> None:
        self.fixture = fixture
        self.root = root

    def materialize(self, task_id: str, **kwargs):
        if kwargs:
            raise ValueError(kwargs)
        if task_id != self.fixture.task.task_id:
            raise KeyError(task_id)
        copy_root = self.root / "fresh-replay"
        shutil.copytree(self.fixture.root, copy_root)
        from coding_agent_policy_opt.harness.environment import RepositoryEnvironment

        environment = RepositoryEnvironment(copy_root, self.fixture.profile)

        def cleanup() -> None:
            def make_writable(function, path, _exc_info) -> None:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                function(path)

            shutil.rmtree(copy_root, onerror=make_writable)

        return self.fixture.task, environment, cleanup


class CheckpointRecoveryTest(unittest.TestCase):
    def test_latest_checkpoint_replays_in_a_fresh_repository(self) -> None:
        fixture = SmokeRepository()
        temporary = tempfile.TemporaryDirectory(prefix="capo_checkpoint_")
        try:
            run_dir = Path(temporary.name) / "run"
            policy = ScriptedPolicy(smoke_decisions()[:2])
            AgentHarness(
                fixture.config,
                fixture.environment,
                RunArtifactStore(run_dir),
            ).run(fixture.task, policy)

            report = replay_latest_checkpoint(
                run_dir,
                FreshCopyMaterializer(fixture, Path(temporary.name)),
            )
            self.assertTrue(report["passed"])
            self.assertTrue(report["state_hash_matches"])
            self.assertTrue(report["diff_hash_matches"])
            self.assertEqual(report["replayed_steps"], 3)
            self.assertEqual(report["skipped_steps"], 0)
        finally:
            fixture.close()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
