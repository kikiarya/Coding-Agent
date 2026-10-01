from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from coding_agent_policy_opt.web.app import create_app
    from coding_agent_policy_opt.web.service import RunManager
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

from coding_agent_policy_opt.harness.smoke import create_smoke_repository
from coding_agent_policy_opt.harness.smoke import smoke_decisions
from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.harness.environment import RepositoryEnvironment, RepositoryProfile
from coding_agent_policy_opt.config import load_experiment_config


@unittest.skipUnless(WEB_AVAILABLE, "install the web extra to run Web API tests")
class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="capo_web_test_")
        root = Path(self.temp.name)
        self.repo = root / "repo"
        create_smoke_repository(self.repo)

        def fake_runner(**kwargs):
            bus = kwargs["event_sink"]
            bus("step_completed", {"step_index": 0, "action": {"type": "finish"}})
            return {
                "run_dir": str(kwargs["run_dir"]),
                "episode": {"resolved": True, "termination_reason": "MODEL_FINISH_SUCCESS"},
            }

        manager = RunManager(
            allowed_roots=(root,),
            artifact_root=root / "artifacts",
            api_config=root / "api.yaml",
            runner=fake_runner,
        )
        manager.api_config.write_text("profile: test\n", encoding="utf-8")
        self.client = TestClient(create_app(manager))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_start_run_and_read_status_and_events(self) -> None:
        response = self.client.post(
            "/api/runs",
            json={
                "repo_path": str(self.repo),
                "issue": "Fix the calculator",
                "task_id": "web_smoke",
                "harness_policy": "adaptive_v1",
            },
        )
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        for _ in range(30):
            status = self.client.get(f"/api/runs/{run_id}").json()
            if status["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        self.assertEqual(status["status"], "completed")
        events = self.client.get(f"/api/runs/{run_id}/events").text
        self.assertIn("run_queued", events)
        self.assertIn("step_completed", events)
        self.assertIn("web_run_finished", events)

    def test_canonical_run_trace_and_idempotency(self) -> None:
        payload = {
            "source_repo": str(self.repo),
            "issue": "Fix the calculator",
            "task_id": "canonical_smoke",
            "idempotency_key": "canonical-key-1",
        }
        first = self.client.post("/runs", json=payload)
        self.assertEqual(first.status_code, 202)
        run_id = first.json()["run_id"]
        second = self.client.post("/runs", json=payload)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(second.json()["run_id"], run_id)
        for _ in range(30):
            status = self.client.get(f"/runs/{run_id}").json()
            if status["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        trace = self.client.get(f"/runs/{run_id}/trace")
        self.assertEqual(trace.status_code, 200)
        self.assertEqual(trace.json()["run"]["run_id"], run_id)
        self.assertGreaterEqual(len(trace.json()["events"]), 3)
        self.assertTrue((Path(status["run_dir"]) / "workspace.json").is_file())

    def test_sqlite_state_and_trace_survive_manager_reconstruction(self) -> None:
        response = self.client.post(
            "/runs",
            json={"source_repo": str(self.repo), "issue": "Fix it", "task_id": "restart_smoke"},
        )
        run_id = response.json()["run_id"]
        for _ in range(30):
            status = self.client.get(f"/runs/{run_id}").json()
            if status["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        rebuilt = RunManager(
            allowed_roots=(Path(self.temp.name),),
            artifact_root=Path(self.temp.name) / "artifacts",
            api_config=Path(self.temp.name) / "api.yaml",
            runner=lambda **kwargs: {"episode": {"resolved": True}},
        )
        restored = rebuilt.get(run_id)
        self.assertEqual(restored.status, status["status"])
        self.assertGreaterEqual(len(rebuilt.trace(run_id)["events"]), 3)

    def test_dirty_worktree_is_rejected(self) -> None:
        (self.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        response = self.client.post(
            "/api/runs",
            json={"repo_path": str(self.repo), "issue": "Fix it"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("dirty", response.json()["detail"])

    def test_path_outside_allowlist_is_rejected(self) -> None:
        outside = Path(self.temp.name).parent
        response = self.client.post(
            "/api/runs",
            json={"repo_path": str(outside), "issue": "Fix it"},
        )
        self.assertEqual(response.status_code, 400)

    def test_api_key_is_not_an_accepted_web_input(self) -> None:
        response = self.client.post(
            "/api/runs",
            json={
                "repo_path": str(self.repo),
                "issue": "Fix it",
                "api_key": "must-not-be-accepted",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("must-not-be-accepted", response.text)

    def test_unknown_run_is_not_exposed(self) -> None:
        response = self.client.get("/api/runs/not-a-real-run")
        self.assertEqual(response.status_code, 404)

    def test_real_harness_emits_product_events(self) -> None:
        config = load_experiment_config(
            Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke.yaml"
        )
        profile = RepositoryProfile(
            test_command=config.repository.test_command,
            default_test_args=config.repository.default_test_args,
            timeout_seconds=config.harness.tool_timeout_seconds,
        )
        environment = RepositoryEnvironment(self.repo, profile)
        run_dir = Path(self.temp.name) / "harness-run"
        events: list[str] = []
        episode = AgentHarness(
            config,
            environment,
            RunArtifactStore(run_dir),
        ).run(
            self._task_for_repo(environment),
            ScriptedPolicy(smoke_decisions()),
            event_sink=lambda event_type, payload: events.append(event_type),
        )
        self.assertTrue(episode.resolved)
        self.assertIn("run_started", events)
        self.assertIn("initial_evaluation", events)
        self.assertIn("step_completed", events)
        self.assertIn("evaluator_completed", events)
        self.assertIn("run_completed", events)

    def _task_for_repo(self, environment: RepositoryEnvironment):
        from coding_agent_policy_opt.models import TaskSpec

        return TaskSpec(
            task_id="web_harness_smoke",
            split="smoke",
            repo_id="local/smoke-calculator",
            issue="Fix the calculator and verify the tests.",
            base_commit=environment.base_commit,
        )


if __name__ == "__main__":
    unittest.main()
