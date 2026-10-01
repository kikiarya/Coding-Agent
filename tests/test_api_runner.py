from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.agent.api_policy import APICompletion, APIModelPolicy, load_api_model_policy_config
from coding_agent_policy_opt.harness.runner import build_api_experiment_config, run_api_repository
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import PROJECT_ROOT, SmokeRepository


class _FakeAPIClient:
    def __init__(self) -> None:
        self.decisions = iter(smoke_decisions())

    def complete(self, messages, *, model_id, max_tokens, temperature, top_p):
        del messages, model_id, max_tokens, temperature, top_p
        return APICompletion(json.dumps(next(self.decisions)), prompt_tokens=10, completion_tokens=10)


class APIRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = SmokeRepository()
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_api_runner_")
        self.api_config_path = PROJECT_ROOT / "configs" / "model" / "api_policy.yaml"
        config = load_api_model_policy_config(self.api_config_path)
        self.policy = APIModelPolicy(_FakeAPIClient(), config, "fake-api-model")

    def tearDown(self) -> None:
        self.repo.close()
        self.temporary.cleanup()

    def test_build_config_marks_product_track_and_preserves_policy_controls(self) -> None:
        run_dir = Path(self.temporary.name) / "run"
        config = build_api_experiment_config(
            run_dir=run_dir,
            model_id="fake-api-model",
            test_command=("{python}", "-m", "unittest"),
            default_test_args=("discover", "-s", "tests"),
        )

        self.assertEqual(config.raw["track"], "api_product")
        self.assertEqual(config.model.model_id, "fake-api-model")
        self.assertEqual(config.harness_policy.variant, "adaptive_v1")
        self.assertEqual(config.repository.test_command, ("{python}", "-m", "unittest"))

    def test_run_executes_in_repository_and_writes_auditable_artifacts(self) -> None:
        run_dir = Path(self.temporary.name) / "run"
        report = run_api_repository(
            repo=self.repo.root,
            issue="Fix the greeting implementation and run the tests.",
            run_dir=run_dir,
            api_config=self.api_config_path,
            test_command=("{python}", "-m", "unittest"),
            default_test_args=("discover", "-s", "tests"),
            policy_override=self.policy,
        )

        self.assertTrue(report["episode"]["resolved"])
        self.assertEqual(report["track"], "api_product")
        self.assertTrue((run_dir / "run.json").is_file())
        self.assertTrue((run_dir / "steps.jsonl").is_file())
        run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run_metadata["config"]["track"], "api_product")
        self.assertEqual(run_metadata["harness_policy"]["variant"], "adaptive_v1")

    def test_dirty_repository_is_rejected_before_artifacts_are_created(self) -> None:
        (self.repo.root / "untracked.txt").write_text("intentional", encoding="utf-8")
        run_dir = Path(self.temporary.name) / "run"

        with self.assertRaisesRegex(ValueError, "dirty"):
            run_api_repository(
                repo=self.repo.root,
                issue="Run the repository task.",
                run_dir=run_dir,
                api_config=self.api_config_path,
                policy_override=self.policy,
            )
        self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
