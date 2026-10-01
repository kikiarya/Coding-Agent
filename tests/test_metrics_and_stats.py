from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.data.manifest import write_jsonl
from coding_agent_policy_opt.evaluation.metrics import TaskMetrics, aggregate_metrics, collect_task_metrics, load_gold_files
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.harness.smoke import smoke_decisions
from coding_agent_policy_opt.evaluation.stats import compare_models, exact_mcnemar_test, paired_bootstrap_difference

from tests.helpers import SmokeRepository


def task_metric(
    task_id: str,
    *,
    resolved: bool,
    duplicate_actions: int = 0,
    environment_valid: bool = True,
) -> TaskMetrics:
    return TaskMetrics(
        task_id=task_id,
        repo_id=f"repo-{task_id}",
        model_id="model",
        adapter_id=None,
        experiment_config_hash="sha256:" + "0" * 64,
        environment_valid=environment_valid,
        resolved=resolved,
        localization_eligible=True,
        localization_hit_at_5=resolved,
        patch_groups=1,
        verified_patch_groups=int(resolved),
        model_finish_success=True,
        premature_success_finish=not resolved,
        recovery_opportunities=0,
        recovered_episodes=0,
        recovery_eligible_episode=False,
        recovered_episode=False,
        mean_steps_to_recovery=None,
        duplicate_actions=duplicate_actions,
        invalid_actions=0,
        total_tool_actions=5,
        total_steps=5,
        input_tokens=100,
        output_tokens=20,
        test_calls=1,
        wall_time_ms=1000,
        give_up=False,
        budget_exhausted=False,
    )


class MetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SmokeRepository()
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_metrics_")
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        harness = AgentHarness(
            self.fixture.config,
            self.fixture.environment,
            RunArtifactStore(self.run_dir),
        )
        harness.run(self.fixture.task, ScriptedPolicy(smoke_decisions()))

    def tearDown(self) -> None:
        self.fixture.close()
        self.temporary.cleanup()

    def test_smoke_artifacts_produce_expected_metrics_and_offline_localization(self) -> None:
        gold_path = self.root / "gold.jsonl"
        write_jsonl(
            gold_path,
            [
                {
                    "schema_version": "2.0.0",
                    "task_id": self.fixture.task.task_id,
                    "gold_files": ["calculator.py"],
                }
            ],
        )
        gold = load_gold_files(gold_path)

        tasks = collect_task_metrics([self.run_dir], gold_by_task=gold)
        summary = aggregate_metrics(tasks)

        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0].localization_hit_at_5)
        self.assertEqual(summary["environment_validity_rate"], 1.0)
        self.assertEqual(summary["resolve_rate"], 1.0)
        self.assertEqual(summary["localization_at_5"], 1.0)
        self.assertEqual(summary["post_patch_verification_rate"], 1.0)
        self.assertEqual(summary["premature_stop_rate"], 0.0)
        self.assertGreaterEqual(summary["costs"]["all_valid"]["wall_time_ms"]["mean"], 0)

    def test_collect_rejects_duplicate_task_runs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation task"):
            collect_task_metrics([self.run_dir, self.run_dir])


class StatisticsTest(unittest.TestCase):
    def test_exact_mcnemar_known_values(self) -> None:
        self.assertEqual(exact_mcnemar_test(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_test(1, 7), 0.0703125)
        self.assertEqual(exact_mcnemar_test(7, 1), 0.0703125)

    def test_paired_bootstrap_is_deterministic_and_candidate_minus_baseline(self) -> None:
        baseline = [task_metric("a", resolved=False), task_metric("b", resolved=False)]
        candidate = [task_metric("a", resolved=True), task_metric("b", resolved=False)]

        first = paired_bootstrap_difference(
            baseline, candidate, "resolve_rate", samples=500, seed=42
        )
        second = paired_bootstrap_difference(
            baseline, candidate, "resolve_rate", samples=500, seed=42
        )

        self.assertEqual(first, second)
        self.assertEqual(first["difference_candidate_minus_baseline"], 0.5)
        self.assertEqual(first["paired_bootstrap_samples_valid"], 500)

    def test_comparison_requires_identical_task_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical task_id sets"):
            compare_models(
                [task_metric("a", resolved=False)],
                [task_metric("b", resolved=True)],
                samples=10,
            )

    def test_behavior_bootstrap_uses_only_jointly_valid_task_pairs(self) -> None:
        baseline = [
            task_metric("a", resolved=False),
            task_metric("b", resolved=False, environment_valid=False),
        ]
        candidate = [
            task_metric("a", resolved=True),
            task_metric("b", resolved=True),
        ]

        result = paired_bootstrap_difference(
            baseline, candidate, "resolve_rate", samples=50, seed=42
        )

        self.assertEqual(result["paired_task_count"], 1)
        self.assertEqual(result["difference_candidate_minus_baseline"], 1.0)


if __name__ == "__main__":
    unittest.main()
