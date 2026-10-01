from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.models import PolicyOutput
from coding_agent_policy_opt.schema import SchemaRegistry
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import SmokeRepository


class AgentHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SmokeRepository()
        self.artifact_temp = tempfile.TemporaryDirectory(prefix="capo_artifacts_")

    def tearDown(self) -> None:
        self.fixture.close()
        self.artifact_temp.cleanup()

    def _run(self, decisions: list[dict]) -> tuple[Path, object]:
        run_dir = Path(self.artifact_temp.name) / "run"
        store = RunArtifactStore(run_dir)
        harness = AgentHarness(
            self.fixture.config,
            self.fixture.environment,
            store,
        )
        episode = harness.run(self.fixture.task, ScriptedPolicy(decisions))
        return run_dir, episode

    def test_successful_smoke_episode_is_resolved_and_schema_valid(self) -> None:
        run_dir, episode = self._run(smoke_decisions())

        self.assertTrue(episode.resolved)
        self.assertTrue(episode.environment_valid)
        self.assertEqual(episode.total_steps, 6)
        self.assertEqual(episode.test_calls, 1)
        self.assertEqual(episode.patch_groups, 1)
        self.assertEqual(episode.verified_patch_groups, 1)
        self.assertEqual(episode.invalid_actions, 0)

        schemas = SchemaRegistry()
        steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
        episodes = [
            json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(steps), 6)
        self.assertEqual(len(episodes), 1)
        for step in steps:
            schemas.validate("step_record", step)
        schemas.validate("episode_record", episodes[0])
        self.assertIn("PATCH_APPLIED", steps[2]["automatic_labels"])
        self.assertIn("POST_PATCH_VERIFIED", steps[3]["automatic_labels"])
        self.assertTrue((run_dir / episode.evaluator_artifact).is_file())
        self.assertTrue((run_dir / episode.initial_evaluator_artifact).is_file())
        self.assertEqual(episode.initial_fail_to_pass_count, 1)
        self.assertEqual(episode.best_fail_to_pass_count, 0)
        self.assertGreaterEqual(len(list((run_dir / "observations").glob("*.json"))), 6)

    def test_unverified_finish_is_rejected_even_if_evaluator_passes(self) -> None:
        decisions = [smoke_decisions()[1], smoke_decisions()[2], smoke_decisions()[-1]]
        run_dir, episode = self._run(decisions)

        self.assertTrue(episode.resolved)
        self.assertEqual(episode.patch_groups, 1)
        self.assertEqual(episode.verified_patch_groups, 0)
        steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
        attempted_finish = steps[2]
        self.assertEqual(attempted_finish["tool_result"]["error_code"], "VERIFICATION_REQUIRED")
        self.assertNotEqual(episode.termination_reason, "MODEL_FINISH_SUCCESS")

    def test_duplicate_action_is_counted_without_state_change(self) -> None:
        repeated = smoke_decisions()[0]
        decisions = [
            repeated,
            repeated,
            {
                "intent": "Stop after duplicate-action test.",
                "action": {
                    "type": "finish",
                    "args": {"status": "give_up", "summary": "Test complete."},
                },
            },
        ]
        run_dir, episode = self._run(decisions)

        self.assertEqual(episode.duplicate_actions, 1)
        steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertIn("DUPLICATE_ACTION", steps[1]["automatic_labels"])

    def test_invalid_action_is_logged_and_policy_can_recover(self) -> None:
        decisions = [
            {
                "intent": "Use an unavailable tool.",
                "action": {"type": "bash", "args": {"command": "echo unsafe"}},
            },
            {
                "intent": "Stop after schema validation test.",
                "action": {
                    "type": "finish",
                    "args": {"status": "give_up", "summary": "Test complete."},
                },
            },
        ]
        run_dir, episode = self._run(decisions)

        self.assertEqual(episode.invalid_actions, 1)
        steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertIn("INVALID_ACTION", steps[0]["automatic_labels"])
        self.assertEqual(steps[0]["tool_result"]["error_code"], "SCHEMA_VALIDATION_FAILED")

    def test_next_decision_receives_full_observation_not_only_hash(self) -> None:
        decisions = smoke_decisions()

        class ObservationPolicy:
            def __init__(self) -> None:
                self.calls = 0

            def decide(self, context: dict) -> PolicyOutput:
                if self.calls == 0:
                    payload = decisions[0]
                else:
                    matches = context["history"][0]["observation"]["data"]["matches"]
                    self.assert_matches(matches)
                    payload = {
                        "intent": "Observation was available; stop the test.",
                        "action": {
                            "type": "finish",
                            "args": {"status": "give_up", "summary": "Context test complete."},
                        },
                    }
                self.calls += 1
                return PolicyOutput(payload=payload)

            @staticmethod
            def assert_matches(matches: list[dict]) -> None:
                if not any(match["path"] == "calculator.py" for match in matches):
                    raise AssertionError("search observation was not passed to the next decision")

        run_dir = Path(self.artifact_temp.name) / "context-run"
        episode = AgentHarness(
            self.fixture.config,
            self.fixture.environment,
            RunArtifactStore(run_dir),
        ).run(self.fixture.task, ObservationPolicy())

        self.assertEqual(episode.total_steps, 2)

    def test_decision_output_token_limit_is_enforced_before_tool_execution(self) -> None:
        class OversizedPolicy:
            def decide(self, context: dict) -> PolicyOutput:
                del context
                return PolicyOutput(payload=smoke_decisions()[2], output_tokens=1025)

        run_dir = Path(self.artifact_temp.name) / "oversized-run"
        episode = AgentHarness(
            self.fixture.config,
            self.fixture.environment,
            RunArtifactStore(run_dir),
        ).run(self.fixture.task, OversizedPolicy())

        self.assertEqual(episode.termination_reason, "TOKEN_BUDGET_EXHAUSTED")
        self.assertEqual(episode.patch_groups, 0)
        steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertIn("BUDGET_EXHAUSTED", steps[0]["automatic_labels"])
        self.assertEqual(steps[0]["tool_result"]["error_code"], "BUDGET_EXHAUSTED")


if __name__ == "__main__":
    unittest.main()
