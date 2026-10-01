from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.harness.policy import (
    AdaptiveHarnessPolicy,
    StaticHarnessPolicy,
    enumerate_harness_candidates,
    select_harness_candidate,
)
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import SmokeRepository


class HarnessPolicyTest(unittest.TestCase):
    def test_harness_candidate_selection_is_dev_only_and_deterministic(self) -> None:
        candidates = enumerate_harness_candidates()
        self.assertEqual(len(candidates), 13)
        reports = [
            {
                "harness_policy": candidates[0],
                "metrics": {
                    "environment_validity_rate": 1.0,
                    "resolve_rate": 0.50,
                    "episode_recovery_rate": 0.30,
                    "premature_stop_rate": 0.10,
                    "invalid_action_rate": 0.05,
                    "duplicate_action_rate": 0.05,
                    "costs": {"all_valid": {"tool_calls": {"mean": 8.0}}},
                },
            },
            {
                "harness_policy": candidates[1],
                "metrics": {
                    "environment_validity_rate": 1.0,
                    "resolve_rate": 0.60,
                    "episode_recovery_rate": 0.35,
                    "premature_stop_rate": 0.08,
                    "invalid_action_rate": 0.04,
                    "duplicate_action_rate": 0.04,
                    "costs": {"all_valid": {"tool_calls": {"mean": 7.0}}},
                },
            },
        ]
        selection = select_harness_candidate(reports)
        self.assertEqual(selection["selection_split"], "dev")
        self.assertEqual(selection["selected"]["harness_policy"], candidates[1])

    def test_candidate_cannot_trade_environment_validity_for_resolve_rate(self) -> None:
        candidates = enumerate_harness_candidates()
        reports = [
            {
                "harness_policy": candidates[0],
                "metrics": {
                    "environment_validity_rate": 1.0,
                    "resolve_rate": 0.50,
                    "episode_recovery_rate": 0.30,
                    "premature_stop_rate": 0.10,
                    "invalid_action_rate": 0.05,
                    "duplicate_action_rate": 0.05,
                    "costs": {"all_valid": {"tool_calls": {"mean": 8.0}}},
                },
            },
            {
                "harness_policy": candidates[1],
                "metrics": {
                    "environment_validity_rate": 0.90,
                    "resolve_rate": 0.99,
                    "episode_recovery_rate": 0.90,
                    "premature_stop_rate": 0.01,
                    "invalid_action_rate": 0.01,
                    "duplicate_action_rate": 0.01,
                    "costs": {"all_valid": {"tool_calls": {"mean": 2.0}}},
                },
            },
        ]
        selection = select_harness_candidate(reports)
        self.assertEqual(selection["selected"]["harness_policy"], candidates[0])
        self.assertEqual(selection["excluded"][0]["reason"], "environment_validity_below_static_baseline")

    def test_adaptive_policy_changes_surface_by_state(self) -> None:
        policy = AdaptiveHarnessPolicy(history_limit=8)

        initial = policy.decide({"policy_state": {}})
        self.assertIn("search_code", initial.allowed_tools)
        self.assertIn("finish", initial.allowed_tools)

        verification = policy.decide(
            {"policy_state": {"patch_pending_verification": True}}
        )
        self.assertIn("run_tests", verification.allowed_tools)
        self.assertIn("finish", verification.allowed_tools)
        self.assertTrue(verification.verification_required)

        recovery = policy.decide({"policy_state": {"recovery_active": True}})
        self.assertIn("apply_patch", recovery.allowed_tools)
        self.assertIn("finish", recovery.allowed_tools)
        self.assertTrue(recovery.recovery_required)

        complete = policy.decide(
            {
                "policy_state": {
                    "last_action_type": "run_tests",
                    "last_test_passed": True,
                }
            }
        )
        self.assertEqual(complete.context_strategy, "completion_focus")
        self.assertIn("finish", complete.allowed_tools)

    def test_static_policy_is_a_non_intervening_control(self) -> None:
        decision = StaticHarnessPolicy(history_limit=8).decide({"anything": True})
        self.assertEqual(
            decision.allowed_tools,
            (
                "search_code",
                "read_file",
                "apply_patch",
                "run_tests",
                "inspect_diff",
                "finish",
            ),
        )
        self.assertFalse(decision.verification_required)
        self.assertFalse(decision.recovery_required)

    def test_adaptive_harness_blocks_premature_finish_and_logs_policy(self) -> None:
        fixture = SmokeRepository()
        artifact_temp = tempfile.TemporaryDirectory(prefix="capo_harness_policy_")
        try:
            run_dir = Path(artifact_temp.name) / "run"
            episode = AgentHarness(
                fixture.config,
                fixture.environment,
                RunArtifactStore(run_dir),
                harness_policy=AdaptiveHarnessPolicy(),
            ).run(
                fixture.task,
                _ScriptedPolicy([smoke_decisions()[1], smoke_decisions()[2], smoke_decisions()[-1]]),
            )

            steps = [
                json.loads(line)
                for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(steps[0]["harness_policy_variant"], "adaptive_v1")
            self.assertIn("HARNESS_VERIFICATION_GUARD", steps[2]["automatic_labels"])
            self.assertEqual(
                steps[2]["tool_result"]["error_code"], "VERIFICATION_REQUIRED"
            )
            self.assertGreaterEqual(episode.invalid_actions, 1)
            self.assertEqual(episode.termination_reason, "MODEL_GIVE_UP")
        finally:
            fixture.close()
            artifact_temp.cleanup()


class _ScriptedPolicy:
    def __init__(self, decisions: list[dict]) -> None:
        self.decisions = decisions
        self.index = 0

    def decide(self, context: dict):
        del context
        from coding_agent_policy_opt.models import PolicyOutput

        if self.index >= len(self.decisions):
            value = {
                "intent": "Give up after policy guard test.",
                "action": {
                    "type": "finish",
                    "args": {"status": "give_up", "summary": "guard test"},
                },
            }
        else:
            value = self.decisions[self.index]
            self.index += 1
        return PolicyOutput(payload=value)


if __name__ == "__main__":
    unittest.main()
