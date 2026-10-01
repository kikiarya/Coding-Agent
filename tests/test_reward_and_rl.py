from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.evaluation.rewards import RewardWeights, compute_trajectory_reward
from coding_agent_policy_opt.training.grpo_environment import CodingGRPOEnvironment
from coding_agent_policy_opt.evaluation.gates import evaluate_rl_gate, load_rl_config
from coding_agent_policy_opt.training.preflight import grpo_chat_template_preflight
from coding_agent_policy_opt.agent.model_policy import load_qwen_policy_config
from coding_agent_policy_opt.training.grpo import build_grpo_run_plan
from coding_agent_policy_opt.data.manifest import TaskManifestEntry, write_jsonl
from coding_agent_policy_opt.hashing import sha256_text
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import PROJECT_ROOT, SmokeRepository


class RewardTest(unittest.TestCase):
    def test_core_reward_uses_bounded_components(self) -> None:
        episode = {
            "resolved": True,
            "initial_fail_to_pass_count": 2,
            "best_fail_to_pass_count": 0,
            "patch_groups": 2,
            "verified_patch_groups": 2,
            "duplicate_actions": 1,
            "invalid_actions": 0,
            "total_steps": 10,
            "recovered_episodes": 1,
        }

        result = compute_trajectory_reward(episode, max_tool_steps=30)

        self.assertEqual(result.success, 1.0)
        self.assertEqual(result.progress, 1.0)
        self.assertEqual(result.verification, 1.0)
        self.assertEqual(result.duplicate, 0.1)
        self.assertEqual(result.invalid, 0.0)
        self.assertAlmostEqual(result.normalized_steps, 1 / 3)
        self.assertEqual(result.recovery, 0.0)
        self.assertAlmostEqual(result.total, 1.2283333333333333)

    def test_recovery_ablation_requires_audit_and_adds_once(self) -> None:
        episode = {
            "resolved": False,
            "initial_fail_to_pass_count": 2,
            "best_fail_to_pass_count": 1,
            "patch_groups": 1,
            "verified_patch_groups": 1,
            "duplicate_actions": 0,
            "invalid_actions": 0,
            "total_steps": 5,
            "recovered_episodes": 3,
        }
        with self.assertRaisesRegex(ValueError, "reward-hacking audit"):
            compute_trajectory_reward(
                episode, max_tool_steps=30, include_recovery_ablation=True
            )

        result = compute_trajectory_reward(
            episode,
            max_tool_steps=30,
            include_recovery_ablation=True,
            reward_hacking_audit_passed=True,
        )

        self.assertEqual(result.recovery, 1.0)

    def test_invalid_actions_reduce_reward_as_a_bounded_component(self) -> None:
        result = compute_trajectory_reward(
            {
                "resolved": False,
                "invalid_actions": 2,
                "total_steps": 4,
            },
            max_tool_steps=10,
        )

        self.assertEqual(result.invalid, 0.5)
        self.assertAlmostEqual(result.total, -0.045)

    def test_reward_weights_are_explicit_and_config_driven(self) -> None:
        weights = RewardWeights(success=2.0)
        result = compute_trajectory_reward(
            {"resolved": True, "total_steps": 1},
            max_tool_steps=10,
            weights=weights,
        )

        self.assertAlmostEqual(result.total, 1.995)
        self.assertEqual(weights.to_config()["success"], 2.0)


class SmokeMaterializer:
    def __init__(self, fixture: SmokeRepository) -> None:
        self.fixture = fixture
        self.cleanup_calls = 0

    def materialize(self, task_id: str, **kwargs):
        if task_id != self.fixture.task.task_id:
            raise KeyError(task_id)

        def cleanup() -> None:
            self.cleanup_calls += 1

        return self.fixture.task, self.fixture.environment, cleanup


class RLEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SmokeRepository()
        self.materializer = SmokeMaterializer(self.fixture)
        self.environment = CodingGRPOEnvironment(self.materializer)

    def tearDown(self) -> None:
        self.environment._close()
        self.fixture.close()

    def test_stateful_tool_episode_produces_reward_and_cleanup(self) -> None:
        issue = self.environment.reset(self.fixture.task.task_id)
        decisions = smoke_decisions()

        self.assertEqual(issue, self.fixture.task.issue)
        self.environment.search_code("def add", ".", "*.py", 20)
        self.environment.read_file("calculator.py", 1, 40)
        self.environment.apply_patch(decisions[2]["action"]["args"]["patch"])
        self.environment.run_tests()
        self.environment.inspect_diff()
        self.environment.finish("success", "Patched and verified.")
        reward = self.environment.get_reward()

        self.assertAlmostEqual(reward, 1.24)
        self.assertEqual(self.environment.reward_breakdown["success"], 1.0)
        self.assertEqual(self.environment.reward_breakdown["progress"], 1.0)
        self.assertEqual(self.environment.reward_breakdown["verification"], 1.0)
        self.assertEqual(len(self.environment.trajectory_records), 6)
        self.assertEqual(self.materializer.cleanup_calls, 1)

    def test_adaptive_harness_policy_gates_unverified_success_but_allows_give_up(self) -> None:
        from coding_agent_policy_opt.training.grpo_environment import GRPOEnvironmentConfig

        environment = CodingGRPOEnvironment(
            self.materializer,
            GRPOEnvironmentConfig(
                harness_policy_variant="adaptive_v1",
                harness_history_limit=8,
                harness_verification_mode="required",
                harness_recovery_mode="required",
            ),
        )
        environment.reset(self.fixture.task.task_id)
        environment.read_file("calculator.py", 1, 40)
        environment.apply_patch(smoke_decisions()[2]["action"]["args"]["patch"])
        blocked = environment.finish("success", "Not verified yet.")
        self.assertIn("VERIFICATION_REQUIRED", blocked)
        self.assertFalse(environment.finished)
        environment.finish("give_up", "Stop after the guard.")
        self.assertTrue(environment.finished)
        self.assertEqual(environment.harness_interventions, 1)
        environment.get_reward()

    def test_duplicate_and_invalid_calls_are_auditable(self) -> None:
        self.environment.reset(self.fixture.task.task_id)

        self.environment.search_code("def add", ".")
        self.environment.search_code("def add", ".")
        self.environment.finish("invalid", "Bad enum")  # type: ignore[arg-type]

        self.assertEqual(self.environment.duplicate_actions, 1)
        self.assertEqual(self.environment.invalid_actions, 1)
        self.assertFalse(self.environment.finished)
        self.assertIn(
            "INVALID_ACTION", self.environment.trajectory_records[-1]["automatic_labels"]
        )

    def test_reset_accepts_trl_prompt_without_forwarding_it(self) -> None:
        issue = self.environment.reset(
            task_id=self.fixture.task.task_id,
            prompt=[{"role": "user", "content": ""}],
        )
        self.environment.finish("give_up", "Contract-only reset test.")
        self.environment.get_reward()

        self.assertEqual(issue, self.fixture.task.issue)
        self.assertEqual(self.materializer.cleanup_calls, 1)

    def test_current_harness_policy_exposes_auditable_turn_surface(self) -> None:
        self.environment.reset(self.fixture.task.task_id)

        decision = self.environment._current_harness_policy()

        self.assertEqual(decision["variant"], "static_v1")
        self.assertIn("search_code", decision["allowed_tools"])
        public_tools = {
            name
            for name, member in inspect.getmembers(self.environment, predicate=inspect.ismethod)
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        }
        self.assertEqual(
            public_tools,
            {
                "search_code",
                "read_file",
                "apply_patch",
                "run_tests",
                "inspect_diff",
                "finish",
            },
        )
        self.environment.finish("give_up", "Contract-only policy snapshot test.")
        self.environment.get_reward()

    def test_reset_rejects_unregistered_dataset_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unregistered GRPO dataset fields"):
            self.environment.reset(
                task_id=self.fixture.task.task_id, unexpected="value"
            )

    def test_reward_emits_schema_valid_trajectory_audit(self) -> None:
        audits = []
        environment = CodingGRPOEnvironment(self.materializer, audit_sink=audits.append)
        environment.reset(self.fixture.task.task_id)
        environment.finish("give_up", "Audit the rollout.")

        environment.get_reward()

        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["task_id"], self.fixture.task.task_id)
        self.assertIn("total", audits[0]["reward_components"])
        self.assertEqual(audits[0]["total_steps"], 1)


class RLGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_rl_gate_")
        self.root = Path(self.temporary.name)
        self.config_path = PROJECT_ROOT / "configs" / "rl" / "grpo.yaml"
        self.sft_gate = self.root / "sft_gate.json"
        self.sft_gate.write_text(
            json.dumps({"report_type": "sft_gate", "passed": True}),
            encoding="utf-8",
        )
        self.template_preflight = self.root / "template_preflight.json"
        self.template_preflight.write_text(
            json.dumps(
                {
                    "report_type": "grpo_chat_template_preflight",
                    "passed": True,
                    "tools_passed": 6,
                    "tokenizer_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                }
            ),
            encoding="utf-8",
        )
        self.checkpoint = self.root / "policy_checkpoint"
        self.checkpoint.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _evidence(self, **overrides) -> Path:
        value = {
            "schema_version": "2.0.0",
            "sft_gate_report": str(self.sft_gate),
            "policy_checkpoint": str(self.checkpoint),
            "policy_checkpoint_frozen": True,
            "chat_template_preflight_report": str(self.template_preflight),
            "gpu_memory_gb": 80,
            "smoke_tasks_completed": 8,
            "smoke_oom_count": 0,
            "smoke_gradient_anomaly_count": 0,
            "smoke_state_isolation_passed": True,
            "audit_trajectories": 20,
            "reward_hacking_findings": 0,
            "reward_component_logs": True,
            "kl_loss_logs": True,
            "loss_logs": True,
            "invalid_action_logs": True,
            "episode_length_logs": True,
            "same_harness_and_budget": True,
        }
        value.update(overrides)
        path = self.root / f"evidence-{len(list(self.root.glob('evidence-*')))}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_config_remains_conditional_and_gate_requires_all_evidence(self) -> None:
        config = load_rl_config(self.config_path)

        passed = evaluate_rl_gate(self._evidence(), self.config_path)
        insufficient_gpu = evaluate_rl_gate(
            self._evidence(gpu_memory_gb=24), self.config_path
        )

        self.assertEqual(config["status"], "gated")
        self.assertEqual(config["trainer"]["max_tool_calling_iterations"], 12)
        self.assertEqual(config["profiles"]["formal"]["update_steps"], 500)
        self.assertTrue(passed["passed"])
        self.assertFalse(insufficient_gpu["passed"])
        self.assertFalse(insufficient_gpu["checks"]["gpu_memory"])

    def test_gate_rejects_failed_chat_template_preflight(self) -> None:
        self.template_preflight.write_text(
            json.dumps(
                {
                    "report_type": "grpo_chat_template_preflight",
                    "passed": False,
                    "tools_passed": 5,
                    "tokenizer_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                }
            ),
            encoding="utf-8",
        )

        report = evaluate_rl_gate(self._evidence(), self.config_path)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["chat_template_prefix_preserving"])


class PrefixToolTokenizer:
    def __init__(self, *, preserving: bool) -> None:
        self.preserving = preserving

    def apply_chat_template(self, conversation, *, tools, tokenize, add_generation_prompt):
        if not tokenize or add_generation_prompt:
            raise AssertionError("Unexpected tokenizer options")
        rendered = json.dumps(
            {"tools": tools, "conversation": conversation},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        if self.preserving:
            tool_count = sum(message.get("role") == "tool" for message in conversation)
            prefix = json.dumps(tools, sort_keys=True).encode("utf-8")
            body = b"".join(
                json.dumps(message, sort_keys=True).encode("utf-8")
                for message in conversation
            )
            return list(prefix + body + (b"T" * tool_count))
        return list(str(len(rendered)).encode("ascii") + rendered)


class GRPOTemplatePreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_qwen_policy_config(
            PROJECT_ROOT / "configs" / "model" / "qwen_policy.yaml"
        )

    def test_all_registered_tools_are_prefix_preserving(self) -> None:
        report = grpo_chat_template_preflight(
            PrefixToolTokenizer(preserving=True), self.config
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["tools_checked"], 6)
        self.assertEqual(report["tools_passed"], 6)

    def test_retokenizing_template_fails_preflight(self) -> None:
        report = grpo_chat_template_preflight(
            PrefixToolTokenizer(preserving=False), self.config
        )

        self.assertFalse(report["passed"])
        self.assertLess(report["tools_passed"], 6)


class GRPORunPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_grpo_plan_")
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.jsonl"
        entries = [
            TaskManifestEntry(
                schema_version="2.0.0",
                task_id=f"task-{index}",
                split="train",
                source="swe-smith",
                repo_id=f"owner/repo-{index}",
                base_commit="a" * 40,
                docker_image_digest="example@sha256:" + f"{index:064x}",
                issue_hash=sha256_text(f"issue-{index}"),
                parent_task_id=None,
                environment_profile="swesmith_wsl_v1",
                initial_test_status="failed",
                stable_repeats=2,
            ).to_dict()
            for index in range(64)
        ]
        write_jsonl(self.manifest, entries)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.sft_gate = self.root / "sft_gate.json"
        self.sft_gate.write_text(
            json.dumps({"report_type": "sft_gate", "passed": True}),
            encoding="utf-8",
        )
        self.template = self.root / "template.json"
        self.template.write_text(
            json.dumps(
                {
                    "report_type": "grpo_chat_template_preflight",
                    "passed": True,
                    "tools_passed": 6,
                    "tokenizer_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                }
            ),
            encoding="utf-8",
        )
        self.rl_gate = self.root / "rl_gate.json"
        self.rl_gate.write_text(
            json.dumps({"report_type": "rl_gate", "passed": True}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_smoke_and_formal_plans_freeze_exact_v2_scales(self) -> None:
        common = {
            "manifest_path": self.manifest,
            "checkpoint_path": self.checkpoint,
            "sft_gate_report_path": self.sft_gate,
            "template_preflight_report_path": self.template,
            "model_config_path": PROJECT_ROOT / "configs" / "model" / "qwen_policy.yaml",
            "rl_config_path": PROJECT_ROOT / "configs" / "rl" / "grpo.yaml",
        }

        smoke = build_grpo_run_plan(
            **common,
            output_dir=self.root / "smoke-output",
            profile="smoke",
        )
        formal = build_grpo_run_plan(
            **common,
            output_dir=self.root / "formal-output",
            profile="formal",
            rl_gate_report_path=self.rl_gate,
        )

        self.assertEqual(smoke["task_count"], 8)
        self.assertEqual(smoke["update_steps"], 50)
        self.assertEqual(formal["task_count"], 64)
        self.assertEqual(formal["update_steps"], 500)
        self.assertEqual(smoke["trainer"]["num_generations"], 4)
        self.assertEqual(smoke["trainer"]["max_tool_calling_iterations"], 12)
        self.assertEqual(len(set(smoke["selected_task_ids"])), 8)
        self.assertTrue(smoke["checkpoint_tree_hash"].startswith("sha256:"))

    def test_formal_plan_cannot_bypass_rl_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a passed RL Gate"):
            build_grpo_run_plan(
                self.manifest,
                self.checkpoint,
                self.sft_gate,
                self.template,
                PROJECT_ROOT / "configs" / "model" / "qwen_policy.yaml",
                PROJECT_ROOT / "configs" / "rl" / "grpo.yaml",
                self.root / "formal-output",
                profile="formal",
            )

    def test_formal_plan_requires_an_rl_gate_report_type(self) -> None:
        self.rl_gate.write_text(
            json.dumps({"report_type": "sft_gate", "passed": True}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Expected rl_gate report"):
            build_grpo_run_plan(
                self.manifest,
                self.checkpoint,
                self.sft_gate,
                self.template,
                PROJECT_ROOT / "configs" / "model" / "qwen_policy.yaml",
                PROJECT_ROOT / "configs" / "rl" / "grpo.yaml",
                self.root / "formal-output",
                profile="formal",
                rl_gate_report_path=self.rl_gate,
            )


if __name__ == "__main__":
    unittest.main()
