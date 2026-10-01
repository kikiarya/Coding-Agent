from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from coding_agent_policy_opt.config import load_experiment_config
from coding_agent_policy_opt.cli import build_parser
from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.harness.smoke import smoke_decisions
from coding_agent_policy_opt.evaluation.two_layer import (
    classify_arm,
    compare_two_layer_arms,
    validate_two_layer_configs,
)
from tests.helpers import SmokeRepository


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiment"


class TwoLayerConfigTest(unittest.TestCase):
    def test_compare_cli_parses_external_scope_guards(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "two-layer",
                "compare",
                "--base-run-dir",
                "base",
                "--model-only-run-dir",
                "model",
                "--harness-only-run-dir",
                "harness",
                "--joint-run-dir",
                "joint",
                "--expected-task-count",
                "50",
                "--expected-split",
                "external",
            ]
        )
        self.assertEqual(args.expected_task_count, 50)
        self.assertEqual(args.expected_split, "external")

    def test_grpo_train_cli_exposes_smoke_and_formal_profiles(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "rl",
                "train",
                "--inventory",
                "inventory.jsonl",
                "--oracle",
                "oracle.jsonl",
                "--manifest",
                "manifest.jsonl",
                "--checkpoint",
                "checkpoint",
                "--sft-gate-report",
                "sft.json",
                "--template-preflight-report",
                "template.json",
                "--output-dir",
                "run",
                "--profile",
                "formal",
                "--rl-gate-report",
                "rl.json",
            ]
        )
        self.assertEqual(args.profile, "formal")
        self.assertEqual(args.rl_gate_report, Path("rl.json"))

    def test_four_arm_matrix_is_frozen_and_classifies_correctly(self) -> None:
        paths = {
            "base": CONFIG_DIR / "two_layer_base.yaml",
            "model_only": CONFIG_DIR / "two_layer_model_only.yaml",
            "harness_only": CONFIG_DIR / "two_layer_harness_only.yaml",
            "joint": CONFIG_DIR / "two_layer_joint.yaml",
        }
        report = validate_two_layer_configs(paths)
        self.assertTrue(report["passed"])
        self.assertEqual(
            {name: classify_arm(load_experiment_config(path)) for name, path in paths.items()},
            {"base": "base", "model_only": "model_only", "harness_only": "harness_only", "joint": "joint"},
        )

    def test_two_layer_configs_reject_unfrozen_harness_budget(self) -> None:
        paths = {
            "base": CONFIG_DIR / "two_layer_base.yaml",
            "model_only": CONFIG_DIR / "two_layer_model_only.yaml",
            "harness_only": CONFIG_DIR / "two_layer_harness_only.yaml",
            "joint": CONFIG_DIR / "two_layer_joint.yaml",
        }
        original = paths["joint"].read_text(encoding="utf-8")
        try:
            paths["joint"].write_text(
                original.replace("max_tool_steps: 30", "max_tool_steps: 31"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unfrozen control variables"):
                validate_two_layer_configs(paths)
        finally:
            paths["joint"].write_text(original, encoding="utf-8")

    def test_four_arm_comparison_checks_runtime_metadata_and_pairs_tasks(self) -> None:
        configs = {
            "base": CONFIG_DIR / "two_layer_base.yaml",
            "model_only": CONFIG_DIR / "two_layer_model_only.yaml",
            "harness_only": CONFIG_DIR / "two_layer_harness_only.yaml",
            "joint": CONFIG_DIR / "two_layer_joint.yaml",
        }
        temporary = tempfile.TemporaryDirectory(prefix="capo_two_layer_runs_")
        fixtures: list[SmokeRepository] = []
        try:
            arm_runs = {}
            for arm, config_path in configs.items():
                fixture = SmokeRepository()
                fixtures.append(fixture)
                run_dir = Path(temporary.name) / arm
                AgentHarness(
                    load_experiment_config(config_path),
                    fixture.environment,
                    RunArtifactStore(run_dir),
                ).run(fixture.task, ScriptedPolicy(smoke_decisions()))
                arm_runs[arm] = [run_dir]

            report = compare_two_layer_arms(arm_runs, bootstrap_samples=20)
            self.assertEqual(report["arms"], ["base", "model_only", "harness_only", "joint"])
            self.assertEqual(report["comparisons"]["base_vs_joint"]["task_count"], 1)
            self.assertEqual(report["runtime_controls"]["joint"]["harness_policy"]["variant"], "adaptive_v1")
            self.assertEqual(report["evaluation_scope"]["task_count_by_arm"]["joint"], 1)
            self.assertFalse(report["resume_evidence"]["eligible"])
            scoped = compare_two_layer_arms(
                arm_runs,
                bootstrap_samples=20,
                expected_task_count=1,
                expected_split="smoke",
            )
            self.assertEqual(scoped["evaluation_scope"]["expected_split"], "smoke")
            with self.assertRaisesRegex(ValueError, "wrong task count"):
                compare_two_layer_arms(
                    arm_runs,
                    bootstrap_samples=20,
                    expected_task_count=2,
                )
        finally:
            for fixture in fixtures:
                fixture.close()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
