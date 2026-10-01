from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from coding_agent_policy_opt.config import load_experiment_config
from coding_agent_policy_opt.evaluation.gates import evaluate_harness_smoke_gate
from coding_agent_policy_opt.data.manifest import read_jsonl, write_jsonl
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.swesmith.rollout import run_base_collection, run_swesmith_rollouts
from coding_agent_policy_opt.training.grpo_environment import CodingGRPOEnvironment
from coding_agent_policy_opt.swesmith.materializer import SweSmithRolloutMaterializer
from coding_agent_policy_opt.swesmith.stability import (
    stabilize_external_candidates,
    stabilize_swesmith_tasks,
)
from coding_agent_policy_opt.swesmith.config import (
    PINNED_SWESMITH_DATASET_REVISION,
    load_swesmith_environment_config,
)

from tests.helpers import PROJECT_ROOT
from tests.test_swesmith import upstream_task


class StableWorker:
    def __init__(self) -> None:
        self.materializations = 0
        self.closed: list[str] = []

    def invoke(self, request: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
        op = request["op"]
        if op == "materialize":
            self.materializations += 1
            container_id = f"stable-container-{self.materializations}"
            return {
                "protocol_version": "1.0.0",
                "ok": True,
                "data": {
                    "container_id": container_id,
                    "base_commit": "a" * 40,
                    "image_id": "sha256:" + "1" * 64,
                    "image_digests": ["example/image@sha256:" + "2" * 64],
                    "test_command": "pytest -q",
                    "diff_hash": "sha256:" + "0" * 64,
                },
            }
        if op == "evaluate":
            state = {
                "status": "failed",
                "fail_to_pass_count": 1,
                "pass_to_pass_failures": 0,
            }
            return {
                "protocol_version": "1.0.0",
                "ok": True,
                "tool_result": {
                    "ok": True,
                    "tool": "evaluator",
                    "data": {"test_state": state, "restored_test_files": ["tests/test_0.py"]},
                    "state_changed": False,
                },
                "diff_hash": "sha256:" + "0" * 64,
                "test_state": state,
            }
        if op == "close":
            self.closed.append(request["container_id"])
            return {
                "protocol_version": "1.0.0",
                "ok": True,
                "data": {"removed_container_id": request["container_id"]},
            }
        raise AssertionError(f"Unexpected worker operation: {op}")


class StabilityAndRolloutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_stability_")
        self.root = Path(self.temporary.name)
        task = upstream_task(0)
        inventory = task.inventory_entry("train", PINNED_SWESMITH_DATASET_REVISION)
        oracle = task.oracle_entry()
        self.inventory_path = self.root / "inventory.jsonl"
        self.oracle_path = self.root / "oracle.jsonl"
        write_jsonl(self.inventory_path, [inventory])
        write_jsonl(self.oracle_path, [oracle])
        self.environment_config = load_swesmith_environment_config(
            PROJECT_ROOT / "configs" / "environment" / "swesmith.yaml"
        )
        self.environment_config = replace(
            self.environment_config,
            counts={"smoke": 1, "train": 1, "dev": 1, "test": 1},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stability_gate_runs_two_evaluations_and_freezes_manifest(self) -> None:
        worker = StableWorker()
        manifest = self.root / "manifest.jsonl"
        report_path = self.root / "stability.json"

        report = stabilize_swesmith_tasks(
            self.inventory_path,
            self.oracle_path,
            manifest,
            report_path,
            self.environment_config,
            split="train",
            worker=worker,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["initialized_tasks"], 1)
        self.assertEqual(report["consistent_tasks"], 1)
        self.assertEqual(report["bug_reproduced_tasks"], 1)
        entry = read_jsonl(manifest)[0]
        self.assertEqual(entry["stable_repeats"], 2)
        self.assertEqual(entry["docker_image_digest"], "example/image@sha256:" + "2" * 64)
        self.assertEqual(entry["environment_profile"], self.environment_config.profile)
        self.assertEqual(worker.closed, ["stable-container-1"])

        server_config = replace(
            self.environment_config,
            profile="swesmith_linux_server_v1",
            platform="linux_server",
            distro="",
            runtime_root_env="CAPO_RUNTIME_ROOT",
        )
        server_manifest = self.root / "server_manifest.jsonl"
        stabilize_swesmith_tasks(
            self.inventory_path,
            self.oracle_path,
            server_manifest,
            self.root / "server_stability.json",
            server_config,
            split="train",
            worker=StableWorker(),
        )
        self.assertEqual(
            read_jsonl(server_manifest)[0]["environment_profile"],
            "swesmith_linux_server_v1",
        )

    def test_external_stability_emits_freeze_candidates(self) -> None:
        task = upstream_task(0)
        external_inventory = self.root / "external_inventory.jsonl"
        external_oracle = self.root / "external_oracle.jsonl"
        write_jsonl(
            external_inventory,
            [task.inventory_entry("external", PINNED_SWESMITH_DATASET_REVISION)],
        )
        write_jsonl(external_oracle, [task.oracle_entry()])
        candidates = self.root / "external_candidates.jsonl"
        report_path = self.root / "external_stability.json"

        report = stabilize_external_candidates(
            external_inventory,
            external_oracle,
            candidates,
            report_path,
            self.environment_config,
            worker=StableWorker(),
            minimum_eligible=1,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["eligible_tasks"], 1)
        self.assertTrue(report["ready_for_external_freeze"])
        candidate = read_jsonl(candidates)[0]
        self.assertEqual(candidate["environment_profile"], self.environment_config.profile)
        self.assertEqual(candidate["stable_repeats"], 2)

    def test_base_collection_runs_exactly_two_seeded_isolated_rollouts(self) -> None:
        manifest = self.root / "manifest.jsonl"
        stabilize_swesmith_tasks(
            self.inventory_path,
            self.oracle_path,
            manifest,
            self.root / "stability.json",
            self.environment_config,
            split="train",
            worker=StableWorker(),
        )
        experiment = load_experiment_config(
            PROJECT_ROOT / "configs" / "experiment" / "base_collection.yaml"
        )
        experiment = replace(
            experiment,
            run=replace(experiment.run, artifacts_dir=self.root / "artifacts"),
        )
        worker = StableWorker()
        decisions = [
            {
                "intent": "No model is loaded in this orchestration test.",
                "action": {
                    "type": "finish",
                    "args": {"status": "give_up", "summary": "orchestration-only"},
                },
            }
        ]

        report = run_base_collection(
            self.inventory_path,
            self.oracle_path,
            manifest,
            self.root / "base_report.json",
            experiment,
            self.environment_config,
            lambda seed: ScriptedPolicy(decisions),
            worker=worker,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["seeds"], [42, 43])
        self.assertEqual(report["expected_rollouts"], 2)
        self.assertEqual(report["completed_rollouts"], 2)
        self.assertEqual(report["resolved_rollouts"], 0)
        self.assertEqual(worker.materializations, 2)
        self.assertEqual(len(set(worker.closed)), 2)
        for result in report["results"]:
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "run.json").is_file())
            self.assertTrue((run_dir / "episodes.jsonl").is_file())

    def test_rl_materializer_uses_frozen_manifest_and_cleans_container(self) -> None:
        manifest = self.root / "manifest.jsonl"
        stabilize_swesmith_tasks(
            self.inventory_path,
            self.oracle_path,
            manifest,
            self.root / "stability.json",
            self.environment_config,
            split="train",
            worker=StableWorker(),
        )
        worker = StableWorker()
        materializer = SweSmithRolloutMaterializer(
            self.inventory_path,
            self.oracle_path,
            manifest,
            self.environment_config,
            worker=worker,
        )
        environment = CodingGRPOEnvironment(materializer)
        task_id = upstream_task(0).instance_id

        issue = environment.reset(task_id)
        environment.finish("give_up", "No policy is used in this contract test.")
        reward = environment.get_reward()

        self.assertIn("Fix task", issue)
        self.assertLess(reward, 0.0)
        self.assertEqual(worker.materializations, 1)
        self.assertEqual(worker.closed, ["stable-container-1"])

    def test_harness_smoke_gate_validates_run_schemas_and_cleanup(self) -> None:
        task = upstream_task(0)
        inventory_path = self.root / "smoke_inventory.jsonl"
        oracle_path = self.root / "smoke_oracle.jsonl"
        write_jsonl(
            inventory_path,
            [task.inventory_entry("smoke", PINNED_SWESMITH_DATASET_REVISION)],
        )
        write_jsonl(oracle_path, [task.oracle_entry()])
        manifest = self.root / "smoke_manifest.jsonl"
        stability_path = self.root / "smoke_stability.json"
        stabilize_swesmith_tasks(
            inventory_path,
            oracle_path,
            manifest,
            stability_path,
            self.environment_config,
            split="smoke",
            worker=StableWorker(),
        )
        experiment = load_experiment_config(
            PROJECT_ROOT / "configs" / "experiment" / "smoke.yaml"
        )
        experiment = replace(
            experiment,
            run=replace(experiment.run, artifacts_dir=self.root / "artifacts"),
        )
        rollout_path = self.root / "smoke_rollout.json"
        run_swesmith_rollouts(
            inventory_path,
            oracle_path,
            manifest,
            rollout_path,
            experiment,
            self.environment_config,
            lambda seed: ScriptedPolicy(
                [
                    {
                        "intent": "Contract-only smoke run.",
                        "action": {
                            "type": "finish",
                            "args": {"status": "give_up", "summary": "contract-only"},
                        },
                    }
                ]
            ),
            split="smoke",
            seeds=(42,),
            run_label="harness_gate_test",
            worker=StableWorker(),
        )

        report = evaluate_harness_smoke_gate(
            stability_path, rollout_path, expected_tasks=1
        )

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["schema_valid_runs"], 1)


if __name__ == "__main__":
    unittest.main()
