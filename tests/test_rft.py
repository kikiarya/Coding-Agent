from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
import json
from pathlib import Path

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.data.manifest import write_jsonl
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.training.rft import build_rft_dataset, select_rft_rollouts
from coding_agent_policy_opt.harness.smoke import smoke_decisions

from tests.helpers import SmokeRepository


HASH = "sha256:" + "0" * 64


def _episode(
    task_id: str,
    *,
    resolved: bool = True,
    environment_valid: bool = True,
    duplicate_actions: int = 0,
    invalid_actions: int = 0,
    split: str = "train",
    total_steps: int = 0,
) -> dict:
    return {
        "schema_version": "2.0.0",
        "run_id": f"run-{task_id}-{duplicate_actions}-{invalid_actions}",
        "episode_id": f"episode-{task_id}-{duplicate_actions}-{invalid_actions}",
        "task_id": task_id,
        "split": split,
        "repo_id": "repo",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "adapter_id": None,
        "experiment_config_hash": HASH,
        "seed": 42,
        "termination_reason": "MODEL_FINISH_SUCCESS" if resolved else "MODEL_GIVE_UP",
        "resolved": resolved,
        "environment_valid": environment_valid,
        "total_steps": total_steps,
        "total_input_tokens": 10,
        "total_output_tokens": 5,
        "test_calls": 0,
        "patch_groups": 0,
        "verified_patch_groups": 0,
        "duplicate_actions": duplicate_actions,
        "invalid_actions": invalid_actions,
        "recovery_opportunities": 0,
        "recovered_episodes": 0,
        "final_diff_hash": HASH,
        "evaluator_artifact": "evaluator.json",
        "initial_evaluator_artifact": "initial.json",
        "initial_fail_to_pass_count": 0,
        "best_fail_to_pass_count": 0,
        "wall_time_ms": 1,
    }


def _steps(episode: dict, count: int) -> list[dict]:
    return [
        {
            "schema_version": "2.0.0",
            "run_id": episode["run_id"],
            "episode_id": episode["episode_id"],
            "task_id": episode["task_id"],
            "split": episode["split"],
            "repo_id": episode["repo_id"],
            "base_commit": "base",
            "step_index": index,
            "model_id": episode["model_id"],
            "model_revision": "revision",
            "adapter_id": episode["adapter_id"],
            "prompt_version": "agent_prompt_v1",
            "harness_version": "harness_v2_two_layer",
            "experiment_config_hash": HASH,
            "state_hash_before": HASH,
            "intent": "test",
            "action": {"type": "read_file", "args": {}},
            "normalized_action_hash": HASH,
            "tool_result": {"ok": True, "error_code": None, "observation_hash": HASH},
            "state_hash_after": HASH,
            "diff_hash": HASH,
            "test_state": None,
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "automatic_labels": [],
        }
        for index in range(count)
    ]


class RFTSelectionTest(unittest.TestCase):
    def test_build_dataset_reuses_verified_decision_samples(self) -> None:
        fixture = SmokeRepository()
        temporary = tempfile.TemporaryDirectory(prefix="capo_rft_build_")
        try:
            run_dir = Path(temporary.name) / "run"
            task = replace(fixture.task, split="train")
            AgentHarness(
                fixture.config,
                fixture.environment,
                RunArtifactStore(run_dir),
            ).run(task, ScriptedPolicy(smoke_decisions()))

            report = build_rft_dataset(
                [run_dir], Path(temporary.name) / "D_success.jsonl", count=1
            )

            self.assertEqual(report["selected_task_count"], 1)
            self.assertGreater(report["sample_count"], 0)
            self.assertTrue((Path(temporary.name) / "D_success.jsonl").is_file())
        finally:
            fixture.close()
            temporary.cleanup()

    def test_selects_one_evaluator_verified_rollout_per_task(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="capo_rft_")
        try:
            root = Path(temporary.name)
            best = root / "best"
            slower = root / "slower"
            failed = root / "failed"
            for path, episode in (
                (best, _episode("task-a", total_steps=10)),
                (slower, _episode("task-a", duplicate_actions=1, total_steps=10)),
                (failed, _episode("task-b", resolved=False)),
            ):
                path.mkdir()
                write_jsonl(path / "episodes.jsonl", [episode])
                write_jsonl(path / "steps.jsonl", _steps(episode, episode["total_steps"]))
                (path / "evaluator.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )
                (path / "initial.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )

            report = select_rft_rollouts([slower, failed, best])

            self.assertEqual(report["candidate_count"], 3)
            self.assertEqual(report["eligible_rollout_count"], 2)
            self.assertEqual(report["selected_task_count"], 1)
            self.assertEqual(report["selected"][0]["run_dir"], str(best.resolve()))
            reasons = {entry["reason"] for entry in report["excluded"]}
            self.assertEqual(reasons, {"not_resolved", "non_best_rollout_for_task"})
        finally:
            temporary.cleanup()

    def test_rft_rejects_invalid_split_and_invalid_actions(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="capo_rft_invalid_")
        try:
            root = Path(temporary.name)
            paths = []
            for name, episode in (
                ("test", _episode("task-test", split="test")),
                ("invalid", _episode("task-invalid", invalid_actions=1)),
            ):
                path = root / name
                path.mkdir()
                write_jsonl(path / "episodes.jsonl", [episode])
                write_jsonl(path / "steps.jsonl", _steps(episode, episode["total_steps"]))
                (path / "evaluator.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )
                (path / "initial.json").write_text(
                    json.dumps({"ok": True}), encoding="utf-8"
                )
                paths.append(path)

            report = select_rft_rollouts(paths)

            self.assertEqual(report["selected_task_count"], 0)
            self.assertEqual(
                {entry["reason"] for entry in report["excluded"]},
                {"split_not_train", "invalid_action"},
            )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
