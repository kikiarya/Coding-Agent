from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from typing import Any

from coding_agent_policy_opt.schema import SchemaRegistry
from coding_agent_policy_opt.swesmith.config import (
    PINNED_SWESMITH_DATASET_REVISION,
    PINNED_SWESMITH_REPOSITORY_REVISION,
    UpstreamSweSmithTask,
    fetch_swesmith_source,
    load_swesmith_environment_config,
    select_repository_disjoint_tasks,
)
from coding_agent_policy_opt.swesmith.environment import WslSweSmithEnvironment
from coding_agent_policy_opt.swesmith.worker import _parse_search_output, _patch_paths, _safe_relpath
from coding_agent_policy_opt.models import ActionType, AgentAction, ErrorCode

from tests.helpers import PROJECT_ROOT


def upstream_task(repo_index: int, task_index: int = 0) -> UpstreamSweSmithTask:
    repo_name = f"owner{repo_index}__repo{repo_index}"
    return UpstreamSweSmithTask.from_dict(
        {
            "instance_id": f"{repo_name}.abcdef{repo_index}.method__{task_index}",
            "patch": (
                f"diff --git a/src/file_{task_index}.py b/src/file_{task_index}.py\n"
                f"--- a/src/file_{task_index}.py\n"
                f"+++ b/src/file_{task_index}.py\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
            "FAIL_TO_PASS": [f"tests/test_{task_index}.py::test_bug"],
            "PASS_TO_PASS": [f"tests/test_{task_index}.py::test_regression"],
            "image_name": f"example/swesmith.x86_64.{repo_name}",
            "repo": f"swesmith/{repo_name}.abcdef{repo_index}",
            "problem_statement": f"Fix task {repo_index}-{task_index}.",
        }
    )


class SweSmithConfigTest(unittest.TestCase):
    def test_environment_config_pins_repository_and_dataset(self) -> None:
        config = load_swesmith_environment_config(
            PROJECT_ROOT / "configs" / "environment" / "swesmith.yaml"
        )

        self.assertEqual(config.repository_revision, PINNED_SWESMITH_REPOSITORY_REVISION)
        self.assertEqual(config.dataset_revision, PINNED_SWESMITH_DATASET_REVISION)
        self.assertEqual(config.counts, {"smoke": 20, "train": 100, "dev": 20, "test": 40})
        self.assertEqual(config.docker_data_root, "D:\\DockerDesktopDataV2")
        self.assertEqual(
            config.dependencies_root, "D:\\CodingAgentPolicyOptimizationDependencies"
        )
        self.assertEqual(config.minimum_free_gib, 200)

    def test_frozen_dataset_fetch_validates_and_writes_only_registered_fields(self) -> None:
        config = load_swesmith_environment_config(
            PROJECT_ROOT / "configs" / "environment" / "swesmith.yaml"
        )
        raw = upstream_task(9).to_upstream_dict() | {"unregistered": "drop-me"}
        calls: list[dict[str, Any]] = []

        def loader(dataset_id: str, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append({"dataset_id": dataset_id, **kwargs})
            return [raw]

        with tempfile.TemporaryDirectory(prefix="capo_fetch_") as temporary:
            output = Path(temporary) / "source.jsonl"
            report = fetch_swesmith_source(output, config, dataset_loader=loader)
            row = __import__("json").loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["rows"], 1)
        self.assertEqual(calls[0]["revision"], PINNED_SWESMITH_DATASET_REVISION)
        self.assertNotIn("unregistered", row)
        self.assertEqual(row["instance_id"], raw["instance_id"])


class SweSmithTaskTest(unittest.TestCase):
    def test_inventory_is_public_but_oracle_retains_evaluator_data(self) -> None:
        task = upstream_task(1)

        inventory = task.inventory_entry("train", PINNED_SWESMITH_DATASET_REVISION)
        oracle = task.oracle_entry()
        task_spec = task.to_task_spec("train", "a" * 40)

        self.assertNotIn("patch", inventory)
        self.assertNotIn("fail_to_pass", inventory)
        self.assertNotIn("pass_to_pass", inventory)
        self.assertNotIn(task.patch, task_spec.issue)
        self.assertEqual(oracle["gold_files"], ["src/file_0.py"])
        self.assertEqual(oracle["patch"], task.patch)
        self.assertEqual(task.to_upstream_dict()["FAIL_TO_PASS"], list(task.fail_to_pass))
        schemas = SchemaRegistry()
        schemas.validate("swe_smith_inventory", inventory)
        schemas.validate("swe_smith_oracle", oracle)

    def test_missing_problem_statement_is_rejected(self) -> None:
        value = {
            "instance_id": "owner__repo.abc.task",
            "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
            "FAIL_TO_PASS": ["test_bug"],
            "PASS_TO_PASS": [],
            "image_name": "image",
            "repo": "swesmith/owner__repo.abc",
            "problem_statement": "",
        }

        with self.assertRaisesRegex(ValueError, "problem_statement"):
            UpstreamSweSmithTask.from_dict(value)

    def test_repository_disjoint_selection_is_exact_and_deterministic(self) -> None:
        tasks = [
            upstream_task(repo_index, task_index)
            for repo_index in range(8)
            for task_index in range(2)
        ]
        counts = {"smoke": 2, "train": 4, "dev": 2, "test": 2}

        first = select_repository_disjoint_tasks(
            tasks, counts, hash_seed=42, max_tasks_per_repository=2
        )
        second = select_repository_disjoint_tasks(
            reversed(tasks), counts, hash_seed=42, max_tasks_per_repository=2
        )

        self.assertEqual(
            {split: [task.instance_id for task in values] for split, values in first.items()},
            {split: [task.instance_id for task in values] for split, values in second.items()},
        )
        seen_repositories: set[str] = set()
        for split, expected_count in counts.items():
            self.assertEqual(len(first[split]), expected_count)
            repositories = {task.source_repo_key for task in first[split]}
            self.assertTrue(seen_repositories.isdisjoint(repositories))
            seen_repositories.update(repositories)


class FakeWorker:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.diff_hash = "sha256:" + "0" * 64

    def invoke(self, request: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
        self.requests.append({"request": request, "timeout_seconds": timeout_seconds})
        op = request["op"]
        if op == "materialize":
            return {
                "protocol_version": "1.0.0",
                "ok": True,
                "data": {
                    "container_id": "container-created-by-test",
                    "base_commit": "a" * 40,
                    "image_id": "sha256:" + "1" * 64,
                    "image_digests": ["example/image@sha256:" + "2" * 64],
                    "test_command": "pytest -q",
                    "diff_hash": self.diff_hash,
                },
            }
        if op == "execute":
            action = request["action"]
            if action["type"] == "apply_patch":
                self.diff_hash = "sha256:" + "3" * 64
                result = {
                    "ok": True,
                    "tool": "apply_patch",
                    "data": {"changed_files": ["src/file_0.py"], "diff_hash": self.diff_hash},
                    "state_changed": True,
                }
                state = None
            elif action["type"] == "run_tests":
                state = {
                    "status": "failed",
                    "fail_to_pass_count": 1,
                    "pass_to_pass_failures": 0,
                }
                result = {
                    "ok": True,
                    "tool": "run_tests",
                    "data": {"test_state": state},
                    "state_changed": False,
                }
            else:
                result = {
                    "ok": False,
                    "tool": action["type"],
                    "error_code": "INVALID_ARGUMENT",
                    "message": "fake rejection",
                    "retryable": True,
                    "state_changed": False,
                    "data": {},
                }
                state = None
            return {
                "protocol_version": "1.0.0",
                "ok": True,
                "tool_result": result,
                "diff_hash": self.diff_hash,
                "test_state": state,
            }
        if op == "evaluate":
            state = {
                "status": "passed",
                "fail_to_pass_count": 0,
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
                "diff_hash": self.diff_hash,
                "test_state": state,
            }
        if op == "close":
            return {
                "protocol_version": "1.0.0",
                "ok": True,
                "data": {"removed_container_id": request["container_id"]},
            }
        raise AssertionError(op)


class SweSmithEnvironmentBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_swesmith_environment_config(
            PROJECT_ROOT / "configs" / "environment" / "swesmith.yaml"
        )
        self.worker = FakeWorker()
        self.environment = WslSweSmithEnvironment(
            upstream_task(0), self.config, worker=self.worker, tool_timeout_seconds=123
        )

    def tearDown(self) -> None:
        self.environment.close()

    def test_materialization_patch_test_evaluator_and_cleanup_are_stateful(self) -> None:
        self.assertEqual(self.environment.base_commit, "a" * 40)
        self.assertEqual(self.environment.materialization_metadata()["container_id"], "container-created-by-test")

        patch = self.environment.execute(
            AgentAction(ActionType.APPLY_PATCH, {"patch": upstream_task(0).patch})
        )
        test = self.environment.execute(AgentAction(ActionType.RUN_TESTS, {}))
        evaluation = self.environment.evaluate()
        self.environment.close()

        self.assertTrue(patch.ok)
        self.assertTrue(test.ok)
        self.assertEqual(test.data["test_state"]["fail_to_pass_count"], 1)
        self.assertTrue(evaluation.ok)
        self.assertEqual(evaluation.data["test_state"]["fail_to_pass_count"], 0)
        self.assertEqual(self.worker.requests[-1]["request"]["op"], "close")
        execute_request = next(
            item for item in self.worker.requests if item["request"]["op"] == "execute"
        )
        self.assertEqual(execute_request["request"]["timeout_seconds"], 123)
        self.assertEqual(execute_request["request"]["expected_image_id"], "sha256:" + "1" * 64)

    def test_worker_tool_error_maps_to_registered_error_code(self) -> None:
        result = self.environment.execute(
            AgentAction(ActionType.READ_FILE, {"path": "missing", "start_line": 1, "end_line": 1})
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.INVALID_ARGUMENT)


class SweSmithWorkerSafetyTest(unittest.TestCase):
    def test_paths_and_patch_headers_reject_escape(self) -> None:
        with self.assertRaises(PermissionError):
            _safe_relpath("../outside")
        with self.assertRaises(PermissionError):
            _safe_relpath(".git/config")
        patch = "diff --git a/a b/a\n--- a/a\n+++ b/../outside\n"
        with self.assertRaises(PermissionError):
            _patch_paths(patch)
        symlink_patch = (
            "diff --git a/link b/link\nnew file mode 120000\n"
            "--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+../outside\n"
        )
        with self.assertRaisesRegex(ValueError, "Symlink"):
            _patch_paths(symlink_patch)

    def test_search_parser_normalizes_container_workdir(self) -> None:
        output = "/testbed/src/a.py:7:needle:with:colon\n"
        self.assertEqual(
            _parse_search_output(output, "/testbed", 10),
            [{"path": "src/a.py", "line": 7, "text": "needle:with:colon"}],
        )


if __name__ == "__main__":
    unittest.main()
