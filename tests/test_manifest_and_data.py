from __future__ import annotations

import tempfile
import unittest
import subprocess
from pathlib import Path

from coding_agent_policy_opt.harness.artifacts import RunArtifactStore
from coding_agent_policy_opt.data.gate import validate_decision_dataset
from coding_agent_policy_opt.harness.runner import AgentHarness
from coding_agent_policy_opt.hashing import sha256_text
from coding_agent_policy_opt.data.manifest import (
    EXTERNAL_SUBSET_COUNT,
    TaskManifestEntry,
    deterministic_external_subset,
    freeze_external_manifest,
    merge_task_manifests,
    read_jsonl,
    validate_manifest,
    write_jsonl,
)
from coding_agent_policy_opt.agent.policy import ScriptedPolicy
from coding_agent_policy_opt.harness.replay import replay_corrected_samples
from coding_agent_policy_opt.data.release import (
    build_dataset_release_manifest,
    validate_dataset_release_manifest,
)
from coding_agent_policy_opt.harness.environment import RepositoryEnvironment
from coding_agent_policy_opt.schema import SchemaRegistry
from coding_agent_policy_opt.harness.smoke import smoke_decisions
from coding_agent_policy_opt.trajectory.samples import build_success_samples, write_success_samples
from coding_agent_policy_opt.swesmith.config import (
    PINNED_SWESMITH_DATASET_REVISION,
    load_selected_swesmith_tasks,
)
from coding_agent_policy_opt.swesmith.external import (
    build_external_inventory,
    build_external_source_inventory,
)

from tests.helpers import SmokeRepository


def manifest_entry(
    task_id: str,
    split: str,
    repo_id: str,
    *,
    parent_task_id: str | None = None,
) -> TaskManifestEntry:
    return TaskManifestEntry(
        schema_version="2.0.0",
        task_id=task_id,
        split=split,
        source="local" if split == "smoke" else "swe-smith",
        repo_id=repo_id,
        base_commit="0123456789abcdef",
        docker_image_digest=None,
        issue_hash=sha256_text(f"issue:{task_id}"),
        parent_task_id=parent_task_id,
        environment_profile="unit-test",
        initial_test_status="failed",
        stable_repeats=2,
    )


class TaskManifestTest(unittest.TestCase):
    def test_external_subset_count_is_frozen(self) -> None:
        self.assertEqual(EXTERNAL_SUBSET_COUNT, 50)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_manifest_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, name: str, entries: list[TaskManifestEntry]) -> Path:
        path = self.root / name
        write_jsonl(path, (entry.to_dict() for entry in entries))
        return path

    def test_valid_manifest_has_repository_isolation(self) -> None:
        path = self._write(
            "valid.jsonl",
            [
                manifest_entry("task-train", "train", "repo-a"),
                manifest_entry("task-dev", "dev", "repo-b"),
                manifest_entry("task-test", "test", "repo-c"),
            ],
        )

        report = validate_manifest(path)

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.split_counts, {"train": 1, "dev": 1, "test": 1})

    def test_manifest_rejects_repository_and_lineage_leakage(self) -> None:
        path = self._write(
            "leakage.jsonl",
            [
                manifest_entry("parent", "train", "shared-repo"),
                manifest_entry(
                    "derived", "test", "shared-repo", parent_task_id="parent"
                ),
            ],
        )

        report = validate_manifest(path)

        self.assertFalse(report.valid)
        self.assertTrue(any("repository leakage" in error for error in report.errors))
        self.assertTrue(any("derived-task leakage" in error for error in report.errors))

    def test_external_subset_is_deterministic(self) -> None:
        entries = [manifest_entry(f"external-{index}", "external", f"repo-{index}") for index in range(8)]

        first = deterministic_external_subset(entries, 5)
        second = deterministic_external_subset(reversed(entries), 5)

        self.assertEqual([entry.task_id for entry in first], [entry.task_id for entry in second])

    def test_merge_builds_exact_deterministic_mvp_manifest(self) -> None:
        inputs = []
        for split, count in (("train", 2), ("dev", 1), ("test", 1)):
            path = self.root / f"{split}.jsonl"
            write_jsonl(
                path,
                [
                    manifest_entry(f"{split}-{index}", split, f"{split}-repo-{index}").to_dict()
                    for index in reversed(range(count))
                ],
            )
            inputs.append(path)
        output = self.root / "mvp.jsonl"

        report = merge_task_manifests(
            inputs,
            output,
            expected_counts={"train": 2, "dev": 1, "test": 1},
        )

        self.assertTrue(report["valid"])
        self.assertEqual(report["entries"], 4)
        values = read_jsonl(output)
        self.assertEqual(
            [(value["split"], value["task_id"]) for value in values],
            [("train", "train-0"), ("train", "train-1"), ("dev", "dev-0"), ("test", "test-0")],
        )

    def test_merge_rejects_cross_split_repository_leakage_before_write(self) -> None:
        train = self.root / "train.jsonl"
        dev = self.root / "dev.jsonl"
        output = self.root / "mvp.jsonl"
        write_jsonl(train, [manifest_entry("train-0", "train", "shared").to_dict()])
        write_jsonl(dev, [manifest_entry("dev-0", "dev", "shared").to_dict()])

        with self.assertRaisesRegex(ValueError, "repository leakage"):
            merge_task_manifests(
                [train, dev],
                output,
                expected_counts={"train": 1, "dev": 1},
            )

        self.assertFalse(output.exists())

    def test_external_freeze_records_exclusions_before_hash_selection(self) -> None:
        candidates = self.root / "external_candidates.jsonl"
        manifest = self.root / "external_5.jsonl"
        exclusions = self.root / "external_exclusions.json"
        values = []
        for index in range(8):
            initialized = index != 3
            values.append(
                {
                    "schema_version": "2.0.0",
                    "instance_id": f"verified-{index}",
                    "repo_id": f"owner/external-{index}",
                    "base_commit": "b" * 40,
                    "docker_image_digest": "external@sha256:" + f"{index:064x}",
                    "issue_hash": sha256_text(f"external-issue-{index}"),
                    "environment_profile": "swebench_verified_v1",
                    "initialized": initialized,
                    "stable_repeats": 2 if initialized else 0,
                    "initial_test_status": "failed" if initialized else None,
                    "exclusion_reason": None if initialized else "image unavailable",
                }
            )
        write_jsonl(candidates, values)

        report = freeze_external_manifest(
            candidates, manifest, exclusions, count=5
        )
        frozen = read_jsonl(manifest)

        self.assertTrue(report["passed"])
        self.assertEqual(report["excluded_count"], 1)
        self.assertEqual(report["exclusions"][0]["instance_id"], "verified-3")
        self.assertEqual(len(frozen), 5)
        self.assertTrue(all(value["split"] == "external" for value in frozen))
        self.assertEqual(
            [value["task_id"] for value in frozen], report["selected_instance_ids"]
        )

    def test_external_freeze_builds_executable_inventory_and_oracle(self) -> None:
        candidates = self.root / "external_candidates.jsonl"
        source = self.root / "external_source.jsonl"
        manifest = self.root / "external_1.jsonl"
        exclusions = self.root / "external_exclusions.json"
        inventory = self.root / "external_inventory.jsonl"
        oracle = self.root / "external_oracle.jsonl"
        problem = "Fix the calculator value."
        task_id = "verified-0"
        patch = (
            "diff --git a/calculator.py b/calculator.py\n"
            "--- a/calculator.py\n"
            "+++ b/calculator.py\n"
            "@@ -1 +1 @@\n"
            "-return 1\n"
            "+return 2\n"
        )
        write_jsonl(
            source,
            [
                {
                    "instance_id": task_id,
                    "patch": patch,
                    "FAIL_TO_PASS": ["tests/test_calculator.py::test_value"],
                    "PASS_TO_PASS": [],
                    "image_name": "verified-image:latest",
                    "repo": "owner/external-0",
                    "problem_statement": problem,
                }
            ],
        )
        write_jsonl(
            candidates,
            [
                {
                    "schema_version": "2.0.0",
                    "instance_id": task_id,
                    "repo_id": "owner/external-0",
                    "base_commit": "b" * 40,
                    "docker_image_digest": "external@sha256:" + "0" * 64,
                    "issue_hash": sha256_text(problem),
                    "environment_profile": "swebench_verified_v1",
                    "initialized": True,
                    "stable_repeats": 2,
                    "initial_test_status": "failed",
                    "exclusion_reason": None,
                }
            ],
        )
        provisional_inventory = self.root / "external_source_inventory.jsonl"
        provisional_oracle = self.root / "external_source_oracle.jsonl"
        provisional = build_external_source_inventory(
            source,
            provisional_inventory,
            provisional_oracle,
            dataset_revision=PINNED_SWESMITH_DATASET_REVISION,
        )
        freeze_report = freeze_external_manifest(
            candidates, manifest, exclusions, count=1
        )

        report = build_external_inventory(
            source,
            candidates,
            exclusions,
            manifest,
            inventory,
            oracle,
            dataset_revision=PINNED_SWESMITH_DATASET_REVISION,
            expected_count=1,
        )
        selected = load_selected_swesmith_tasks(inventory, oracle, split="external")

        self.assertTrue(freeze_report["passed"])
        self.assertFalse(provisional["final_scoring_ready"])
        self.assertEqual(provisional["count"], 1)
        self.assertTrue(report["passed"])
        self.assertEqual(report["count"], 1)
        self.assertEqual(report["dataset_id"], "SWE-bench Verified")
        self.assertEqual(selected[0].task.instance_id, task_id)
        self.assertEqual(selected[0].task.patch, patch)


class DecisionDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SmokeRepository()
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_data_")
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        harness = AgentHarness(
            self.fixture.config,
            self.fixture.environment,
            RunArtifactStore(self.run_dir),
        )
        self.episode = harness.run(self.fixture.task, ScriptedPolicy(smoke_decisions()))

    def tearDown(self) -> None:
        self.fixture.close()
        self.temporary.cleanup()

    def test_resolved_run_builds_traceable_success_samples(self) -> None:
        samples = build_success_samples(self.run_dir)

        self.assertEqual(len(samples), 6)
        self.assertTrue(all(sample.source_run_id == self.run_dir.name for sample in samples))
        self.assertTrue(all(sample.task_id == self.fixture.task.task_id for sample in samples))
        self.assertTrue(all(sample.repo_id == self.fixture.task.repo_id for sample in samples))
        self.assertEqual(samples[3].category, "verification")
        self.assertEqual(samples[-1].category, "termination_efficiency")
        registry = SchemaRegistry()
        for sample in samples:
            registry.validate("decision_sample", sample.to_dict())
            registry.validate("agent_action", sample.target)

    def test_data_gate_checks_count_manifest_and_allowed_split(self) -> None:
        dataset_path = self.root / "D_success.jsonl"
        self.assertEqual(write_success_samples(self.run_dir, dataset_path), 6)
        manifest_path = self.root / "manifest.jsonl"
        entry = manifest_entry(
            self.fixture.task.task_id,
            "smoke",
            self.fixture.task.repo_id,
        )
        write_jsonl(manifest_path, [entry.to_dict()])

        report = validate_decision_dataset(
            dataset_path,
            "D_success",
            expected_count=6,
            manifest_path=manifest_path,
            allowed_split="smoke",
        )
        wrong_split = validate_decision_dataset(
            dataset_path,
            "D_success",
            expected_count=6,
            manifest_path=manifest_path,
            allowed_split="train",
        )

        self.assertTrue(report.valid, report.errors)
        self.assertFalse(wrong_split.valid)
        self.assertTrue(any("not allowed" in error for error in wrong_split.errors))

    def test_corrected_policy_sample_requires_sandbox_and_review_acceptance(self) -> None:
        value = build_success_samples(self.run_dir)[0].to_dict()
        value.update(
            {
                "dataset": "D_policy",
                "correction_method": "manual_repair",
                "rejected_action": value["target"],
                "error_labels": ["WRONG_FILE"],
                "sandbox_validated": True,
                "review_status": "pending",
            }
        )
        path = self.root / "D_policy.jsonl"
        write_jsonl(path, [value])

        report = validate_decision_dataset(path, "D_policy", expected_count=1)

        self.assertFalse(report.valid)
        self.assertTrue(any("corrected sample is not accepted" in error for error in report.errors))

    def test_invalid_action_is_not_a_target_but_remains_in_later_sample_history(self) -> None:
        fixture = SmokeRepository()
        run_dir = self.root / "invalid-history-run"
        invalid = {
            "intent": "Try an unregistered command.",
            "action": {"type": "bash", "args": {"command": "echo unsafe"}},
        }
        try:
            episode = AgentHarness(
                fixture.config,
                fixture.environment,
                RunArtifactStore(run_dir),
            ).run(fixture.task, ScriptedPolicy([invalid, *smoke_decisions()]))
            samples = build_success_samples(run_dir)
        finally:
            fixture.close()

        self.assertTrue(episode.resolved)
        self.assertEqual(len(samples), 6)
        rendered = "\n".join(message["content"] for message in samples[0].messages)
        self.assertIn("bash", rendered)
        self.assertIn("SCHEMA_VALIDATION_FAILED", rendered)

    def test_corrected_target_is_replayed_from_exact_source_state(self) -> None:
        value = build_success_samples(self.run_dir)[2].to_dict()
        value.update(
            {
                "dataset": "D_policy",
                "correction_method": "manual_repair",
                "rejected_action": {
                    "intent": "Read again instead of applying the required repair.",
                    "action": {
                        "type": "read_file",
                        "args": {"path": "calculator.py", "start_line": 1, "end_line": 20},
                    },
                },
                "error_labels": ["WRONG_PATCH"],
                "sandbox_validated": False,
                "review_status": "accepted",
            }
        )
        candidate = self.root / "corrected_candidate.jsonl"
        output = self.root / "corrected_validated.jsonl"
        ledger = self.root / "corrected_replay.jsonl"
        write_jsonl(candidate, [value])
        fixture = self.fixture
        root = self.root

        class FreshCloneMaterializer:
            def __init__(self) -> None:
                self.calls = 0
                self.cleanup_calls = 0

            def materialize(self, task_id: str, **kwargs):
                if kwargs or task_id != fixture.task.task_id:
                    raise ValueError("Unexpected replay materialization request")
                self.calls += 1
                destination = root / f"replay-{self.calls}"
                result = subprocess.run(
                    ["git", "clone", "--quiet", "--config", "core.autocrlf=false", str(fixture.root), str(destination)],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr)
                environment = RepositoryEnvironment(destination, fixture.profile)

                def cleanup() -> None:
                    self.cleanup_calls += 1

                return fixture.task, environment, cleanup

        materializer = FreshCloneMaterializer()

        report = replay_corrected_samples(
            candidate, [self.run_dir], output, ledger, materializer
        )
        self.assertTrue(report["passed"], read_jsonl(ledger))
        validated = read_jsonl(output)[0]
        replay_entry = read_jsonl(ledger)[0]
        gate = validate_decision_dataset(
            output,
            "D_policy",
            expected_count=1,
            replay_ledger_path=ledger,
        )

        self.assertTrue(report["passed"], replay_entry)
        self.assertTrue(replay_entry["passed"])
        self.assertEqual(
            replay_entry["state_hash_actual"], replay_entry["state_hash_expected"]
        )
        self.assertTrue(validated["sandbox_validated"])
        self.assertTrue(gate.valid, gate.errors)
        self.assertEqual(materializer.cleanup_calls, 1)

    def test_dataset_release_freezes_file_manifest_and_sample_hashes(self) -> None:
        values = []
        for index, sample in enumerate(build_success_samples(self.run_dir)):
            value = sample.to_dict()
            value["split"] = "train"
            if index < 2:
                value["review_status"] = "accepted"
            values.append(value)
        dataset = self.root / "D_success_train.jsonl"
        manifest = self.root / "mvp_manifest.jsonl"
        output = self.root / "D_success_release.json"
        write_jsonl(dataset, values)
        write_jsonl(
            manifest,
            [
                manifest_entry(
                    self.fixture.task.task_id,
                    "train",
                    self.fixture.task.repo_id,
                ).to_dict()
            ],
        )

        release = build_dataset_release_manifest(
            dataset,
            "D_success",
            manifest,
            output,
            expected_count=6,
        )

        self.assertTrue(release["valid"])
        self.assertEqual(release["sample_count"], 6)
        self.assertEqual(len(release["sample_hashes"]), 6)
        self.assertTrue(release["dataset_sha256"].startswith("sha256:"))
        self.assertTrue(release["task_manifest_sha256"].startswith("sha256:"))
        self.assertTrue(output.is_file())
        verified = validate_dataset_release_manifest(
            output,
            dataset,
            "D_success",
            manifest,
            None,
            expected_count=6,
        )
        self.assertTrue(verified["passed"])

        with dataset.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        tampered = validate_dataset_release_manifest(
            output,
            dataset,
            "D_success",
            manifest,
            None,
            expected_count=6,
        )
        self.assertFalse(tampered["passed"])
        self.assertFalse(tampered["checks"]["dataset_hash"])


if __name__ == "__main__":
    unittest.main()
