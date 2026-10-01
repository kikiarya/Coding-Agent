from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.data.gate import validate_decision_dataset
from coding_agent_policy_opt.data.builder import (
    POLICY_CATEGORY_QUOTAS,
    apply_review_decisions,
    assemble_policy_dataset,
    assemble_success_dataset,
    select_review_queue,
)
from coding_agent_policy_opt.hashing import sha256_text
from coding_agent_policy_opt.data.manifest import read_jsonl, write_jsonl


def decision_sample(index: int, dataset: str, category: str) -> dict:
    return {
        "schema_version": "2.0.0",
        "sample_id": f"sample-{index:04d}",
        "source_run_id": f"run-{index // 10}",
        "source_episode_id": f"episode-{index // 5}",
        "source_step_index": index % 5,
        "task_id": f"task-{index // 5}",
        "split": "train",
        "repo_id": f"repo-{index // 20}",
        "dataset": dataset,
        "category": category,
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": f"Fix task {index}."},
        ],
        "target": {
            "intent": "Read the target file.",
            "action": {
                "type": "read_file",
                "args": {"path": "src/a.py", "start_line": 1, "end_line": 20},
            },
        },
        "rejected_action": None,
        "error_labels": [],
        "correction_method": "natural_success",
        "sandbox_validated": True,
        "review_status": "pending",
        "provenance_hash": sha256_text(f"provenance-{index}"),
    }


class DatasetBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_builder_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_success_assembly_is_exact_deterministic_and_never_duplicates(self) -> None:
        source = self.root / "success-candidates.jsonl"
        write_jsonl(
            source,
            [decision_sample(index, "D_success", "planning_tool") for index in range(5)],
        )
        first = self.root / "success-first.jsonl"
        second = self.root / "success-second.jsonl"

        first_report = assemble_success_dataset([source], first, count=3, seed=42)
        assemble_success_dataset([source], second, count=3, seed=42)

        self.assertEqual(first_report["selected"], 3)
        self.assertEqual(read_jsonl(first), read_jsonl(second))
        self.assertEqual(len({value["sample_id"] for value in read_jsonl(first)}), 3)

    def test_policy_assembly_enforces_frozen_category_quotas(self) -> None:
        source = self.root / "policy-candidates.jsonl"
        values = []
        index = 0
        for category, quota in POLICY_CATEGORY_QUOTAS.items():
            for _ in range(quota):
                values.append(decision_sample(index, "D_policy", category))
                index += 1
        write_jsonl(source, values)
        output = self.root / "D_policy.jsonl"

        report = assemble_policy_dataset([source], output, seed=42)
        gate = validate_decision_dataset(
            output,
            "D_policy",
            expected_count=600,
            allowed_split="train",
        )

        self.assertEqual(report["category_counts"], POLICY_CATEGORY_QUOTAS)
        self.assertTrue(gate.valid, gate.errors)

    def test_review_queue_selects_exact_twenty_percent_deterministically(self) -> None:
        dataset = self.root / "dataset.jsonl"
        write_jsonl(
            dataset,
            [decision_sample(index, "D_success", "planning_tool") for index in range(10)],
        )
        first = self.root / "review-first.jsonl"
        second = self.root / "review-second.jsonl"

        first_report = select_review_queue(dataset, first, fraction=0.2, seed=42)
        select_review_queue(dataset, second, fraction=0.2, seed=42)

        self.assertEqual(first_report["review_samples"], 2)
        self.assertEqual(read_jsonl(first), read_jsonl(second))

    def test_review_decisions_are_provenance_checked_and_applied_immutably(self) -> None:
        dataset = self.root / "dataset.jsonl"
        values = [decision_sample(index, "D_success", "planning_tool") for index in range(5)]
        write_jsonl(dataset, values)
        reviews = self.root / "reviews.jsonl"
        write_jsonl(
            reviews,
            [
                {
                    "schema_version": "2.0.0",
                    "sample_id": values[0]["sample_id"],
                    "provenance_hash": values[0]["provenance_hash"],
                    "review_status": "accepted",
                }
            ],
        )
        output = self.root / "reviewed.jsonl"

        report = apply_review_decisions(dataset, reviews, output)

        self.assertEqual(report["applied_reviews"], 1)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(read_jsonl(dataset)[0]["review_status"], "pending")
        self.assertEqual(read_jsonl(output)[0]["review_status"], "accepted")


if __name__ == "__main__":
    unittest.main()
