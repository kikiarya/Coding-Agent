from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent_policy_opt.data.manifest import write_jsonl
from coding_agent_policy_opt.evaluation.gates import evaluate_sft_gate, select_checkpoint


def write_summary(
    path: Path,
    *,
    resolve: float,
    verification: float,
    duplicate: float,
    tokens: float,
    invalid: float = 0.0,
    task_ids: tuple[str, ...] = ("a", "b"),
) -> None:
    value = {
        "schema_version": "2.0.0",
        "report_type": "evaluation_summary",
        "metrics": {
            "resolve_rate": resolve,
            "resolved_count": round(resolve * len(task_ids)),
            "post_patch_verification_rate": verification,
            "duplicate_action_rate": duplicate,
            "invalid_action_rate": invalid,
            "costs": {"all_valid": {"total_tokens": {"mean": tokens}}},
            "tasks": [{"task_id": task_id} for task_id in task_ids],
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")


class CheckpointSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_selection_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selection_uses_resolve_verification_tokens_then_earlier_epoch(self) -> None:
        reports = []
        values = [
            (1, 0.5, 0.8, 100.0),
            (2, 0.5, 0.9, 120.0),
            (3, 0.5, 0.9, 110.0),
            (4, 0.5, 0.9, 110.0),
        ]
        for epoch, resolve, verification, tokens in values:
            report = self.root / f"epoch-{epoch}.json"
            write_summary(
                report,
                resolve=resolve,
                verification=verification,
                duplicate=0.1,
                tokens=tokens,
            )
            reports.append(
                {
                    "schema_version": "2.0.0",
                    "epoch": epoch,
                    "checkpoint_path": str(self.root / f"checkpoint-{epoch}"),
                    "evaluation_report": str(report),
                }
            )
        candidates = self.root / "candidates.jsonl"
        write_jsonl(candidates, reports)

        result = select_checkpoint(candidates)

        self.assertEqual(result["selected"]["epoch"], 3)
        self.assertEqual([item["epoch"] for item in result["ranking"]], [3, 4, 2, 1])

    def test_selection_rejects_different_dev_task_sets(self) -> None:
        first = self.root / "first.json"
        second = self.root / "second.json"
        write_summary(first, resolve=0.5, verification=0.5, duplicate=0.1, tokens=100)
        write_summary(
            second,
            resolve=0.5,
            verification=0.5,
            duplicate=0.1,
            tokens=100,
            task_ids=("a", "c"),
        )
        candidates = self.root / "candidates.jsonl"
        write_jsonl(
            candidates,
            [
                {
                    "schema_version": "2.0.0",
                    "epoch": 1,
                    "checkpoint_path": "checkpoint-1",
                    "evaluation_report": str(first),
                },
                {
                    "schema_version": "2.0.0",
                    "epoch": 2,
                    "checkpoint_path": "checkpoint-2",
                    "evaluation_report": str(second),
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "identical dev task set"):
            select_checkpoint(candidates)


class SFTGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="capo_sft_gate_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _reports(
        self,
        *,
        success_resolve: float,
        policy_resolve: float,
        success_verification: float,
        policy_verification: float,
        success_duplicate: float,
        policy_duplicate: float,
        success_invalid: float = 0.0,
        policy_invalid: float = 0.0,
    ) -> tuple[Path, Path, Path]:
        success = self.root / "success.json"
        policy = self.root / "policy.json"
        write_summary(
            success,
            resolve=success_resolve,
            verification=success_verification,
            duplicate=success_duplicate,
            tokens=100,
            invalid=success_invalid,
        )
        write_summary(
            policy,
            resolve=policy_resolve,
            verification=policy_verification,
            duplicate=policy_duplicate,
            tokens=100,
            invalid=policy_invalid,
        )
        comparison = self.root / "comparison.json"
        metric_values = {
            "resolve_rate": (success_resolve, policy_resolve),
            "post_patch_verification_rate": (
                success_verification,
                policy_verification,
            ),
            "duplicate_action_rate": (success_duplicate, policy_duplicate),
            "invalid_action_rate": (success_invalid, policy_invalid),
        }
        comparison.write_text(
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "report_type": "paired_model_comparison",
                    "bootstrap_samples": 10_000,
                    "seed": 42,
                    "comparison": {
                        "task_count": 2,
                        "resolve_exact_mcnemar": {"two_sided_p_value": 1.0},
                        "paired_bootstrap": {
                            name: {
                                "baseline": baseline,
                                "candidate": candidate,
                                "difference_candidate_minus_baseline": candidate - baseline,
                                "confidence_interval_95": [
                                    candidate - baseline,
                                    candidate - baseline,
                                ],
                            }
                            for name, (baseline, candidate) in metric_values.items()
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return success, policy, comparison

    def test_gate_passes_resolve_noninferiority_and_verification_improvement(self) -> None:
        success, policy, comparison = self._reports(
            success_resolve=0.50,
            policy_resolve=0.47,
            success_verification=0.60,
            policy_verification=0.72,
            success_duplicate=0.10,
            policy_duplicate=0.10,
        )

        result = evaluate_sft_gate(success, policy, comparison)

        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["resolve_non_inferiority"])
        self.assertTrue(result["checks"]["verification_improvement"])

    def test_gate_can_pass_via_relative_duplicate_reduction(self) -> None:
        success, policy, comparison = self._reports(
            success_resolve=0.50,
            policy_resolve=0.50,
            success_verification=0.60,
            policy_verification=0.60,
            success_duplicate=0.10,
            policy_duplicate=0.08,
        )

        result = evaluate_sft_gate(success, policy, comparison)

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["observed"]["duplicate_relative_reduction"], 0.2)

    def test_gate_fails_when_resolve_drops_more_than_five_points(self) -> None:
        success, policy, comparison = self._reports(
            success_resolve=0.50,
            policy_resolve=0.44,
            success_verification=0.60,
            policy_verification=0.75,
            success_duplicate=0.10,
            policy_duplicate=0.05,
        )

        result = evaluate_sft_gate(success, policy, comparison)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["resolve_non_inferiority"])

    def test_gate_fails_when_invalid_action_rate_rises_over_five_points(self) -> None:
        success, policy, comparison = self._reports(
            success_resolve=0.50,
            policy_resolve=0.50,
            success_verification=0.60,
            policy_verification=0.75,
            success_duplicate=0.10,
            policy_duplicate=0.10,
            success_invalid=0.01,
            policy_invalid=0.07,
        )

        result = evaluate_sft_gate(success, policy, comparison)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["invalid_action_non_degradation"])


if __name__ == "__main__":
    unittest.main()
