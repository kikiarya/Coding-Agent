from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

from .manifest import TaskManifestEntry, read_jsonl
from ..schema import SchemaRegistry


@dataclass
class DatasetValidationReport:
    dataset: str
    samples: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    review_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"valid": self.valid}


def validate_decision_dataset(
    path: Path,
    dataset: str,
    *,
    expected_count: int | None = None,
    require_reviewed: bool = False,
    min_review_fraction: float | None = None,
    manifest_path: Path | None = None,
    allowed_split: str | None = None,
    replay_ledger_path: Path | None = None,
) -> DatasetValidationReport:
    if dataset not in {"D_success", "D_policy"}:
        raise ValueError("dataset must be D_success or D_policy")
    if min_review_fraction is not None and not 0.0 <= min_review_fraction <= 1.0:
        raise ValueError("min_review_fraction must be between 0 and 1")
    registry = SchemaRegistry()
    report = DatasetValidationReport(dataset=dataset)
    seen_sample_ids: set[str] = set()
    seen_provenance: set[str] = set()
    manifest_by_task: dict[str, TaskManifestEntry] = {}
    replay_by_sample: dict[str, dict[str, Any]] = {}
    if manifest_path is not None:
        for raw in read_jsonl(manifest_path):
            registry.validate("task_manifest", raw)
            entry = TaskManifestEntry.from_dict(raw)
            if entry.task_id in manifest_by_task:
                raise ValueError(f"duplicate task_id in manifest: {entry.task_id}")
            manifest_by_task[entry.task_id] = entry
    if replay_ledger_path is not None:
        for raw in read_jsonl(replay_ledger_path):
            registry.validate("corrected_replay", raw)
            sample_id = raw["sample_id"]
            if sample_id in replay_by_sample:
                raise ValueError(f"duplicate sample_id in replay ledger: {sample_id}")
            replay_by_sample[sample_id] = raw

    for line_number, value in enumerate(read_jsonl(path), start=1):
        try:
            registry.validate("decision_sample", value)
            registry.validate("agent_action", value["target"])
        except (ValidationError, KeyError) as exc:
            message = exc.message if isinstance(exc, ValidationError) else str(exc)
            report.errors.append(f"line {line_number}: {message}")
            continue

        report.samples += 1
        if value["dataset"] != dataset:
            report.errors.append(
                f"line {line_number}: expected dataset {dataset}, found {value['dataset']}"
            )
        category = value["category"]
        report.category_counts[category] = report.category_counts.get(category, 0) + 1
        review_status = value["review_status"]
        report.review_counts[review_status] = report.review_counts.get(review_status, 0) + 1

        if allowed_split is not None and value["split"] != allowed_split:
            report.errors.append(
                f"line {line_number}: split {value['split']} is not allowed; expected {allowed_split}"
            )
        if manifest_by_task:
            manifest_entry = manifest_by_task.get(value["task_id"])
            if manifest_entry is None:
                report.errors.append(
                    f"line {line_number}: task {value['task_id']} is missing from manifest"
                )
            else:
                if value["split"] != manifest_entry.split:
                    report.errors.append(
                        f"line {line_number}: split does not match manifest for {value['task_id']}"
                    )
                if value["repo_id"] != manifest_entry.repo_id:
                    report.errors.append(
                        f"line {line_number}: repo_id does not match manifest for {value['task_id']}"
                    )

        if value["sample_id"] in seen_sample_ids:
            report.errors.append(f"line {line_number}: duplicate sample_id {value['sample_id']}")
        seen_sample_ids.add(value["sample_id"])
        if value["provenance_hash"] in seen_provenance:
            report.errors.append(
                f"line {line_number}: duplicate provenance_hash {value['provenance_hash']}"
            )
        seen_provenance.add(value["provenance_hash"])

        if value["correction_method"] in {"controlled_perturbation_repair", "manual_repair"}:
            if value["rejected_action"] is None:
                report.errors.append(f"line {line_number}: corrected sample has no rejected_action")
            if not value["error_labels"]:
                report.errors.append(f"line {line_number}: corrected sample has no error_labels")
            if value["sandbox_validated"] is not True:
                report.errors.append(f"line {line_number}: corrected sample is not sandbox validated")
            if value["review_status"] != "accepted":
                report.errors.append(f"line {line_number}: corrected sample is not accepted")
            replay = replay_by_sample.get(value["sample_id"])
            if replay is None:
                report.errors.append(
                    f"line {line_number}: corrected sample has no replay ledger evidence"
                )
            elif (
                replay["passed"] is not True
                or replay["cleanup_ok"] is not True
                or replay["provenance_hash"] != value["provenance_hash"]
                or replay["source_run_id"] != value["source_run_id"]
                or replay["source_episode_id"] != value["source_episode_id"]
                or replay["source_step_index"] != value["source_step_index"]
                or replay["task_id"] != value["task_id"]
            ):
                report.errors.append(
                    f"line {line_number}: corrected sample replay evidence is invalid"
                )
        if require_reviewed and value["review_status"] != "accepted":
            report.errors.append(f"line {line_number}: sample review is not accepted")

    if expected_count is not None and report.samples != expected_count:
        report.errors.append(f"expected {expected_count} valid samples, found {report.samples}")
    if min_review_fraction is not None and report.samples:
        accepted_fraction = report.review_counts.get("accepted", 0) / report.samples
        if accepted_fraction < min_review_fraction:
            report.errors.append(
                f"accepted review fraction {accepted_fraction:.4f} is below "
                f"required {min_review_fraction:.4f}"
            )

    if dataset == "D_policy" and expected_count == 600:
        expected_categories = {
            "planning_tool": 240,
            "verification": 150,
            "recovery": 150,
            "termination_efficiency": 60,
        }
        for category, count in expected_categories.items():
            actual = report.category_counts.get(category, 0)
            if actual != count:
                report.errors.append(
                    f"D_policy category {category} requires {count} samples, found {actual}"
                )
    if report.samples == 0:
        report.errors.append("dataset contains no valid samples")
    return report
