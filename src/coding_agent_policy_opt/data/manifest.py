from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from jsonschema import ValidationError

from ..hashing import sha256_text
from ..schema import SchemaRegistry


EXTERNAL_SUBSET_COUNT = 50


@dataclass(frozen=True)
class TaskManifestEntry:
    schema_version: str
    task_id: str
    split: str
    source: str
    repo_id: str
    base_commit: str
    docker_image_digest: str | None
    issue_hash: str
    parent_task_id: str | None
    environment_profile: str
    initial_test_status: str
    stable_repeats: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskManifestEntry":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManifestValidationReport:
    entries: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"valid": self.valid}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        values.append(value)
    return values


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_manifest_values(
    raw_entries: list[dict[str, Any]], schemas: SchemaRegistry | None = None
) -> ManifestValidationReport:
    registry = schemas or SchemaRegistry()
    report = ManifestValidationReport()
    entries: list[TaskManifestEntry] = []
    for index, raw in enumerate(raw_entries, start=1):
        try:
            registry.validate("task_manifest", raw)
            entries.append(TaskManifestEntry.from_dict(raw))
        except (ValidationError, TypeError) as exc:
            report.errors.append(f"line {index}: schema validation failed: {exc.message if isinstance(exc, ValidationError) else exc}")

    report.entries = len(entries)
    task_to_split: dict[str, str] = {}
    repo_to_split: dict[str, str] = {}
    parent_to_split: dict[str, str] = {}
    for entry in entries:
        report.split_counts[entry.split] = report.split_counts.get(entry.split, 0) + 1
        previous_task_split = task_to_split.get(entry.task_id)
        if previous_task_split is not None:
            report.errors.append(
                f"duplicate task_id {entry.task_id!r} in {previous_task_split!r} and {entry.split!r}"
            )
        task_to_split[entry.task_id] = entry.split

        if entry.split in {"train", "dev", "test"}:
            previous_repo_split = repo_to_split.get(entry.repo_id)
            if previous_repo_split is not None and previous_repo_split != entry.split:
                report.errors.append(
                    f"repository leakage: {entry.repo_id!r} appears in {previous_repo_split!r} and {entry.split!r}"
                )
            repo_to_split[entry.repo_id] = entry.split

        lineage = entry.parent_task_id or entry.task_id
        previous_lineage_split = parent_to_split.get(lineage)
        if previous_lineage_split is not None and previous_lineage_split != entry.split:
            report.errors.append(
                f"derived-task leakage: lineage {lineage!r} appears in {previous_lineage_split!r} and {entry.split!r}"
            )
        parent_to_split[lineage] = entry.split

        if entry.stable_repeats < 2 and entry.split != "smoke":
            report.warnings.append(
                f"task {entry.task_id!r} has only {entry.stable_repeats} stable initial-test repeats"
            )

    if not entries:
        report.errors.append("manifest contains no valid entries")
    return report


def validate_manifest(path: Path, schemas: SchemaRegistry | None = None) -> ManifestValidationReport:
    return _validate_manifest_values(read_jsonl(path), schemas)


def merge_task_manifests(
    inputs: Iterable[Path],
    output: Path,
    *,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    input_paths = [path.resolve() for path in inputs]
    if not input_paths:
        raise ValueError("At least one input manifest is required")
    if len(input_paths) != len(set(input_paths)):
        raise ValueError("Input manifest paths must be unique")
    if not expected_counts or any(count <= 0 for count in expected_counts.values()):
        raise ValueError("expected_counts must contain positive split counts")

    values: list[dict[str, Any]] = []
    for path in input_paths:
        source_report = validate_manifest(path)
        if not source_report.valid:
            raise ValueError(f"Input manifest is invalid ({path}): {source_report.errors}")
        values.extend(read_jsonl(path))

    split_order = {split: index for index, split in enumerate(expected_counts)}
    unexpected = sorted(
        {str(value.get("split")) for value in values} - set(expected_counts)
    )
    if unexpected:
        raise ValueError(f"Unexpected splits in input manifests: {unexpected}")
    values.sort(
        key=lambda value: (
            split_order[str(value["split"])],
            str(value["task_id"]),
        )
    )
    report = _validate_manifest_values(values)
    for split, expected in expected_counts.items():
        actual = report.split_counts.get(split, 0)
        if actual != expected:
            report.errors.append(
                f"split {split!r} must contain exactly {expected} tasks; found {actual}"
            )
    if not report.valid:
        raise ValueError(f"Merged manifest is invalid: {report.errors}")

    write_jsonl(output, values)
    return {
        "schema_version": "2.0.0",
        "report_type": "manifest_merge",
        "inputs": [str(path) for path in input_paths],
        "output": str(output.resolve()),
        "entries": report.entries,
        "split_counts": report.split_counts,
        "expected_counts": dict(expected_counts),
        "valid": True,
    }


def deterministic_external_subset(
    entries: Iterable[TaskManifestEntry], count: int
) -> list[TaskManifestEntry]:
    if count <= 0:
        raise ValueError("count must be positive")
    eligible = [entry for entry in entries if entry.split == "external"]
    if len(eligible) < count:
        raise ValueError(f"Need {count} external entries, found {len(eligible)}")
    return sorted(eligible, key=lambda item: sha256_text(item.task_id))[:count]


def freeze_external_manifest(
    candidates_path: Path,
    manifest_output: Path,
    exclusion_report_output: Path,
    *,
    count: int = 50,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    for output in (manifest_output.resolve(), exclusion_report_output.resolve()):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    registry = SchemaRegistry()
    candidates = read_jsonl(candidates_path)
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for value in candidates:
        registry.validate("external_candidate", value)
        instance_id = value["instance_id"]
        if instance_id in seen:
            raise ValueError(f"Duplicate external instance_id: {instance_id}")
        seen.add(instance_id)
        reasons: list[str] = []
        if value["initialized"] is not True:
            reasons.append(value["exclusion_reason"] or "environment initialization failed")
        if value["stable_repeats"] < 2:
            reasons.append("fewer than two stable initial-test repeats")
        if value["docker_image_digest"] is None:
            reasons.append("missing Docker image digest")
        if value["initial_test_status"] not in {"passed", "failed"}:
            reasons.append("initial test result is not executable")
        if reasons:
            exclusions.append(
                {"instance_id": instance_id, "reason": "; ".join(reasons)}
            )
        else:
            eligible.append(value)

    selected = sorted(
        eligible, key=lambda value: sha256_text(value["instance_id"])
    )[:count]
    passed = len(selected) == count
    report = {
        "schema_version": "2.0.0",
        "report_type": "external_subset_freeze",
        "selection_method": "sha256(instance_id)_ascending",
        "candidate_source": str(candidates_path.resolve()),
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "excluded_count": len(exclusions),
        "requested_count": count,
        "selected_count": len(selected),
        "selected_instance_ids": [value["instance_id"] for value in selected],
        "exclusions": sorted(exclusions, key=lambda item: item["instance_id"]),
        "passed": passed,
    }
    exclusion_report_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with exclusion_report_output.resolve().open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if passed:
        entries = [
            TaskManifestEntry(
                schema_version="2.0.0",
                task_id=value["instance_id"],
                split="external",
                source="swe-bench-verified",
                repo_id=value["repo_id"],
                base_commit=value["base_commit"],
                docker_image_digest=value["docker_image_digest"],
                issue_hash=value["issue_hash"],
                parent_task_id=None,
                environment_profile=value["environment_profile"],
                initial_test_status=value["initial_test_status"],
                stable_repeats=value["stable_repeats"],
            ).to_dict()
            for value in selected
        ]
        merged_report = _validate_manifest_values(entries)
        if not merged_report.valid:
            raise ValueError(f"Frozen external manifest is invalid: {merged_report.errors}")
        write_jsonl(manifest_output, entries)
    return report
