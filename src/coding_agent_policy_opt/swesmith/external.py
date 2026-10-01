from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..data.manifest import (
    EXTERNAL_SUBSET_COUNT,
    TaskManifestEntry,
    read_jsonl,
    validate_manifest,
    write_jsonl,
)
from ..hashing import sha256_text
from ..schema import SchemaRegistry
from .config import UpstreamSweSmithTask


EXTERNAL_DATASET_ID = "SWE-bench Verified"


def _validate_external_revision(dataset_revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", dataset_revision):
        raise ValueError(
            "external dataset_revision must be a 40-character lowercase commit SHA"
        )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.resolve()}")
    return value


def _normalise_external_task(value: dict[str, Any]) -> UpstreamSweSmithTask:
    """Accept the common SWE-bench image-name alias without hiding other fields."""

    normalized = dict(value)
    if "image_name" not in normalized and "docker_image_name" in normalized:
        normalized["image_name"] = normalized["docker_image_name"]
    return UpstreamSweSmithTask.from_dict(normalized)


def build_external_inventory(
    source_path: Path,
    candidates_path: Path,
    freeze_report_path: Path,
    manifest_path: Path,
    inventory_output: Path,
    oracle_output: Path,
    *,
    dataset_revision: str,
    expected_count: int = EXTERNAL_SUBSET_COUNT,
) -> dict[str, Any]:
    """Materialize executable inventory/oracle for a frozen external set."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    _validate_external_revision(dataset_revision)
    for output in (inventory_output.resolve(), oracle_output.resolve()):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    freeze_report = _read_object(freeze_report_path)
    if freeze_report.get("report_type") != "external_subset_freeze":
        raise ValueError("freeze report must be an external_subset_freeze report")
    if freeze_report.get("passed") is not True:
        raise ValueError("external subset freeze report did not pass")
    selected_ids = freeze_report.get("selected_instance_ids")
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != expected_count
        or len(set(selected_ids)) != expected_count
    ):
        raise ValueError(
            f"freeze report must contain exactly {expected_count} unique selected IDs"
        )

    registry = SchemaRegistry()
    candidates: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(candidates_path):
        registry.validate("external_candidate", value)
        instance_id = value["instance_id"]
        if instance_id in candidates:
            raise ValueError(f"Duplicate external candidate: {instance_id}")
        candidates[instance_id] = value

    source_tasks: dict[str, UpstreamSweSmithTask] = {}
    for value in read_jsonl(source_path):
        task = _normalise_external_task(value)
        if task.instance_id in source_tasks:
            raise ValueError(f"Duplicate external source task: {task.instance_id}")
        source_tasks[task.instance_id] = task

    manifest_report = validate_manifest(manifest_path)
    if not manifest_report.valid:
        raise ValueError(f"External manifest is invalid: {manifest_report.errors}")
    manifest = {
        value["task_id"]: TaskManifestEntry.from_dict(value)
        for value in read_jsonl(manifest_path)
    }
    if set(manifest) != set(selected_ids):
        raise ValueError("Frozen external manifest IDs differ from the freeze report")

    inventory: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for task_id in sorted(selected_ids):
        candidate = candidates.get(task_id)
        task = source_tasks.get(task_id)
        entry = manifest[task_id]
        if candidate is None or task is None:
            raise ValueError(f"Frozen external task is missing candidate or source: {task_id}")
        if not candidate["initialized"]:
            raise ValueError(f"Frozen external candidate is not initialized: {task_id}")
        if candidate["stable_repeats"] < 2:
            raise ValueError(f"Frozen external candidate is not stable: {task_id}")
        if candidate["initial_test_status"] not in {"passed", "failed"}:
            raise ValueError(f"Frozen external candidate has no executable tests: {task_id}")
        if candidate["docker_image_digest"] is None:
            raise ValueError(f"Frozen external candidate has no image digest: {task_id}")
        if candidate["base_commit"] != entry.base_commit:
            raise ValueError(f"Base commit mismatch for external task: {task_id}")
        if candidate["docker_image_digest"] != entry.docker_image_digest:
            raise ValueError(f"Docker image digest mismatch for external task: {task_id}")
        if candidate["issue_hash"] != sha256_text(task.problem_statement):
            raise ValueError(f"Issue hash mismatch for external task: {task_id}")
        if entry.issue_hash != candidate["issue_hash"]:
            raise ValueError(f"Manifest issue hash mismatch for external task: {task_id}")
        if candidate["repo_id"] not in {task.repo, task.source_repo_key}:
            raise ValueError(f"Repository mismatch for external task: {task_id}")
        if entry.split != "external":
            raise ValueError(f"External manifest entry has wrong split: {task_id}")
        inventory_entry = task.inventory_entry("external", dataset_revision)
        oracle_entry = task.oracle_entry()
        registry.validate("swe_smith_inventory", inventory_entry)
        registry.validate("swe_smith_oracle", oracle_entry)
        inventory.append(inventory_entry)
        oracle.append(oracle_entry)

    write_jsonl(inventory_output, inventory)
    write_jsonl(oracle_output, oracle)
    return {
        "schema_version": "2.0.0",
        "report_type": "external_inventory_build",
        "source": str(source_path.resolve()),
        "candidates": str(candidates_path.resolve()),
        "freeze_report": str(freeze_report_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "inventory": str(inventory_output.resolve()),
        "oracle": str(oracle_output.resolve()),
        "dataset_id": EXTERNAL_DATASET_ID,
        "dataset_revision": dataset_revision,
        "count": len(inventory),
        "passed": len(inventory) == expected_count,
    }


def build_external_source_inventory(
    source_path: Path,
    inventory_output: Path,
    oracle_output: Path,
    *,
    dataset_revision: str,
) -> dict[str, Any]:
    """Build a provisional external inventory before environment stabilization.

    This inventory contains every source task and is only an input to the
    external candidate-stability pass. It must not be used for final scoring;
    the final 50-task inventory is rebuilt after freeze using
    :func:`build_external_inventory`.
    """

    _validate_external_revision(dataset_revision)
    for output in (inventory_output.resolve(), oracle_output.resolve()):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    registry = SchemaRegistry()
    tasks: dict[str, UpstreamSweSmithTask] = {}
    for value in read_jsonl(source_path):
        task = _normalise_external_task(value)
        if task.instance_id in tasks:
            raise ValueError(f"Duplicate external source task: {task.instance_id}")
        tasks[task.instance_id] = task

    inventory: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        inventory_entry = task.inventory_entry("external", dataset_revision)
        oracle_entry = task.oracle_entry()
        registry.validate("swe_smith_inventory", inventory_entry)
        registry.validate("swe_smith_oracle", oracle_entry)
        inventory.append(inventory_entry)
        oracle.append(oracle_entry)

    write_jsonl(inventory_output, inventory)
    write_jsonl(oracle_output, oracle)
    return {
        "schema_version": "2.0.0",
        "report_type": "external_source_inventory_build",
        "dataset_id": EXTERNAL_DATASET_ID,
        "dataset_revision": dataset_revision,
        "source": str(source_path.resolve()),
        "inventory": str(inventory_output.resolve()),
        "oracle": str(oracle_output.resolve()),
        "count": len(inventory),
        "passed": bool(inventory),
        "final_scoring_ready": False,
    }
