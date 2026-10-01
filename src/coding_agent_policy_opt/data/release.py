from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .gate import validate_decision_dataset
from ..hashing import sha256_json
from .manifest import read_jsonl
from ..schema import SchemaRegistry


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _increment(target: dict[str, int], key: str) -> None:
    target[key] = target.get(key, 0) + 1


def build_dataset_release_manifest(
    dataset_path: Path,
    dataset: str,
    task_manifest_path: Path,
    output_path: Path,
    *,
    expected_count: int = 600,
    replay_ledger_path: Path | None = None,
) -> dict[str, Any]:
    resolved_output = output_path.resolve()
    if resolved_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved_output}")
    validation = validate_decision_dataset(
        dataset_path,
        dataset,
        expected_count=expected_count,
        manifest_path=task_manifest_path,
        allowed_split="train",
        replay_ledger_path=replay_ledger_path,
        min_review_fraction=0.20 if dataset == "D_success" else None,
    )
    if not validation.valid:
        raise ValueError(f"Dataset release gate failed: {validation.errors}")

    values = read_jsonl(dataset_path)
    category_counts: dict[str, int] = {}
    correction_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for value in values:
        _increment(category_counts, value["category"])
        _increment(correction_counts, value["correction_method"])
        _increment(review_counts, value["review_status"])
        _increment(split_counts, value["split"])

    release = {
        "schema_version": "2.0.0",
        "report_type": "dataset_release_manifest",
        "generator_version": "dataset_release_v1",
        "dataset": dataset,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _file_hash(dataset_path),
        "task_manifest_path": str(task_manifest_path.resolve()),
        "task_manifest_sha256": _file_hash(task_manifest_path),
        "replay_ledger_path": (
            str(replay_ledger_path.resolve()) if replay_ledger_path is not None else None
        ),
        "replay_ledger_sha256": (
            _file_hash(replay_ledger_path) if replay_ledger_path is not None else None
        ),
        "sample_count": len(values),
        "category_counts": category_counts,
        "correction_counts": correction_counts,
        "review_counts": review_counts,
        "split_counts": split_counts,
        "source_run_ids": sorted({value["source_run_id"] for value in values}),
        "sample_hashes": [
            {
                "sample_id": value["sample_id"],
                "provenance_hash": value["provenance_hash"],
                "content_hash": sha256_json(value),
            }
            for value in sorted(values, key=lambda item: item["sample_id"])
        ],
        "validation": validation.to_dict(),
        "valid": True,
    }
    SchemaRegistry().validate("dataset_release", release)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(release, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return release


def validate_dataset_release_manifest(
    release_path: Path,
    dataset_path: Path,
    dataset: str,
    task_manifest_path: Path,
    replay_ledger_path: Path | None,
    *,
    expected_count: int = 600,
) -> dict[str, Any]:
    value = json.loads(release_path.resolve().read_text(encoding="utf-8"))
    SchemaRegistry().validate("dataset_release", value)
    expected_replay_path = (
        str(replay_ledger_path.resolve()) if replay_ledger_path is not None else None
    )
    expected_replay_hash = (
        _file_hash(replay_ledger_path) if replay_ledger_path is not None else None
    )
    checks = {
        "dataset_name": value["dataset"] == dataset,
        "dataset_path": value["dataset_path"] == str(dataset_path.resolve()),
        "dataset_hash": value["dataset_sha256"] == _file_hash(dataset_path),
        "task_manifest_path": value["task_manifest_path"]
        == str(task_manifest_path.resolve()),
        "task_manifest_hash": value["task_manifest_sha256"]
        == _file_hash(task_manifest_path),
        "replay_ledger_path": value["replay_ledger_path"] == expected_replay_path,
        "replay_ledger_hash": value["replay_ledger_sha256"]
        == expected_replay_hash,
        "sample_count": value["sample_count"] == expected_count,
        "sample_hash_count": len(value["sample_hashes"]) == expected_count,
    }
    return {
        "schema_version": "2.0.0",
        "report_type": "dataset_release_verification",
        "release_manifest": str(release_path.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }
