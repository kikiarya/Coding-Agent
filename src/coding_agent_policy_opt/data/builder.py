from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..hashing import sha256_text
from .manifest import read_jsonl, write_jsonl
from ..schema import SchemaRegistry


POLICY_CATEGORY_QUOTAS = {
    "planning_tool": 240,
    "verification": 150,
    "recovery": 150,
    "termination_efficiency": 60,
}


def _load_candidates(paths: Iterable[Path], dataset: str) -> list[dict[str, Any]]:
    registry = SchemaRegistry()
    result: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_provenance: set[str] = set()
    for path in paths:
        for value in read_jsonl(path):
            registry.validate("decision_sample", value)
            registry.validate("agent_action", value["target"])
            if value["dataset"] != dataset:
                raise ValueError(
                    f"Candidate {value['sample_id']} belongs to {value['dataset']}, not {dataset}"
                )
            if value["sample_id"] in seen_samples:
                raise ValueError(f"Duplicate candidate sample_id: {value['sample_id']}")
            if value["provenance_hash"] in seen_provenance:
                raise ValueError(
                    f"Duplicate candidate provenance_hash: {value['provenance_hash']}"
                )
            seen_samples.add(value["sample_id"])
            seen_provenance.add(value["provenance_hash"])
            result.append(value)
    return result


def _deterministic_order(values: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        values,
        key=lambda value: sha256_text(f"{seed}:{value['sample_id']}"),
    )


def assemble_success_dataset(
    candidate_paths: Iterable[Path],
    output_path: Path,
    *,
    count: int = 600,
    seed: int = 42,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    candidates = _load_candidates(candidate_paths, "D_success")
    eligible = [
        value
        for value in candidates
        if value["sandbox_validated"] is True
        and value["correction_method"] in {"natural_success", "recovery_success"}
    ]
    if len(eligible) < count:
        raise ValueError(f"D_success requires {count} eligible samples, found {len(eligible)}")
    selected = _deterministic_order(eligible, seed)[:count]
    write_jsonl(output_path, selected)
    return {
        "schema_version": "2.0.0",
        "dataset": "D_success",
        "candidates": len(candidates),
        "eligible": len(eligible),
        "selected": len(selected),
        "seed": seed,
        "output": str(output_path.resolve()),
    }


def assemble_policy_dataset(
    candidate_paths: Iterable[Path],
    output_path: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    candidates = _load_candidates(candidate_paths, "D_policy")
    groups: dict[str, list[dict[str, Any]]] = {
        category: [] for category in POLICY_CATEGORY_QUOTAS
    }
    for value in candidates:
        if value["sandbox_validated"] is not True:
            continue
        if value["correction_method"] in {
            "controlled_perturbation_repair",
            "manual_repair",
        } and value["review_status"] != "accepted":
            continue
        groups[value["category"]].append(value)

    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for category, quota in POLICY_CATEGORY_QUOTAS.items():
        ordered = _deterministic_order(groups[category], seed)
        if len(ordered) < quota:
            raise ValueError(
                f"D_policy category {category} requires {quota} eligible samples, "
                f"found {len(ordered)}"
            )
        chosen = ordered[:quota]
        selected.extend(chosen)
        selected_counts[category] = len(chosen)
    selected = _deterministic_order(selected, seed)
    write_jsonl(output_path, selected)
    return {
        "schema_version": "2.0.0",
        "dataset": "D_policy",
        "candidates": len(candidates),
        "selected": len(selected),
        "category_counts": selected_counts,
        "seed": seed,
        "output": str(output_path.resolve()),
    }


def select_review_queue(
    dataset_path: Path,
    output_path: Path,
    *,
    fraction: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    values = read_jsonl(dataset_path)
    count = max(1, round(len(values) * fraction)) if values else 0
    selected = _deterministic_order(values, seed)[:count]
    queue = [
        {
            "schema_version": "2.0.0",
            "sample_id": value["sample_id"],
            "provenance_hash": value["provenance_hash"],
            "review_status": value["review_status"],
        }
        for value in selected
    ]
    registry = SchemaRegistry()
    for entry in queue:
        registry.validate("review_queue", entry)
    write_jsonl(output_path, queue)
    return {
        "schema_version": "2.0.0",
        "dataset_samples": len(values),
        "review_samples": len(queue),
        "fraction": fraction,
        "seed": seed,
        "output": str(output_path.resolve()),
    }


def apply_review_decisions(
    dataset_path: Path,
    reviews_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    registry = SchemaRegistry()
    reviews: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(reviews_path):
        registry.validate("review_queue", value)
        if value["review_status"] == "pending":
            raise ValueError(f"Review for {value['sample_id']} is still pending")
        if value["sample_id"] in reviews:
            raise ValueError(f"Duplicate review sample_id: {value['sample_id']}")
        reviews[value["sample_id"]] = value

    samples = read_jsonl(dataset_path)
    updated: list[dict[str, Any]] = []
    seen_reviews: set[str] = set()
    for sample in samples:
        registry.validate("decision_sample", sample)
        review = reviews.get(sample["sample_id"])
        value = dict(sample)
        if review is not None:
            if review["provenance_hash"] != sample["provenance_hash"]:
                raise ValueError(
                    f"Review provenance mismatch for {sample['sample_id']}"
                )
            value["review_status"] = review["review_status"]
            seen_reviews.add(sample["sample_id"])
        registry.validate("decision_sample", value)
        updated.append(value)
    unused = sorted(set(reviews) - seen_reviews)
    if unused:
        raise ValueError(f"Reviews reference samples absent from the dataset: {unused}")
    write_jsonl(output_path, updated)
    return {
        "schema_version": "2.0.0",
        "dataset_samples": len(samples),
        "applied_reviews": len(seen_reviews),
        "accepted": sum(value["review_status"] == "accepted" for value in updated),
        "rejected": sum(value["review_status"] == "rejected" for value in updated),
        "pending": sum(value["review_status"] == "pending" for value in updated),
        "output": str(output_path.resolve()),
    }
