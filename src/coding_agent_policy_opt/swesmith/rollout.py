from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..harness.artifacts import RunArtifactStore
from ..config import ExperimentConfig, experiment_config_with_seed
from ..harness.runner import AgentHarness
from ..hashing import sha256_text
from ..data.manifest import (
    EXTERNAL_SUBSET_COUNT,
    TaskManifestEntry,
    read_jsonl,
    validate_manifest,
)
from ..agent.policy import AgentPolicy
from ..schema import SchemaRegistry
from .config import (
    SweSmithEnvironmentConfig,
    load_selected_swesmith_tasks,
)
from .environment import (
    WorkerClient,
    WslSweSmithEnvironment,
    default_swesmith_worker,
)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _manifest_entries(path: Path, split: str) -> dict[str, TaskManifestEntry]:
    report = validate_manifest(path)
    if not report.valid:
        raise ValueError(f"Task manifest is invalid: {report.errors}")
    registry = SchemaRegistry()
    entries: dict[str, TaskManifestEntry] = {}
    for value in read_jsonl(path):
        registry.validate("task_manifest", value)
        entry = TaskManifestEntry.from_dict(value)
        if entry.split != split:
            continue
        if entry.task_id in entries:
            raise ValueError(f"Duplicate manifest task_id: {entry.task_id}")
        entries[entry.task_id] = entry
    if not entries:
        raise ValueError(f"Manifest contains no tasks for split={split!r}")
    return entries


def _image_matches(environment: WslSweSmithEnvironment, expected: str | None) -> bool:
    if expected is None:
        return False
    return expected == environment.image_id or expected in environment.image_digests


def run_swesmith_rollouts(
    inventory_path: Path,
    oracle_path: Path,
    manifest_path: Path,
    report_output: Path,
    experiment_config: ExperimentConfig,
    environment_config: SweSmithEnvironmentConfig,
    policy_factory: Callable[[int], AgentPolicy],
    *,
    split: str,
    seeds: tuple[int, ...],
    run_label: str,
    worker: WorkerClient | None = None,
) -> dict[str, Any]:
    if report_output.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {report_output.resolve()}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique values")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_label):
        raise ValueError("run_label may contain only letters, digits, dot, underscore, and hyphen")
    selected = load_selected_swesmith_tasks(inventory_path, oracle_path, split=split)
    selected_by_id = {item.task.instance_id: item for item in selected}
    manifest = _manifest_entries(manifest_path, split)
    if not set(manifest).issubset(selected_by_id):
        raise ValueError(
            "Frozen manifest contains tasks absent from the selected source: "
            f"{sorted(set(manifest) - set(selected_by_id))}"
        )
    # External is frozen after the SWE-smith environment profile is loaded and
    # therefore is intentionally not part of the repository-disjoint MVP
    # counts in swesmith_server.yaml.
    expected_tasks = (
        EXTERNAL_SUBSET_COUNT
        if split == "external"
        else environment_config.counts[split]
    )
    minimum_tasks = (
        math.ceil(expected_tasks * 0.95) if split == "smoke" else expected_tasks
    )
    if not minimum_tasks <= len(manifest) <= expected_tasks:
        raise ValueError(
            f"Frozen {split} manifest must contain between {minimum_tasks} and "
            f"{expected_tasks} tasks; found {len(manifest)}"
        )
    client = worker or default_swesmith_worker(environment_config)
    artifact_root = experiment_config.run.artifacts_dir.resolve() / "rollouts" / run_label
    results: list[dict[str, Any]] = []

    for task_id in sorted(manifest):
        selected_task = selected_by_id[task_id]
        manifest_entry = manifest[task_id]
        for seed in seeds:
            environment: WslSweSmithEnvironment | None = None
            task_hash = sha256_text(task_id).removeprefix("sha256:")[:12]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_name = (
                f"{run_label}_{task_hash}_seed{seed}_{timestamp}_{uuid.uuid4().hex[:8]}"
            )
            run_dir = artifact_root / run_name
            result: dict[str, Any] = {
                "task_id": task_id,
                "split": split,
                "seed": seed,
                "run_dir": str(run_dir),
                "completed": False,
                "resolved": False,
                "error": None,
            }
            try:
                seeded_config = experiment_config_with_seed(experiment_config, seed)
                environment = WslSweSmithEnvironment(
                    selected_task.task,
                    environment_config,
                    worker=client,
                    tool_timeout_seconds=seeded_config.harness.tool_timeout_seconds,
                )
                if environment.base_commit != manifest_entry.base_commit:
                    raise RuntimeError(
                        "Materialized base commit differs from the frozen manifest"
                    )
                if not _image_matches(environment, manifest_entry.docker_image_digest):
                    raise RuntimeError(
                        "Materialized image digest differs from the frozen manifest"
                    )
                task = selected_task.task.to_task_spec(split, environment.base_commit)
                store = RunArtifactStore(run_dir)
                episode = AgentHarness(seeded_config, environment, store).run(
                    task, policy_factory(seed)
                )
                result.update(
                    {
                        "completed": True,
                        "resolved": episode.resolved,
                        "environment_valid": episode.environment_valid,
                        "episode_id": episode.episode_id,
                        "termination_reason": episode.termination_reason,
                    }
                )
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if environment is not None:
                    result["cleanup_ok"] = environment.close()
                    result["cleanup_error"] = environment.cleanup_error
            results.append(result)

    expected_rollouts = len(manifest) * len(seeds)
    completed_rollouts = sum(bool(item["completed"]) for item in results)
    cleanup_failures = sum(item.get("cleanup_ok") is False for item in results)
    report = {
        "schema_version": "2.0.0",
        "report_type": "swesmith_rollout_batch",
        "run_label": run_label,
        "split": split,
        "seeds": list(seeds),
        "tasks": len(manifest),
        "expected_tasks": expected_tasks,
        "minimum_tasks": minimum_tasks,
        "task_coverage_rate": len(manifest) / expected_tasks,
        "expected_rollouts": expected_rollouts,
        "completed_rollouts": completed_rollouts,
        "environment_valid_rollouts": sum(
            bool(item.get("environment_valid")) for item in results
        ),
        "resolved_rollouts": sum(bool(item["resolved"]) for item in results),
        "cleanup_failures": cleanup_failures,
        "passed": completed_rollouts == expected_rollouts and cleanup_failures == 0,
        "results": results,
    }
    _write_json_exclusive(report_output, report)
    return report


def run_base_collection(
    inventory_path: Path,
    oracle_path: Path,
    manifest_path: Path,
    report_output: Path,
    experiment_config: ExperimentConfig,
    environment_config: SweSmithEnvironmentConfig,
    policy_factory: Callable[[int], AgentPolicy],
    *,
    worker: WorkerClient | None = None,
) -> dict[str, Any]:
    return run_swesmith_rollouts(
        inventory_path,
        oracle_path,
        manifest_path,
        report_output,
        experiment_config,
        environment_config,
        policy_factory,
        split="train",
        seeds=(42, 43),
        run_label="base_collection",
        worker=worker,
    )
