from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.manifest import EXTERNAL_SUBSET_COUNT, TaskManifestEntry, write_jsonl
from ..models import ToolResult
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


def _test_state(result: ToolResult) -> dict[str, Any] | None:
    value = result.data.get("test_state")
    return dict(value) if isinstance(value, dict) else None


def stabilize_swesmith_tasks(
    inventory_path: Path,
    oracle_path: Path,
    manifest_output: Path,
    report_output: Path,
    environment_config: SweSmithEnvironmentConfig,
    *,
    split: str,
    worker: WorkerClient | None = None,
    tool_timeout_seconds: int = 600,
) -> dict[str, Any]:
    for output in (manifest_output.resolve(), report_output.resolve()):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    selected = load_selected_swesmith_tasks(
        inventory_path, oracle_path, split=split
    )
    client = worker or default_swesmith_worker(environment_config)
    registry = SchemaRegistry()
    entries: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    initialized = 0
    consistent = 0
    bug_reproduced = 0
    container_ids: list[str] = []

    for selected_task in selected:
        environment: WslSweSmithEnvironment | None = None
        task_result: dict[str, Any] = {
            "task_id": selected_task.task.instance_id,
            "split": split,
            "initialized": False,
            "consistent": False,
            "bug_reproduced": False,
            "included_in_manifest": False,
            "error": None,
        }
        try:
            environment = WslSweSmithEnvironment(
                selected_task.task,
                environment_config,
                worker=client,
                tool_timeout_seconds=tool_timeout_seconds,
            )
            initialized += 1
            task_result["initialized"] = True
            metadata = environment.materialization_metadata()
            container_id = metadata["container_id"]
            if not isinstance(container_id, str) or not container_id:
                raise RuntimeError("Materialization did not return a container ID")
            container_ids.append(container_id)

            first = environment.evaluate()
            second = environment.evaluate()
            first_state = _test_state(first)
            second_state = _test_state(second)
            task_result["initial_test_1"] = first_state
            task_result["initial_test_2"] = second_state
            task_result["evaluator_1_ok"] = first.ok
            task_result["evaluator_2_ok"] = second.ok
            is_consistent = bool(
                first.ok
                and second.ok
                and first_state is not None
                and first_state == second_state
            )
            reproduces = bool(
                is_consistent
                and first_state is not None
                and isinstance(first_state.get("fail_to_pass_count"), int)
                and first_state["fail_to_pass_count"] > 0
            )
            task_result["consistent"] = is_consistent
            task_result["bug_reproduced"] = reproduces
            if is_consistent:
                consistent += 1
            if reproduces:
                bug_reproduced += 1

            if is_consistent and reproduces and first_state is not None:
                image_digests = metadata["image_digests"]
                image_digest = image_digests[0] if image_digests else metadata["image_id"]
                entry = TaskManifestEntry(
                    schema_version="2.0.0",
                    task_id=selected_task.task.instance_id,
                    split=split,
                    source="swe-smith",
                    repo_id=selected_task.task.source_repo_key,
                    base_commit=environment.base_commit,
                    docker_image_digest=image_digest,
                    issue_hash=selected_task.issue_hash,
                    parent_task_id=None,
                    environment_profile=environment_config.profile,
                    initial_test_status=str(first_state["status"]),
                    stable_repeats=2,
                ).to_dict()
                registry.validate("task_manifest", entry)
                entries.append(entry)
                task_result["included_in_manifest"] = True
                task_result["base_commit"] = environment.base_commit
                task_result["docker_image_digest"] = image_digest
        except Exception as exc:
            task_result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if environment is not None:
                cleanup_ok = environment.close()
                task_result["cleanup_ok"] = cleanup_ok
                task_result["cleanup_error"] = environment.cleanup_error
        task_results.append(task_result)

    requested = len(selected)
    expected = environment_config.counts[split]
    initialization_rate = initialized / requested
    consistency_rate = consistent / initialized if initialized else 0.0
    bug_reproduction_rate = bug_reproduced / initialized if initialized else 0.0
    container_ids_unique = len(container_ids) == len(set(container_ids))
    cleanup_complete = all(
        bool(item.get("cleanup_ok"))
        for item in task_results
        if item.get("initialized")
    )
    schema_completeness = len(entries) / bug_reproduced if bug_reproduced else 0.0
    passed = bool(
        requested == expected
        and initialization_rate >= 0.95
        and consistency_rate >= 0.95
        and bug_reproduction_rate >= 0.95
        and container_ids_unique
        and cleanup_complete
        and schema_completeness == 1.0
    )
    report = {
        "schema_version": "2.0.0",
        "report_type": "swesmith_stability_gate",
        "split": split,
        "requested_tasks": requested,
        "expected_tasks": expected,
        "selection_count_matches": requested == expected,
        "initialized_tasks": initialized,
        "consistent_tasks": consistent,
        "bug_reproduced_tasks": bug_reproduced,
        "manifest_tasks": len(entries),
        "initialization_rate": initialization_rate,
        "initial_test_consistency_rate": consistency_rate,
        "bug_reproduction_rate": bug_reproduction_rate,
        "schema_completeness": schema_completeness,
        "container_ids_unique": container_ids_unique,
        "cleanup_complete": cleanup_complete,
        "passed": passed,
        "tasks": task_results,
    }
    write_jsonl(manifest_output, entries)
    _write_json_exclusive(report_output, report)
    return report


def stabilize_external_candidates(
    inventory_path: Path,
    oracle_path: Path,
    candidates_output: Path,
    report_output: Path,
    environment_config: SweSmithEnvironmentConfig,
    *,
    worker: WorkerClient | None = None,
    tool_timeout_seconds: int = 600,
    minimum_eligible: int = EXTERNAL_SUBSET_COUNT,
) -> dict[str, Any]:
    """Materialize every provisional external task and emit freeze candidates.

    The output is deliberately a candidate report, not a final evaluation
    manifest. Only candidates that pass the later deterministic freeze are
    eligible for the final external-50 inventory.
    """

    if minimum_eligible <= 0:
        raise ValueError("minimum_eligible must be positive")
    for output in (candidates_output.resolve(), report_output.resolve()):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    selected = load_selected_swesmith_tasks(
        inventory_path, oracle_path, split="external"
    )
    client = worker or default_swesmith_worker(environment_config)
    registry = SchemaRegistry()
    candidates: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    initialized = 0
    consistent = 0
    executable = 0

    for selected_task in selected:
        task = selected_task.task
        environment: WslSweSmithEnvironment | None = None
        candidate: dict[str, Any] = {
            "schema_version": "2.0.0",
            "instance_id": task.instance_id,
            "repo_id": task.repo,
            "base_commit": "unavailable",
            "docker_image_digest": None,
            "issue_hash": selected_task.issue_hash,
            "environment_profile": environment_config.profile,
            "initialized": False,
            "stable_repeats": 0,
            "initial_test_status": None,
            "exclusion_reason": "environment initialization failed",
        }
        task_result: dict[str, Any] = {
            "instance_id": task.instance_id,
            "initialized": False,
            "consistent": False,
            "executable": False,
            "included_as_candidate": False,
            "error": None,
        }
        try:
            environment = WslSweSmithEnvironment(
                task,
                environment_config,
                worker=client,
                tool_timeout_seconds=tool_timeout_seconds,
            )
            initialized += 1
            candidate["initialized"] = True
            task_result["initialized"] = True
            metadata = environment.materialization_metadata()
            candidate["base_commit"] = environment.base_commit
            image_digests = metadata.get("image_digests", [])
            candidate["docker_image_digest"] = (
                image_digests[0] if image_digests else metadata.get("image_id")
            )
            first = environment.evaluate()
            second = environment.evaluate()
            first_state = _test_state(first)
            second_state = _test_state(second)
            task_result["initial_test_1"] = first_state
            task_result["initial_test_2"] = second_state
            task_result["evaluator_1_ok"] = first.ok
            task_result["evaluator_2_ok"] = second.ok
            if first_state is not None:
                candidate["initial_test_status"] = first_state.get("status")
            is_consistent = bool(
                first.ok
                and second.ok
                and first_state is not None
                and first_state == second_state
            )
            is_executable = bool(
                first_state is not None
                and second_state is not None
                and first_state.get("status") in {"passed", "failed"}
                and second_state.get("status") in {"passed", "failed"}
            )
            task_result["consistent"] = is_consistent
            task_result["executable"] = is_executable
            if is_consistent:
                consistent += 1
                candidate["stable_repeats"] = 2
            if is_executable:
                executable += 1
            reasons: list[str] = []
            if not is_consistent:
                reasons.append("initial test results were not stable")
            if not is_executable:
                reasons.append("initial test result is not executable")
            candidate["exclusion_reason"] = "; ".join(reasons) or None
            task_result["included_as_candidate"] = candidate["exclusion_reason"] is None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            candidate["exclusion_reason"] = message[-2000:]
            task_result["error"] = message[-2000:]
        finally:
            if environment is not None:
                task_result["cleanup_ok"] = environment.close()
                task_result["cleanup_error"] = environment.cleanup_error
        registry.validate("external_candidate", candidate)
        candidates.append(candidate)
        task_results.append(task_result)

    eligible_count = sum(
        bool(
            value["initialized"]
            and value["stable_repeats"] >= 2
            and value["docker_image_digest"]
            and value["initial_test_status"] in {"passed", "failed"}
            and value["exclusion_reason"] is None
        )
        for value in candidates
    )
    cleanup_complete = all(
        bool(item.get("cleanup_ok"))
        for item in task_results
        if item.get("initialized")
    )
    report = {
        "schema_version": "2.0.0",
        "report_type": "external_candidate_stability",
        "split": "external",
        "requested_tasks": len(selected),
        "initialized_tasks": initialized,
        "consistent_tasks": consistent,
        "executable_tasks": executable,
        "eligible_tasks": eligible_count,
        "minimum_final_external_tasks": EXTERNAL_SUBSET_COUNT,
        "minimum_eligible": minimum_eligible,
        "ready_for_external_freeze": eligible_count >= minimum_eligible,
        "cleanup_complete": cleanup_complete,
        "passed": bool(
            len(selected) > 0
            and len(candidates) == len(selected)
            and cleanup_complete
            and eligible_count >= minimum_eligible
        ),
        "candidates_output": str(candidates_output.resolve()),
        "tasks": task_results,
    }
    write_jsonl(candidates_output, candidates)
    _write_json_exclusive(report_output, report)
    return report
