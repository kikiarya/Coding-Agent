from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..data.manifest import read_jsonl
from ..schema import SchemaRegistry


REGISTERED_ACTIONS = {
    "search_code",
    "read_file",
    "apply_patch",
    "run_tests",
    "inspect_diff",
    "finish",
}
BUDGET_TERMINATIONS = {
    "STEP_BUDGET_EXHAUSTED",
    "TOKEN_BUDGET_EXHAUSTED",
    "TEST_BUDGET_EXHAUSTED",
}


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


@dataclass(frozen=True)
class TaskMetrics:
    task_id: str
    repo_id: str
    model_id: str
    adapter_id: str | None
    experiment_config_hash: str
    environment_valid: bool
    resolved: bool
    localization_eligible: bool
    localization_hit_at_5: bool | None
    patch_groups: int
    verified_patch_groups: int
    model_finish_success: bool
    premature_success_finish: bool
    recovery_opportunities: int
    recovered_episodes: int
    recovery_eligible_episode: bool
    recovered_episode: bool
    mean_steps_to_recovery: float | None
    duplicate_actions: int
    invalid_actions: int
    total_tool_actions: int
    total_steps: int
    input_tokens: int
    output_tokens: int
    test_calls: int
    wall_time_ms: int
    give_up: bool
    budget_exhausted: bool
    harness_policy_variant: str = "static_v1"
    harness_interventions: int = 0
    tool_calls: int = 0
    test_passed: bool = False
    replan_count: int = 0
    checkpoint_recovery_success: bool | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"total_tokens": self.total_tokens}


def load_gold_files(path: Path) -> dict[str, set[str]]:
    registry = SchemaRegistry()
    result: dict[str, set[str]] = {}
    for line_number, value in enumerate(read_jsonl(path), start=1):
        registry.validate("gold_files", value)
        task_id = value["task_id"]
        if task_id in result:
            raise ValueError(f"line {line_number}: duplicate gold files for task {task_id}")
        result[task_id] = {_normalize_path(item) for item in value["gold_files"]}
    return result


def _load_observation(run_dir: Path, observation_hash: str) -> dict[str, Any]:
    name = observation_hash.removeprefix("sha256:")
    path = run_dir / "observations" / f"{name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Observation is not an object: {path}")
    return value


def _localization_hit(
    run_dir: Path,
    steps: list[dict[str, Any]],
    gold_files: set[str],
) -> bool:
    tool_steps = [step for step in steps if step.get("action", {}).get("type") in REGISTERED_ACTIONS]
    for step in tool_steps[:5]:
        action = step["action"]
        if action["type"] == "read_file":
            path = action.get("args", {}).get("path")
            if isinstance(path, str) and _normalize_path(path) in gold_files:
                return True
        if action["type"] == "search_code" and step.get("tool_result", {}).get("ok") is True:
            observation = _load_observation(
                run_dir, step["tool_result"]["observation_hash"]
            )
            matches = observation.get("data", {}).get("matches", [])
            for match in matches:
                path = match.get("path") if isinstance(match, dict) else None
                if isinstance(path, str) and _normalize_path(path) in gold_files:
                    return True
    return False


def _steps_to_recovery(steps: list[dict[str, Any]]) -> list[int]:
    recovery_start: int | None = None
    deltas: list[int] = []
    for step in steps:
        labels = set(step.get("automatic_labels", []))
        if "RECOVERY_OPPORTUNITY" in labels and recovery_start is None:
            recovery_start = int(step["step_index"])
        if "RECOVERED" in labels and recovery_start is not None:
            deltas.append(int(step["step_index"]) - recovery_start)
            recovery_start = None
    return deltas


def collect_task_metrics(
    run_dirs: Iterable[Path],
    *,
    gold_by_task: dict[str, set[str]] | None = None,
) -> list[TaskMetrics]:
    gold = gold_by_task or {}
    result: list[TaskMetrics] = []
    seen_tasks: set[str] = set()
    for unresolved_run_dir in run_dirs:
        run_dir = unresolved_run_dir.resolve()
        run_metadata: dict[str, Any] = {}
        metadata_path = run_dir / "run.json"
        if metadata_path.is_file():
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metadata, dict):
                run_metadata = loaded_metadata
        policy_metadata = run_metadata.get("harness_policy", {})
        harness_policy_variant = (
            policy_metadata.get("variant", "static_v1")
            if isinstance(policy_metadata, dict)
            else "static_v1"
        )
        episodes = read_jsonl(run_dir / "episodes.jsonl")
        steps = read_jsonl(run_dir / "steps.jsonl")
        steps_by_episode: dict[str, list[dict[str, Any]]] = {}
        for step in steps:
            steps_by_episode.setdefault(step["episode_id"], []).append(step)

        for episode in episodes:
            task_id = episode["task_id"]
            if task_id in seen_tasks:
                raise ValueError(f"Duplicate evaluation task across runs: {task_id}")
            seen_tasks.add(task_id)
            episode_steps = sorted(
                steps_by_episode.get(episode["episode_id"], []),
                key=lambda item: item["step_index"],
            )
            if len(episode_steps) != episode["total_steps"]:
                raise ValueError(
                    f"Step count mismatch for {task_id}: "
                    f"episode={episode['total_steps']} records={len(episode_steps)}"
                )

            gold_files = gold.get(task_id)
            recovery_deltas = _steps_to_recovery(episode_steps)
            labels = {
                label
                for step in episode_steps
                for label in step.get("automatic_labels", [])
            }
            total_tool_actions = sum(
                step.get("action", {}).get("type") in REGISTERED_ACTIONS
                for step in episode_steps
            )
            tool_calls = sum(
                step.get("action", {}).get("type") in REGISTERED_ACTIONS
                and step.get("action", {}).get("type") != "finish"
                for step in episode_steps
            )
            harness_interventions = sum(
                any(
                    label in {
                        "HARNESS_TOOL_GATED",
                        "HARNESS_VERIFICATION_GUARD",
                        "HARNESS_RECOVERY_GUARD",
                    }
                    for label in step.get("automatic_labels", [])
                )
                for step in episode_steps
            )
            result.append(
                TaskMetrics(
                    task_id=task_id,
                    repo_id=episode["repo_id"],
                    model_id=episode["model_id"],
                    adapter_id=episode.get("adapter_id"),
                    experiment_config_hash=episode["experiment_config_hash"],
                    environment_valid=episode["environment_valid"],
                    resolved=episode["resolved"],
                    localization_eligible=gold_files is not None,
                    localization_hit_at_5=(
                        _localization_hit(run_dir, episode_steps, gold_files)
                        if gold_files is not None
                        else None
                    ),
                    patch_groups=episode["patch_groups"],
                    verified_patch_groups=episode["verified_patch_groups"],
                    model_finish_success=(
                        episode["termination_reason"] == "MODEL_FINISH_SUCCESS"
                    ),
                    premature_success_finish="PREMATURE_STOP" in labels,
                    recovery_opportunities=episode["recovery_opportunities"],
                    recovered_episodes=episode["recovered_episodes"],
                    recovery_eligible_episode=episode["recovery_opportunities"] > 0,
                    recovered_episode=episode["recovered_episodes"] > 0,
                    mean_steps_to_recovery=(
                        statistics.mean(recovery_deltas) if recovery_deltas else None
                    ),
                    duplicate_actions=episode["duplicate_actions"],
                    invalid_actions=episode["invalid_actions"],
                    total_tool_actions=total_tool_actions,
                    total_steps=episode["total_steps"],
                    input_tokens=episode["total_input_tokens"],
                    output_tokens=episode["total_output_tokens"],
                    test_calls=episode["test_calls"],
                    wall_time_ms=episode["wall_time_ms"],
                    give_up=episode["termination_reason"] == "MODEL_GIVE_UP",
                    budget_exhausted=episode["termination_reason"] in BUDGET_TERMINATIONS,
                    harness_policy_variant=harness_policy_variant,
                    harness_interventions=harness_interventions,
                     tool_calls=tool_calls,
                     test_passed=(
                         episode.get("environment_valid") is True
                         and episode.get("best_fail_to_pass_count") == 0
                         and episode.get("termination_reason") != "ENVIRONMENT_ERROR"
                     ),
                     replan_count=sum(
                         "REPLAN" in step.get("automatic_labels", [])
                         for step in episode_steps
                     ),
                     checkpoint_recovery_success=(
                         bool(run_dir / "checkpoint_recovery.json")
                         if (run_dir / "checkpoint_recovery.json").is_file()
                         else None
                     ),
                 )
            )
    return sorted(result, key=lambda item: item.task_id)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _cost_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None}
    return {"mean": statistics.mean(values), "median": statistics.median(values)}


def aggregate_metrics(tasks: Iterable[TaskMetrics]) -> dict[str, Any]:
    task_list = list(tasks)
    valid = [task for task in task_list if task.environment_valid]
    resolved = [task for task in valid if task.resolved]
    localization = [task for task in valid if task.localization_eligible]
    recovery_steps = [
        task.mean_steps_to_recovery
        for task in valid
        if task.mean_steps_to_recovery is not None
    ]

    def costs(group: list[TaskMetrics]) -> dict[str, Any]:
        return {
            "steps": _cost_summary([task.total_steps for task in group]),
            "input_tokens": _cost_summary([task.input_tokens for task in group]),
            "output_tokens": _cost_summary([task.output_tokens for task in group]),
            "total_tokens": _cost_summary([task.total_tokens for task in group]),
            "test_calls": _cost_summary([task.test_calls for task in group]),
            "tool_calls": _cost_summary([task.tool_calls for task in group]),
            "wall_time_ms": _cost_summary([task.wall_time_ms for task in group]),
        }

    patch_groups = sum(task.patch_groups for task in valid)
    finish_success = sum(task.model_finish_success for task in valid)
    recovery_eligible_episodes = sum(task.recovery_eligible_episode for task in valid)
    tool_actions = sum(task.total_tool_actions for task in valid)
    decisions = sum(task.total_steps for task in valid)
    harness_interventions = sum(task.harness_interventions for task in valid)
    return {
        "task_count": len(task_list),
        "environment_valid_count": len(valid),
        "environment_validity_rate": _safe_ratio(len(valid), len(task_list)),
        "resolved_count": len(resolved),
        "resolve_rate": _safe_ratio(len(resolved), len(valid)),
        "test_pass_rate": _safe_ratio(
            sum(task.test_passed for task in valid), len(valid)
        ),
        "localization_eligible_count": len(localization),
        "localization_at_5": _safe_ratio(
            sum(task.localization_hit_at_5 is True for task in localization),
            len(localization),
        ),
        "patch_groups": patch_groups,
        "post_patch_verification_rate": _safe_ratio(
            sum(task.verified_patch_groups for task in valid), patch_groups
        ),
        "model_finish_success_count": finish_success,
        "premature_stop_rate": _safe_ratio(
            sum(task.premature_success_finish for task in valid), finish_success
        ),
        "recovery_eligible_episodes": recovery_eligible_episodes,
        "recovery_opportunity_events": sum(
            task.recovery_opportunities for task in valid
        ),
        "episode_recovery_rate": _safe_ratio(
            sum(task.recovered_episode for task in valid), recovery_eligible_episodes
        ),
        "mean_steps_to_recovery": (
            statistics.mean(recovery_steps) if recovery_steps else None
        ),
        "replan_count": sum(task.replan_count for task in valid),
        "checkpoint_recovery_success_rate": _safe_ratio(
            sum(task.checkpoint_recovery_success is True for task in valid),
            sum(task.checkpoint_recovery_success is not None for task in valid),
        ),
        "duplicate_action_rate": _safe_ratio(
            sum(task.duplicate_actions for task in valid), tool_actions
        ),
        "invalid_action_rate": _safe_ratio(
            sum(task.invalid_actions for task in valid), decisions
        ),
        "harness_policy_variants": sorted(
            {task.harness_policy_variant for task in task_list}
        ),
        "harness_intervention_count": harness_interventions,
        "harness_intervention_rate": _safe_ratio(harness_interventions, decisions),
        "give_up_rate": _safe_ratio(sum(task.give_up for task in valid), len(valid)),
        "budget_exhaustion_rate": _safe_ratio(
            sum(task.budget_exhausted for task in valid), len(valid)
        ),
        "costs": {"all_valid": costs(valid), "resolved_only": costs(resolved)},
        "tasks": [task.to_dict() for task in task_list],
    }
