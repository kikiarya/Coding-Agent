from __future__ import annotations

import math
import random
from typing import Any

from .metrics import TaskMetrics


Component = tuple[float, float]


def exact_mcnemar_test(baseline_only: int, candidate_only: int) -> float:
    if baseline_only < 0 or candidate_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _component(task: TaskMetrics, metric: str) -> Component:
    if metric == "environment_validity_rate":
        return float(task.environment_valid), 1.0
    if not task.environment_valid:
        return 0.0, 0.0
    if metric == "resolve_rate":
        return float(task.resolved), 1.0
    if metric == "localization_at_5":
        return (
            float(task.localization_hit_at_5 is True),
            float(task.localization_eligible),
        )
    if metric == "post_patch_verification_rate":
        return float(task.verified_patch_groups), float(task.patch_groups)
    if metric == "premature_stop_rate":
        return float(task.premature_success_finish), float(task.model_finish_success)
    if metric == "episode_recovery_rate":
        return float(task.recovered_episode), float(task.recovery_eligible_episode)
    if metric == "duplicate_action_rate":
        return float(task.duplicate_actions), float(task.total_tool_actions)
    if metric == "invalid_action_rate":
        return float(task.invalid_actions), float(task.total_steps)
    if metric == "give_up_rate":
        return float(task.give_up), 1.0
    if metric == "budget_exhaustion_rate":
        return float(task.budget_exhausted), 1.0
    if metric == "average_steps":
        return float(task.total_steps), 1.0
    if metric == "average_total_tokens":
        return float(task.total_tokens), 1.0
    if metric == "average_test_calls":
        return float(task.test_calls), 1.0
    if metric == "average_tool_calls":
        return float(task.tool_calls), 1.0
    if metric == "average_wall_time_ms":
        return float(task.wall_time_ms), 1.0
    raise ValueError(f"Unsupported metric: {metric}")


def _ratio(tasks: list[TaskMetrics], metric: str) -> float | None:
    components = [_component(task, metric) for task in tasks]
    denominator = sum(item[1] for item in components)
    if denominator == 0:
        return None
    return sum(item[0] for item in components) / denominator


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_difference(
    baseline: list[TaskMetrics],
    candidate: list[TaskMetrics],
    metric: str,
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    baseline_by_task = {task.task_id: task for task in baseline}
    candidate_by_task = {task.task_id: task for task in candidate}
    if len(baseline_by_task) != len(baseline) or len(candidate_by_task) != len(candidate):
        raise ValueError("each side must contain unique task_id values")
    if baseline_by_task.keys() != candidate_by_task.keys():
        raise ValueError("paired comparison requires identical task_id sets")
    task_ids = sorted(baseline_by_task)
    if not task_ids:
        raise ValueError("paired comparison requires at least one task")
    for task_id in task_ids:
        if baseline_by_task[task_id].repo_id != candidate_by_task[task_id].repo_id:
            raise ValueError(f"repo_id mismatch for task {task_id}")

    if metric != "environment_validity_rate":
        task_ids = [
            task_id
            for task_id in task_ids
            if baseline_by_task[task_id].environment_valid
            and candidate_by_task[task_id].environment_valid
        ]
        if not task_ids:
            return {
                "metric": metric,
                "paired_task_count": 0,
                "baseline": None,
                "candidate": None,
                "difference_candidate_minus_baseline": None,
                "paired_bootstrap_samples_requested": samples,
                "paired_bootstrap_samples_valid": 0,
                "confidence_interval_95": None,
                "seed": seed,
            }

    baseline_tasks = [baseline_by_task[task_id] for task_id in task_ids]
    candidate_tasks = [candidate_by_task[task_id] for task_id in task_ids]
    baseline_value = _ratio(baseline_tasks, metric)
    candidate_value = _ratio(candidate_tasks, metric)
    point_difference = (
        candidate_value - baseline_value
        if baseline_value is not None and candidate_value is not None
        else None
    )

    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(task_ids)) for _ in task_ids]
        sampled_baseline = [baseline_tasks[index] for index in indices]
        sampled_candidate = [candidate_tasks[index] for index in indices]
        sampled_baseline_value = _ratio(sampled_baseline, metric)
        sampled_candidate_value = _ratio(sampled_candidate, metric)
        if sampled_baseline_value is None or sampled_candidate_value is None:
            continue
        differences.append(sampled_candidate_value - sampled_baseline_value)

    differences.sort()
    ci = (
        [_percentile(differences, 0.025), _percentile(differences, 0.975)]
        if differences
        else None
    )
    return {
        "metric": metric,
        "paired_task_count": len(task_ids),
        "baseline": baseline_value,
        "candidate": candidate_value,
        "difference_candidate_minus_baseline": point_difference,
        "paired_bootstrap_samples_requested": samples,
        "paired_bootstrap_samples_valid": len(differences),
        "confidence_interval_95": ci,
        "seed": seed,
    }


def compare_models(
    baseline: list[TaskMetrics],
    candidate: list[TaskMetrics],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    metrics = [
        "environment_validity_rate",
        "resolve_rate",
        "localization_at_5",
        "post_patch_verification_rate",
        "premature_stop_rate",
        "episode_recovery_rate",
        "duplicate_action_rate",
        "invalid_action_rate",
        "give_up_rate",
        "budget_exhaustion_rate",
        "average_steps",
        "average_total_tokens",
        "average_test_calls",
        "average_tool_calls",
        "average_wall_time_ms",
    ]
    baseline_by_task = {task.task_id: task for task in baseline}
    candidate_by_task = {task.task_id: task for task in candidate}
    if baseline_by_task.keys() != candidate_by_task.keys():
        raise ValueError("paired comparison requires identical task_id sets")
    discordant_baseline_only = 0
    discordant_candidate_only = 0
    for task_id in baseline_by_task:
        left = baseline_by_task[task_id]
        right = candidate_by_task[task_id]
        if not left.environment_valid or not right.environment_valid:
            continue
        if left.resolved and not right.resolved:
            discordant_baseline_only += 1
        elif right.resolved and not left.resolved:
            discordant_candidate_only += 1

    return {
        "task_count": len(baseline_by_task),
        "resolve_exact_mcnemar": {
            "baseline_only_resolved": discordant_baseline_only,
            "candidate_only_resolved": discordant_candidate_only,
            "two_sided_p_value": exact_mcnemar_test(
                discordant_baseline_only, discordant_candidate_only
            ),
        },
        "paired_bootstrap": {
            metric: paired_bootstrap_difference(
                baseline, candidate, metric, samples=samples, seed=seed
            )
            for metric in metrics
        },
    }
