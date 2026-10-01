from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..data.manifest import read_jsonl
from .metrics import REGISTERED_ACTIONS, _normalize_path


def _observation(run_dir: Path, observation_hash: str) -> dict[str, Any]:
    filename = observation_hash.removeprefix("sha256:") + ".json"
    value = json.loads((run_dir / "observations" / filename).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("observation must be an object")
    return value


def _relevant(step: dict[str, Any], run_dir: Path, gold_files: set[str]) -> bool:
    action = step.get("action", {})
    action_type = action.get("type")
    args = action.get("args", {})
    if action_type == "read_file":
        return _normalize_path(args.get("path", "")) in gold_files
    if action_type == "search_code" and step.get("tool_result", {}).get("ok") is True:
        observation = _observation(
            run_dir, step["tool_result"]["observation_hash"]
        )
        return any(
            isinstance(match, dict)
            and _normalize_path(match.get("path", "")) in gold_files
            for match in observation.get("data", {}).get("matches", [])
        )
    return False


def analyze_exploration(
    run_dirs: Iterable[Path],
    *,
    gold_by_task: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Analyze repository exploration traces for SWE-Explore-style slices.

    This is a local trace analysis, not a reimplementation of the official
    SWE-Explore benchmark. It intentionally reports localization and context
    mechanisms separately from final Resolve.
    """

    gold = gold_by_task or {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unresolved in run_dirs:
        run_dir = unresolved.resolve()
        for episode in read_jsonl(run_dir / "episodes.jsonl"):
            task_id = episode["task_id"]
            if task_id in seen:
                raise ValueError(f"Duplicate exploration task: {task_id}")
            seen.add(task_id)
            steps = sorted(
                [
                    step
                    for step in read_jsonl(run_dir / "steps.jsonl")
                    if step["episode_id"] == episode["episode_id"]
                ],
                key=lambda value: value["step_index"],
            )
            gold_files = gold.get(task_id)
            relevant_indices = [
                index
                for index, step in enumerate(steps)
                if gold_files is not None and _relevant(step, run_dir, gold_files)
            ]
            tool_steps = [
                step
                for step in steps
                if step.get("action", {}).get("type") in REGISTERED_ACTIONS
            ]
            strategy_counts = Counter(
                step.get("harness_context_strategy", "unknown") for step in steps
            )
            rows.append(
                {
                    "task_id": task_id,
                    "repo_id": episode["repo_id"],
                    "resolved": bool(episode["resolved"]),
                    "localization_eligible": gold_files is not None,
                    "localization_hit_at_5": bool(
                        relevant_indices and relevant_indices[0] < 5
                    )
                    if gold_files is not None
                    else None,
                    "first_relevant_action_index": (
                        relevant_indices[0] + 1 if relevant_indices else None
                    ),
                    "first_relevant_tool": (
                        steps[relevant_indices[0]]["action"]["type"]
                        if relevant_indices
                        else None
                    ),
                    "search_calls": sum(
                        step.get("action", {}).get("type") == "search_code"
                        for step in tool_steps
                    ),
                    "read_calls": sum(
                        step.get("action", {}).get("type") == "read_file"
                        for step in tool_steps
                    ),
                    "tool_calls": sum(
                        step.get("action", {}).get("type") != "finish"
                        for step in tool_steps
                    ),
                    "context_trimmed": any(
                        "HARNESS_CONTEXT_TRIMMED"
                        in step.get("automatic_labels", [])
                        for step in steps
                    ),
                    "context_strategy_counts": dict(strategy_counts),
                    "harness_gate_count": sum(
                        "HARNESS_TOOL_GATED" in step.get("automatic_labels", [])
                        for step in steps
                    ),
                }
            )

    eligible = [row for row in rows if row["localization_eligible"]]
    first_relevant = [
        row["first_relevant_action_index"]
        for row in eligible
        if row["first_relevant_action_index"] is not None
    ]
    return {
        "schema_version": "2.0.0",
        "report_type": "exploration_analysis",
        "analysis_name": "swe_explore_style_trace_analysis",
        "method_note": (
            "Trace-based localization/context analysis inspired by SWE-Explore; "
            "not the official SWE-Explore benchmark score."
        ),
        "task_count": len(rows),
        "localization_eligible_count": len(eligible),
        "localization_at_5": (
            sum(row["localization_hit_at_5"] is True for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "mean_first_relevant_action_index": (
            statistics.mean(first_relevant) if first_relevant else None
        ),
        "mean_search_calls": statistics.mean(row["search_calls"] for row in rows) if rows else None,
        "mean_read_calls": statistics.mean(row["read_calls"] for row in rows) if rows else None,
        "context_trim_rate": (
            sum(row["context_trimmed"] for row in rows) / len(rows) if rows else None
        ),
        "harness_gate_count": sum(row["harness_gate_count"] for row in rows),
        "context_strategy_counts": dict(
            Counter(
                strategy
                for row in rows
                for strategy in row["context_strategy_counts"]
            )
        ),
        "tasks": sorted(rows, key=lambda value: value["task_id"]),
    }
