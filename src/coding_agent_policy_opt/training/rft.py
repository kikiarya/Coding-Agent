from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..hashing import sha256_json
from ..data.manifest import read_jsonl, write_jsonl
from ..schema import SchemaRegistry
from ..trajectory.samples import build_success_samples


RFT_SELECTION_RULE = (
    "environment_valid_and_resolved",
    "all_patch_groups_verified",
    "no_invalid_actions",
    "duplicate_rate_at_most_0_10",
    "one_best_rollout_per_task",
    "recovery_desc",
    "duplicate_actions_asc",
    "invalid_actions_asc",
    "tool_steps_asc",
    "total_tokens_asc",
    "run_dir_lexicographic",
)


def _load_single_episode(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = run_dir.resolve()
    episodes = read_jsonl(resolved / "episodes.jsonl")
    if len(episodes) != 1:
        raise ValueError(
            f"RFT candidate must contain exactly one episode: {resolved}"
        )
    steps = read_jsonl(resolved / "steps.jsonl")
    registry = SchemaRegistry()
    registry.validate("episode_record", episodes[0])
    episode = episodes[0]
    for field in ("initial_evaluator_artifact", "evaluator_artifact"):
        relative_artifact = episode[field]
        artifact = (resolved / relative_artifact).resolve()
        try:
            artifact.relative_to(resolved)
        except ValueError as exc:
            raise ValueError(
                f"RFT evaluator artifact escapes the run directory: {relative_artifact}"
            ) from exc
        if not artifact.is_file():
            raise ValueError(f"RFT evaluator artifact is missing: {artifact}")
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"RFT evaluator artifact is unreadable: {artifact}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"RFT evaluator artifact must be an object: {artifact}")
        if field == "evaluator_artifact" and payload.get("ok") is not True:
            raise ValueError(f"RFT final evaluator did not pass: {artifact}")
    for step in steps:
        registry.validate("step_record", step)
    if len(steps) != episodes[0]["total_steps"]:
        raise ValueError(
            f"RFT step count mismatch for {resolved}: "
            f"episode={episodes[0]['total_steps']} records={len(steps)}"
        )
    return episode, sorted(steps, key=lambda value: value["step_index"])


def _duplicate_rate(episode: dict[str, Any]) -> float:
    actions = max(
        int(episode["total_steps"])
        - int(episode["invalid_actions"]),
        1,
    )
    return int(episode["duplicate_actions"]) / actions


def _eligibility(
    episode: dict[str, Any],
    steps: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    if episode["split"] != "train":
        return False, "split_not_train"
    if episode["environment_valid"] is not True:
        return False, "environment_invalid"
    if episode["resolved"] is not True:
        return False, "not_resolved"
    if int(episode["patch_groups"]) != int(episode["verified_patch_groups"]):
        return False, "unverified_patch_group"
    if int(episode["invalid_actions"]) != 0:
        return False, "invalid_action"
    if _duplicate_rate(episode) > 0.10:
        return False, "duplicate_rate_above_0_10"
    labels = {
        label
        for step in steps
        for label in step.get("automatic_labels", [])
    }
    if "PREMATURE_STOP" in labels:
        return False, "premature_stop"
    return True, None


def _ranking_key(episode: dict[str, Any], run_dir: Path) -> tuple[Any, ...]:
    return (
        -int(episode["recovered_episodes"] > 0),
        int(episode["duplicate_actions"]),
        int(episode["invalid_actions"]),
        int(episode["total_steps"]),
        int(episode["total_input_tokens"]) + int(episode["total_output_tokens"]),
        run_dir.resolve().as_posix(),
    )


def select_rft_rollouts(run_dirs: Iterable[Path]) -> dict[str, Any]:
    """Filter sampled rollouts and select one verified trajectory per train task.

    The terminal evaluator and the recorded step labels are the only acceptance
    signals. This function deliberately does not ask an LLM judge to relabel a
    failed run, so an RFT dataset cannot silently turn an unresolved trajectory
    into a positive target.
    """

    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_run_dirs: set[Path] = set()
    for unresolved in run_dirs:
        run_dir = unresolved.resolve()
        if run_dir in seen_run_dirs:
            raise ValueError(f"Duplicate RFT run directory: {run_dir}")
        seen_run_dirs.add(run_dir)
        episode, steps = _load_single_episode(run_dir)
        eligible, reason = _eligibility(episode, steps)
        entry = {
            "run_dir": str(run_dir),
            "task_id": episode["task_id"],
            "episode_id": episode["episode_id"],
            "resolved": episode["resolved"],
            "environment_valid": episode["environment_valid"],
            "duplicate_actions": episode["duplicate_actions"],
            "invalid_actions": episode["invalid_actions"],
            "total_steps": episode["total_steps"],
            "total_tokens": episode["total_input_tokens"] + episode["total_output_tokens"],
            "recovered": episode["recovered_episodes"] > 0,
            "eligible": eligible,
        }
        if eligible:
            entry["ranking_key"] = list(_ranking_key(episode, run_dir))
            candidates.append(entry)
        else:
            entry["reason"] = reason
            excluded.append(entry)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in candidates:
        by_task.setdefault(entry["task_id"], []).append(entry)
    selected: list[dict[str, Any]] = []
    duplicate_task_candidates: list[dict[str, Any]] = []
    for task_id, task_candidates in sorted(by_task.items()):
        ranked = sorted(
            task_candidates,
            key=lambda entry: tuple(entry["ranking_key"]),
        )
        selected.append(ranked[0])
        duplicate_task_candidates.extend(ranked[1:])
    for entry in duplicate_task_candidates:
        excluded.append(entry | {"eligible": False, "reason": "non_best_rollout_for_task"})

    selected.sort(key=lambda entry: (entry["task_id"], entry["run_dir"]))
    report = {
        "schema_version": "2.0.0",
        "report_type": "rft_rollout_selection",
        "selection_split": "train",
        "selection_rule": list(RFT_SELECTION_RULE),
        "candidate_count": len(seen_run_dirs),
        "eligible_rollout_count": len(candidates),
        "selected_task_count": len(selected),
        "selected": selected,
        "excluded": excluded,
    }
    report["selection_hash"] = sha256_json(report)
    return report


def build_rft_dataset(
    run_dirs: Iterable[Path],
    output_path: Path,
    *,
    count: int = 600,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a reviewed-later D_success dataset from selected RFT rollouts."""

    if count <= 0:
        raise ValueError("RFT dataset count must be positive")
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RFT dataset: {output}")
    report = select_rft_rollouts(run_dirs)
    if not report["selected"]:
        raise ValueError("RFT selection produced no eligible rollout")

    samples: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for selected in report["selected"]:
        selected_samples = build_success_samples(Path(selected["run_dir"]))
        if not selected_samples:
            raise ValueError(
                f"Selected RFT rollout produced no decision samples: {selected['run_dir']}"
            )
        for sample in selected_samples:
            value = sample.to_dict()
            if value["sample_id"] in seen_sample_ids:
                raise ValueError(f"Duplicate RFT sample_id: {value['sample_id']}")
            seen_sample_ids.add(value["sample_id"])
            samples.append(value)

    if len(samples) < count:
        raise ValueError(
            f"RFT D_success requires {count} samples after selection; found {len(samples)}"
        )
    selected_samples = sorted(
        samples,
        key=lambda value: sha256_json(
            {"seed": seed, "sample_id": value["sample_id"]}
        ),
    )[:count]
    write_jsonl(output, selected_samples)
    return {
        **report,
        "report_type": "rft_dataset_build",
        "candidate_sample_count": len(samples),
        "sample_count": len(selected_samples),
        "sample_selection_seed": seed,
        "output": str(output),
        "dataset": "D_success",
    }


def load_rft_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("report_type") not in {
        "rft_rollout_selection",
        "rft_dataset_build",
    }:
        raise ValueError(f"Expected an RFT selection report: {path.resolve()}")
    return value
