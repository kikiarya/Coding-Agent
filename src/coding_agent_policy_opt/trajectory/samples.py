from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..hashing import canonical_json, sha256_json
from ..data.manifest import read_jsonl, write_jsonl
from ..models import DecisionSample, SCHEMA_VERSION
from ..schema import SchemaRegistry


SYSTEM_PROMPT = (
    "You are a repository coding agent. Use exactly one registered JSON tool action per turn. "
    "Work only inside the task repository and verify code changes before finishing."
)


def _category_for_step(step: dict[str, Any]) -> str:
    labels = set(step.get("automatic_labels", []))
    action_type = step.get("action", {}).get("type")
    if "RECOVERED" in labels:
        return "recovery"
    if action_type == "run_tests":
        return "verification"
    if action_type == "finish" or "DUPLICATE_ACTION" in labels:
        return "termination_efficiency"
    return "planning_tool"


def build_success_samples(run_dir: Path) -> list[DecisionSample]:
    run_dir = run_dir.resolve()
    run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    steps = read_jsonl(run_dir / "steps.jsonl")
    resolved_episode_ids = {
        episode["episode_id"]
        for episode in episodes
        if episode.get("resolved") is True and episode.get("environment_valid") is True
    }
    issue = run_metadata.get("issue")
    if not isinstance(issue, str) or not issue:
        raise ValueError("run.json is missing the task issue; rerun with the current harness")

    observations_dir = run_dir / "observations"
    history_by_episode: dict[str, list[dict[str, str]]] = {}
    samples: list[DecisionSample] = []
    registry = SchemaRegistry()
    for step in sorted(steps, key=lambda value: (value["episode_id"], value["step_index"])):
        episode_id = step["episode_id"]
        if episode_id not in resolved_episode_ids:
            continue
        action = step.get("action", {})
        history = history_by_episode.setdefault(
            episode_id,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": issue},
            ],
        )
        observation_hash = step["tool_result"]["observation_hash"].removeprefix("sha256:")
        observation_path = observations_dir / f"{observation_hash}.json"
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        is_eligible_target = bool(
            action.get("type") != "invalid"
            and step.get("tool_result", {}).get("ok") is True
        )
        if is_eligible_target:
            target = {"intent": step.get("intent", ""), "action": action}
            registry.validate("agent_action", target)
            provenance = {
                "run_id": step["run_id"],
                "episode_id": episode_id,
                "step_index": step["step_index"],
                "task_id": step["task_id"],
                "action_hash": step["normalized_action_hash"],
                "observation_hash": step["tool_result"]["observation_hash"],
            }
            sample_id = sha256_json(provenance).removeprefix("sha256:")[:24]
            sample = DecisionSample(
                schema_version=SCHEMA_VERSION,
                sample_id=f"sample_{sample_id}",
                source_run_id=step["run_id"],
                source_episode_id=episode_id,
                source_step_index=step["step_index"],
                task_id=step["task_id"],
                split=step["split"],
                repo_id=step["repo_id"],
                dataset="D_success",
                category=_category_for_step(step),
                messages=[dict(message) for message in history],
                target=target,
                rejected_action=None,
                error_labels=[],
                correction_method=(
                    "recovery_success"
                    if "RECOVERED" in step.get("automatic_labels", [])
                    else "natural_success"
                ),
                sandbox_validated=True,
                review_status="pending",
                provenance_hash=sha256_json(provenance),
            )
            registry.validate("decision_sample", sample.to_dict())
            samples.append(sample)
            assistant_payload: Any = target
        else:
            assistant_payload = action.get("args", {}).get("raw", action)
        history.append(
            {"role": "assistant", "content": canonical_json(assistant_payload)}
        )
        history.append({"role": "tool", "content": canonical_json(observation)})
    return samples


def write_success_samples(run_dir: Path, output_path: Path) -> int:
    samples = build_success_samples(run_dir)
    write_jsonl(output_path, (sample.to_dict() for sample in samples))
    return len(samples)
