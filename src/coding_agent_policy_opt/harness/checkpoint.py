from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .environment import AgentEnvironment
from ..hashing import sha256_json
from ..data.manifest import read_jsonl
from ..models import ActionType, AgentAction, TaskSpec
from ..schema import SchemaRegistry
from .tools import dispatch_action


class CheckpointMaterializer(Protocol):
    def materialize(
        self, task_id: str, **kwargs: Any
    ) -> tuple[TaskSpec, AgentEnvironment, Callable[[], None]]: ...


def read_recovery_state(run_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    reference = checkpoint.get("recovery_state_ref")
    if reference is None:
        return checkpoint.get("recovery_state")
    if not isinstance(reference, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", reference):
        raise ValueError("Invalid recovery state reference")
    path = run_dir / "recovery" / (reference.removeprefix("sha256:") + ".json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if sha256_json(value) != reference:
        raise ValueError("Recovery state checksum mismatch")
    return value


@dataclass(frozen=True)
class CheckpointRecoveryReport:
    run_id: str
    episode_id: str
    task_id: str
    checkpoint_step_index: int
    replayed_steps: int
    skipped_steps: int
    state_hash_matches: bool
    diff_hash_matches: bool
    passed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0.0",
            "report_type": "checkpoint_recovery",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "checkpoint_step_index": self.checkpoint_step_index,
            "replayed_steps": self.replayed_steps,
            "skipped_steps": self.skipped_steps,
            "state_hash_matches": self.state_hash_matches,
            "diff_hash_matches": self.diff_hash_matches,
            "passed": self.passed,
            "error": self.error,
        }


def _combined_state_hash(
    environment: AgentEnvironment,
    harness_config: dict[str, Any],
    steps: list[dict[str, Any]],
    *, budget_used: dict[str, Any] | None = None, legacy: bool = False,
) -> str:
    tokens_used = sum(
        int(step["input_tokens"]) + int(step["output_tokens"])
        for step in steps
    )
    test_calls = sum(
        step.get("action", {}).get("type") == ActionType.RUN_TESTS.value
        and step.get("tool_result", {}).get("error_code") != "BUDGET_EXHAUSTED"
        and step.get("tool_result", {}).get("error_code")
        not in {"TOOL_NOT_AVAILABLE", "VERIFICATION_REQUIRED", "RECOVERY_REQUIRED"}
        for step in steps
    )
    used_steps = len(steps)
    if budget_used is not None:
        tokens_used = int(budget_used["input_tokens"]) + int(budget_used["output_tokens"])
        test_calls = int(budget_used["test_calls"])
        used_steps = int(budget_used["steps"])
    payload = getattr(environment, "durable_state_payload", environment.state_payload)()
    if legacy:
        payload = environment.state_payload()
        payload.pop("workspace_version", None)
        payload.pop("recovery", None)
    return sha256_json(
        {
            "environment": payload,
            "budget": {
                "remaining_steps": max(
                    int(harness_config["max_tool_steps"]) - used_steps, 0
                ),
                "remaining_tokens": max(
                    int(harness_config["max_total_tokens"]) - tokens_used, 0
                ),
                "remaining_test_calls": max(
                    int(harness_config["max_test_calls"]) - test_calls, 0
                ),
            },
        }
    )


def replay_latest_checkpoint(
    run_dir: Path,
    materializer: CheckpointMaterializer,
) -> dict[str, Any]:
    """Rebuild the latest environment state and verify its hashes.

    The method never mutates the source run. It materializes a fresh task
    environment, replays only actions that the source Harness executed, and
    closes the fresh environment before returning the audit report.
    """

    resolved = run_dir.resolve()
    registry = SchemaRegistry()
    metadata = json.loads((resolved / "run.json").read_text(encoding="utf-8"))
    checkpoints = read_jsonl(resolved / "checkpoints.jsonl")
    for value in checkpoints:
        registry.validate("checkpoint_record", value)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in run: {resolved}")
    latest = checkpoints[-1]
    if latest["run_id"] != metadata.get("run_id"):
        raise ValueError("Checkpoint run_id differs from run metadata")
    if latest["task_id"] != metadata.get("task_id"):
        raise ValueError("Checkpoint task_id differs from run metadata")
    journal = read_jsonl(resolved / "steps.jsonl")
    steps = sorted(
        [
            value
            for value in journal
            if value["episode_id"] == latest["episode_id"]
            and value["step_index"] <= latest["step_index"]
        ],
        key=lambda value: value["step_index"],
    )
    if [step["step_index"] for step in steps] != list(
        range(latest["next_step_index"])
    ):
        raise ValueError("Checkpoint does not have a contiguous source step prefix")
    if steps[-1].get("tool_result", {}).get("observation_hash") != latest["observation_hash"]:
        raise ValueError("Checkpoint observation hash differs from source step")

    # A resumed attempt starts at local step zero; rebuilding from the base
    # requires the earlier attempts as well, in durable journal order.
    end = next(index for index, value in enumerate(journal)
               if value["episode_id"] == latest["episode_id"] and value["step_index"] == latest["step_index"])
    replay_steps = journal[:end + 1]

    task, environment, cleanup = materializer.materialize(latest["task_id"])
    replayed = 0
    skipped = 0
    state_matches = False
    diff_matches = False
    error: str | None = None
    try:
        if task.base_commit != metadata["base_commit"]:
            raise ValueError("Fresh checkpoint environment has a different base commit")
        initial = environment.evaluate()
        if not initial.ok:
            raise ValueError("Fresh checkpoint environment failed initial evaluation")
        previous_episode = replay_steps[0]["episode_id"]
        for step in replay_steps:
            if step["episode_id"] != previous_episode:
                reset_reads = getattr(environment, "reset_read_evidence", None)
                if callable(reset_reads):
                    reset_reads()
                initial = environment.evaluate()
                if not initial.ok:
                    raise ValueError("Fresh resumed evaluation failed")
                previous_episode = step["episode_id"]
            raw_action = step.get("action", {})
            error_code = step.get("tool_result", {}).get("error_code")
            if raw_action.get("type") == "invalid" or error_code in {
                "SCHEMA_VALIDATION_FAILED",
                "BUDGET_EXHAUSTED",
                "TOOL_NOT_AVAILABLE",
                "VERIFICATION_REQUIRED",
                "RECOVERY_REQUIRED",
            }:
                skipped += 1
                continue
            dispatch_action(environment, AgentAction.from_dict(raw_action))
            replayed += 1
        actual_state_hash = _combined_state_hash(
            environment, metadata["config"]["harness"], replay_steps,
            budget_used=latest.get("budget_used"), legacy=latest.get("state_contract") != "workspace-v1",
        )
        state_matches = actual_state_hash == latest["state_hash_after"]
        diff_matches = environment.diff_hash() == latest["diff_hash"]
        if not state_matches:
            raise ValueError("Replayed checkpoint state hash differs from source")
        if not diff_matches:
            raise ValueError("Replayed checkpoint diff hash differs from source")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            cleanup()
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            error = (
                f"{error}; cleanup failed: {cleanup_error}"
                if error is not None
                else f"cleanup failed: {cleanup_error}"
            )

    report = CheckpointRecoveryReport(
        run_id=metadata["run_id"],
        episode_id=latest["episode_id"],
        task_id=latest["task_id"],
        checkpoint_step_index=latest["step_index"],
        replayed_steps=replayed,
        skipped_steps=skipped,
        state_hash_matches=state_matches,
        diff_hash_matches=diff_matches,
        passed=error is None,
        error=error,
    )
    return report.to_dict()
