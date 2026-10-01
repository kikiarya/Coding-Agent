from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Protocol

from .environment import AgentEnvironment
from ..hashing import sha256_json
from ..data.manifest import read_jsonl, write_jsonl
from ..models import ActionType, AgentAction, TaskSpec
from ..schema import SchemaRegistry
from .tools import dispatch_action


CORRECTED_METHODS = {"controlled_perturbation_repair", "manual_repair"}


class ReplayMaterializer(Protocol):
    def materialize(
        self, task_id: str, **kwargs: Any
    ) -> tuple[TaskSpec, AgentEnvironment, Any]: ...


def _combined_state_hash(
    environment: AgentEnvironment,
    run_metadata: dict[str, Any],
    prior_steps: list[dict[str, Any]],
) -> str:
    from .checkpoint import _combined_state_hash as checkpoint_hash
    return checkpoint_hash(
        environment, run_metadata["config"]["harness"], prior_steps,
        legacy=run_metadata.get("state_contract") != "workspace-v1",
    )


def _is_resolved(evaluator: Any) -> bool:
    test_state = evaluator.data.get("test_state") if evaluator.ok else None
    return bool(
        isinstance(test_state, dict)
        and test_state.get("status") == "passed"
        and test_state.get("fail_to_pass_count") == 0
        and test_state.get("pass_to_pass_failures") == 0
    )


def _validate_target_result(action: AgentAction, result: Any) -> None:
    if not result.ok:
        raise RuntimeError(
            f"Corrected target failed: {result.error_code}: {result.message}"
        )
    if action.type is ActionType.APPLY_PATCH and not result.state_changed:
        raise RuntimeError("Corrected apply_patch target did not change repository state")
    if action.type is ActionType.RUN_TESTS and not isinstance(
        result.data.get("test_state"), dict
    ):
        raise RuntimeError("Corrected run_tests target returned no parsed test state")


def replay_corrected_samples(
    dataset_path: Path,
    run_dirs: Iterable[Path],
    output_dataset_path: Path,
    ledger_path: Path,
    materializer: ReplayMaterializer,
) -> dict[str, Any]:
    for output in (output_dataset_path.resolve(), ledger_path.resolve()):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    registry = SchemaRegistry()
    samples = read_jsonl(dataset_path)
    runs: dict[str, Path] = {}
    for path in run_dirs:
        resolved = path.resolve()
        if resolved.name in runs:
            raise ValueError(f"Duplicate source run_id: {resolved.name}")
        runs[resolved.name] = resolved

    corrected = [
        sample for sample in samples if sample.get("correction_method") in CORRECTED_METHODS
    ]
    if not corrected:
        raise ValueError("Dataset contains no corrected samples to replay")
    ledgers: list[dict[str, Any]] = []
    validated_sample_ids: set[str] = set()

    for sample in corrected:
        registry.validate("decision_sample", sample)
        registry.validate("agent_action", sample["target"])
        action = AgentAction.from_dict(sample["target"]["action"])
        source_run = runs.get(sample["source_run_id"])
        entry: dict[str, Any] = {
            "schema_version": "2.0.0",
            "sample_id": sample["sample_id"],
            "provenance_hash": sample["provenance_hash"],
            "source_run_id": sample["source_run_id"],
            "source_episode_id": sample["source_episode_id"],
            "source_step_index": sample["source_step_index"],
            "task_id": sample["task_id"],
            "state_hash_expected": "sha256:" + "0" * 64,
            "state_hash_actual": None,
            "replayed_prior_actions": 0,
            "target_action_hash": action.normalized_hash,
            "target_tool": action.type.value,
            "target_ok": False,
            "state_changed": False,
            "evaluator_checked": False,
            "evaluator_resolved": None,
            "cleanup_ok": False,
            "passed": False,
            "error": None,
        }
        cleanup = None
        try:
            if source_run is None:
                raise FileNotFoundError(
                    f"No --run-dir matches source_run_id {sample['source_run_id']}"
                )
            run_metadata = json.loads(
                (source_run / "run.json").read_text(encoding="utf-8")
            )
            steps = [
                step
                for step in read_jsonl(source_run / "steps.jsonl")
                if step["episode_id"] == sample["source_episode_id"]
            ]
            steps.sort(key=lambda step: step["step_index"])
            source_step = next(
                (
                    step
                    for step in steps
                    if step["step_index"] == sample["source_step_index"]
                ),
                None,
            )
            if source_step is None:
                raise ValueError("Source step is absent from the source episode")
            for field in ("task_id", "repo_id"):
                if run_metadata.get(field) != sample[field]:
                    raise ValueError(f"Sample {field} differs from source run metadata")
            entry["state_hash_expected"] = source_step["state_hash_before"]
            task, environment, cleanup = materializer.materialize(sample["task_id"])
            if task.base_commit != run_metadata["base_commit"]:
                raise ValueError("Fresh replay base commit differs from source run")
            initial = environment.evaluate()
            if not initial.ok:
                raise RuntimeError("Fresh replay initial evaluator failed")

            prior_steps = [
                step
                for step in steps
                if step["step_index"] < sample["source_step_index"]
            ]
            for prior in prior_steps:
                raw_action = prior.get("action", {})
                if raw_action.get("type") == "invalid":
                    continue
                prior_action = AgentAction.from_dict(raw_action)
                if prior_action.type is ActionType.FINISH:
                    raise ValueError("Source episode finished before corrected sample state")
                dispatch_action(environment, prior_action)
                entry["replayed_prior_actions"] += 1

            actual_state_hash = _combined_state_hash(
                environment, run_metadata, prior_steps
            )
            entry["state_hash_actual"] = actual_state_hash
            if actual_state_hash != entry["state_hash_expected"]:
                raise RuntimeError("Replayed state hash differs from source state hash")

            target_result = dispatch_action(environment, action)
            entry["target_ok"] = bool(target_result.ok)
            entry["state_changed"] = bool(target_result.state_changed)
            _validate_target_result(action, target_result)
            if (
                action.type is ActionType.FINISH
                and action.args.get("status") == "success"
            ):
                evaluator = environment.evaluate()
                entry["evaluator_checked"] = True
                entry["evaluator_resolved"] = _is_resolved(evaluator)
                if not entry["evaluator_resolved"]:
                    raise RuntimeError(
                        "Corrected finish(success) target did not pass the evaluator"
                    )
            entry["passed"] = True
            validated_sample_ids.add(sample["sample_id"])
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if cleanup is not None:
                try:
                    cleanup()
                    entry["cleanup_ok"] = True
                except Exception as exc:
                    entry["passed"] = False
                    entry["error"] = f"cleanup failed: {type(exc).__name__}: {exc}"
            if not entry["cleanup_ok"]:
                entry["passed"] = False
            registry.validate("corrected_replay", entry)
            ledgers.append(entry)

    write_jsonl(ledger_path, ledgers)
    passed = all(entry["passed"] for entry in ledgers)
    if passed:
        output_samples: list[dict[str, Any]] = []
        for sample in samples:
            value = dict(sample)
            if value["sample_id"] in validated_sample_ids:
                value["sandbox_validated"] = True
            registry.validate("decision_sample", value)
            output_samples.append(value)
        write_jsonl(output_dataset_path, output_samples)
    return {
        "schema_version": "2.0.0",
        "report_type": "corrected_sample_replay",
        "dataset_samples": len(samples),
        "corrected_samples": len(corrected),
        "passed_replays": sum(entry["passed"] for entry in ledgers),
        "ledger": str(ledger_path.resolve()),
        "output_dataset": str(output_dataset_path.resolve()) if passed else None,
        "passed": passed,
    }
