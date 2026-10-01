from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

from ..agent.api_policy import APIModelPolicy, load_api_model_policy

from jsonschema import ValidationError

from .artifacts import RunArtifactStore
from ..config import (
    ExperimentConfig,
    HarnessConfig,
    HarnessPolicyConfig,
    ModelConfig,
    RepositoryConfig,
    RunConfig,
)
from .environment import AgentEnvironment, RepositoryEnvironment, RepositoryProfile
from ..hashing import sha256_json
from .policy import HarnessPolicy, build_harness_policy
from .tools import DEFAULT_TOOL_REGISTRY, ToolRegistry, dispatch_action
from .recovery import RECOVERY_PROFILES
from ..models import (
    SCHEMA_VERSION,
    ActionType,
    AgentDecision,
    EpisodeRecord,
    ErrorCode,
    StepRecord,
    TaskSpec,
    TerminationReason,
    TestState,
    TestStatus,
    ToolResult,
)
from ..agent.policy import AgentPolicy
from ..schema import SchemaRegistry


EventSink = Callable[[str, dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class AgentHarness:
    def __init__(
        self,
        config: ExperimentConfig,
        environment: AgentEnvironment,
        artifacts: RunArtifactStore,
        schemas: SchemaRegistry | None = None,
        harness_policy: HarnessPolicy | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.environment = environment
        self.artifacts = artifacts
        self.schemas = schemas or SchemaRegistry()
        self.harness_policy = harness_policy or build_harness_policy(config)
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        recovery_config = config.raw.get("recovery", RECOVERY_PROFILES["full"])
        if not isinstance(recovery_config, dict) or any(
            key not in RECOVERY_PROFILES["full"] or not isinstance(value, bool)
            for key, value in recovery_config.items()
        ):
            raise ValueError("recovery must contain boolean candidates/evidence_guidance/candidate_switching settings")
        if isinstance(environment, RepositoryEnvironment):
            environment.recovery.candidates_enabled = bool(recovery_config.get("candidates", True))
            environment.evidence_guidance = bool(recovery_config.get("evidence_guidance", True))
            environment.candidate_switching = bool(recovery_config.get("candidate_switching", True))

    def _combined_state_hash(self, step_index: int, tokens_used: int, test_calls: int) -> str:
        return sha256_json(
            {
                "environment": getattr(self.environment, "durable_state_payload", self.environment.state_payload)(),
                "budget": {
                    "remaining_steps": max(self.config.harness.max_tool_steps - step_index, 0),
                    "remaining_tokens": max(self.config.harness.max_total_tokens - tokens_used, 0),
                    "remaining_test_calls": max(
                        self.config.harness.max_test_calls - test_calls, 0
                    ),
                },
            }
        )

    @staticmethod
    def _test_state_from_result(result: ToolResult) -> TestState | None:
        value = result.data.get("test_state")
        if not isinstance(value, dict):
            return None
        return TestState(
            status=TestStatus(value["status"]),
            fail_to_pass_count=value.get("fail_to_pass_count"),
            pass_to_pass_failures=value.get("pass_to_pass_failures"),
        )

    def run(
        self,
        task: TaskSpec,
        policy: AgentPolicy,
        *,
        event_sink: EventSink | None = None,
        should_cancel: CancelCheck | None = None,
        resume_state: dict[str, Any] | None = None,
        attempt_id: str | None = None,
    ) -> EpisodeRecord:
        def emit(event_type: str, payload: dict[str, Any]) -> None:
            if event_sink is not None:
                event_sink(event_type, payload)

        if task.base_commit != self.environment.base_commit:
            raise ValueError(
                "Task base_commit does not match the materialized environment: "
                f"{task.base_commit} != {self.environment.base_commit}"
            )
        run_id = self.artifacts.run_dir.name
        episode_id = f"ep_{uuid.uuid4().hex}"
        parent_episode_id = (
            str(resume_state.get("episode_id"))
            if resume_state and resume_state.get("episode_id")
            else None
        )
        resumed_from_step = (
            int(resume_state["next_step_index"])
            if resume_state and resume_state.get("next_step_index") is not None
            else None
        )
        started = time.monotonic()
        prior_steps = int(resume_state.get("steps_used", 0)) if resume_state else 0
        prior_elapsed = float(resume_state.get("elapsed_seconds", 0)) if resume_state else 0.0
        records: list[StepRecord] = []
        total_input_tokens = int(resume_state.get("input_tokens", 0)) if resume_state else 0
        total_output_tokens = int(resume_state.get("output_tokens", 0)) if resume_state else 0
        test_calls = int(resume_state.get("test_calls", 0)) if resume_state else 0
        patch_groups = 0
        verified_patch_groups = 0
        duplicate_actions = 0
        invalid_actions = 0
        recovery_opportunities = 0
        recovered_episodes = 0
        patch_group_open = False
        active_recovery_baseline: int | None = None
        recovery_patch_applied = False
        action_hashes_since_change: set[str] = set()
        interaction_history: list[dict[str, Any]] = list(
            resume_state.get("history", []) if resume_state else []
        )
        if resume_state is not None:
            restore_state = getattr(self.environment, "restore_recovery_state", None)
            if callable(restore_state) and resume_state.get("recovery_state"):
                restore_state(resume_state["recovery_state"])
            reset_reads = getattr(self.environment, "reset_read_evidence", None)
            if callable(reset_reads):
                reset_reads()
        if resume_state and isinstance(resume_state.get("test_state"), dict):
            self.environment.last_test_state = TestState(
                status=TestStatus(resume_state["test_state"]["status"]),
                fail_to_pass_count=resume_state["test_state"].get("fail_to_pass_count"),
                pass_to_pass_failures=resume_state["test_state"].get("pass_to_pass_failures"),
            )
        termination: TerminationReason | None = None

        materialization = getattr(self.environment, "materialization_metadata", None)
        self.artifacts.write_run_metadata(
            {
                "schema_version": SCHEMA_VERSION,
                "state_contract": "workspace-v1" if isinstance(self.environment, RepositoryEnvironment) else "legacy",
                "run_id": run_id,
                "task_id": task.task_id,
                "split": task.split,
                "repo_id": task.repo_id,
                "base_commit": task.base_commit,
                "issue": task.issue,
                "seed": self.config.run.seed,
                "experiment_config_hash": self.config.config_hash,
                "config": self.config.raw,
                "harness_policy": self.harness_policy.describe(),
                "environment_materialization": (
                    materialization() if callable(materialization) else None
                ),
                "attempt_id": attempt_id,
                "parent_episode_id": parent_episode_id,
                "resumed_from_step": resumed_from_step,
            }
        )
        emit(
            "run_started",
            {
                "run_id": run_id,
                "task_id": task.task_id,
                "repo_id": task.repo_id,
                "base_commit": task.base_commit,
                "harness_policy": self.harness_policy.describe(),
            },
        )

        initial_result = self.environment.evaluate()
        self.artifacts.write_observation(initial_result)
        initial_evaluator_artifact = self.artifacts.write_evaluator(
            f"{episode_id}_initial", initial_result
        )
        initial_test = self._test_state_from_result(initial_result)
        initial_environment_valid = bool(
            initial_result.ok
            and initial_test is not None
            and initial_test.status in {TestStatus.PASSED, TestStatus.FAILED}
            and initial_test.fail_to_pass_count is not None
        )
        initial_fail_to_pass_count = (
            initial_test.fail_to_pass_count if initial_environment_valid and initial_test else None
        )
        best_fail_to_pass_count = initial_fail_to_pass_count
        if not initial_environment_valid:
            termination = TerminationReason.ENVIRONMENT_ERROR
        emit(
            "initial_evaluation",
            {
                "ok": initial_result.ok,
                "tool": initial_result.tool,
                "test_state": initial_test.to_dict() if initial_test else None,
                "environment_valid": initial_environment_valid,
                "observation_hash": initial_result.observation_hash,
            },
        )

        for step_index in range(max(self.config.harness.max_tool_steps - prior_steps, 0)):
            if termination is not None:
                break
            if should_cancel is not None and should_cancel():
                termination = TerminationReason.CANCELLED
                emit("run_cancelled", {"step_index": step_index})
                break
            if prior_elapsed + time.monotonic() - started >= self.config.harness.task_timeout_seconds:
                termination = TerminationReason.TASK_TIMEOUT
                break
            if total_input_tokens + total_output_tokens >= self.config.harness.max_total_tokens:
                termination = TerminationReason.TOKEN_BUDGET_EXHAUSTED
                break

            base_context = {
                "task": {
                    "task_id": task.task_id,
                    "issue": task.issue,
                    "repo_id": task.repo_id,
                },
                "step_index": step_index,
                "environment_state": self.environment.state_payload(),
                "history": interaction_history,
                "recent_steps": [record.to_dict() for record in records[-8:]],
            }
            last_record = records[-1] if records else None
            last_tool_result = last_record.tool_result if last_record else {}
            policy_state = {
                "patch_pending_verification": patch_group_open,
                "recovery_active": active_recovery_baseline is not None,
                "last_action_type": (
                    last_record.action.get("type") if last_record else None
                ),
                "last_error_code": last_tool_result.get("error_code"),
                "last_test_passed": bool(
                    self.environment.last_test_state
                    and self.environment.last_test_state.status is TestStatus.PASSED
                ),
                "duplicate_actions": duplicate_actions,
                "invalid_actions": invalid_actions,
            }
            base_context["policy_state"] = policy_state
            policy_decision = self.harness_policy.decide(base_context)
            omitted_history = max(
                len(interaction_history) - policy_decision.history_limit, 0
            )
            context = {
                **base_context,
                "history": interaction_history[-policy_decision.history_limit :],
                "harness_policy": policy_decision.to_dict(),
                "policy_state": policy_state | {"history_omitted": omitted_history},
            }
            state_hash_before = self._combined_state_hash(
                prior_steps + step_index, total_input_tokens + total_output_tokens, test_calls
            )
            emit(
                "step_started",
                {
                    "step_index": step_index,
                    "policy": policy_decision.to_dict(),
                    "state_hash_before": state_hash_before,
                },
            )
            output = policy.decide(context)
            decision_input_tokens = max(output.input_tokens, 0)
            decision_output_tokens = max(output.output_tokens, 0)
            total_input_tokens += decision_input_tokens
            total_output_tokens += decision_output_tokens
            labels: list[str] = []
            tool_latency_ms = 0
            if omitted_history:
                labels.append("HARNESS_CONTEXT_TRIMMED")

            if (
                decision_output_tokens > self.config.harness.max_decision_output_tokens
                or total_input_tokens + total_output_tokens
                > self.config.harness.max_total_tokens
            ):
                labels.append("BUDGET_EXHAUSTED")
                decision = None
                action_dict = {"type": "invalid", "args": {"raw": output.payload}}
                action_hash = sha256_json(action_dict)
                result = ToolResult.failure(
                    "invalid",
                    ErrorCode.BUDGET_EXHAUSTED,
                    "Model output exceeded the frozen token budget",
                )
                termination = TerminationReason.TOKEN_BUDGET_EXHAUSTED
            else:
                try:
                    self.schemas.validate("agent_action", output.payload)
                    decision = AgentDecision.from_dict(output.payload)
                    action_dict = decision.action.to_dict()
                    action_hash = decision.action.normalized_hash
                except (ValidationError, KeyError, TypeError, ValueError) as exc:
                    invalid_actions += 1
                    labels.append("INVALID_ACTION")
                    decision = None
                    action_dict = {"type": "invalid", "args": {"raw": output.payload}}
                    action_hash = sha256_json(action_dict)
                    result = ToolResult.failure(
                        "invalid",
                        ErrorCode.SCHEMA_VALIDATION_FAILED,
                        str(exc),
                        retryable=True,
                    )

            if decision is not None:
                if (
                    decision.action.type is not ActionType.FINISH
                    and action_hash in action_hashes_since_change
                ):
                    duplicate_actions += 1
                    labels.append("DUPLICATE_ACTION")

                action_type = decision.action.type.value
                if action_type not in policy_decision.allowed_tools:
                    invalid_actions += 1
                    labels.append("HARNESS_TOOL_GATED")
                    error_code = ErrorCode.TOOL_NOT_AVAILABLE
                    message = (
                        f"Tool {action_type!r} is not exposed in the current Harness Policy state."
                    )
                    result = ToolResult.failure(
                        action_type,
                        error_code,
                        message,
                        retryable=True,
                        data={
                            "allowed_tools": list(policy_decision.allowed_tools),
                            "context_strategy": policy_decision.context_strategy,
                        },
                    )
                elif (
                    decision.action.type is ActionType.FINISH
                    and decision.action.args.get("status") == "success"
                    and policy_decision.verification_required
                ):
                    invalid_actions += 1
                    labels.extend(("HARNESS_TOOL_GATED", "HARNESS_VERIFICATION_GUARD"))
                    result = ToolResult.failure(
                        action_type,
                        ErrorCode.VERIFICATION_REQUIRED,
                        "The Harness Policy requires a valid test after the patch before a success finish.",
                        retryable=True,
                        data={
                            "allowed_tools": list(policy_decision.allowed_tools),
                            "context_strategy": policy_decision.context_strategy,
                        },
                    )
                elif (
                    decision.action.type is ActionType.FINISH
                    and decision.action.args.get("status") == "success"
                    and policy_decision.recovery_required
                ):
                    invalid_actions += 1
                    labels.extend(("HARNESS_TOOL_GATED", "HARNESS_RECOVERY_GUARD"))
                    result = ToolResult.failure(
                        action_type,
                        ErrorCode.RECOVERY_REQUIRED,
                        "The Harness Policy requires resolution of the active recovery condition before a success finish.",
                        retryable=True,
                        data={
                            "allowed_tools": list(policy_decision.allowed_tools),
                            "context_strategy": policy_decision.context_strategy,
                        },
                    )
                elif decision.action.type is ActionType.RUN_TESTS:
                    if test_calls >= self.config.harness.max_test_calls:
                        result = ToolResult.failure(
                            "run_tests",
                            ErrorCode.BUDGET_EXHAUSTED,
                            "Test-call budget exhausted",
                        )
                        termination = TerminationReason.TEST_BUDGET_EXHAUSTED
                    else:
                        test_calls += 1
                        previous_test = self.environment.last_test_state
                        tool_started = time.perf_counter()
                        result = dispatch_action(
                            self.environment,
                            decision.action,
                            registry=self.tool_registry,
                            granted_permissions={
                                permission
                                for spec in self.tool_registry.list()
                                for permission in spec.permissions
                            },
                        )
                        tool_latency_ms = max(round((time.perf_counter() - tool_started) * 1000), 0)
                        current_test = self._test_state_from_result(result)
                        if result.ok and current_test is not None:
                            if current_test.fail_to_pass_count is not None:
                                best_fail_to_pass_count = (
                                    current_test.fail_to_pass_count
                                    if best_fail_to_pass_count is None
                                    else min(
                                        best_fail_to_pass_count,
                                        current_test.fail_to_pass_count,
                                    )
                                )
                            if current_test.status is TestStatus.PASSED:
                                labels.append("TEST_PASSED")
                            elif current_test.status is TestStatus.FAILED:
                                labels.append("TEST_FAILED")

                            if patch_group_open:
                                verified_patch_groups += 1
                                patch_group_open = False
                                labels.append("POST_PATCH_VERIFIED")

                            if (
                                previous_test is not None
                                and previous_test.fail_to_pass_count is not None
                                and current_test.fail_to_pass_count is not None
                                and current_test.fail_to_pass_count
                                < previous_test.fail_to_pass_count
                            ):
                                labels.append("TEST_IMPROVED")

                            if (
                                active_recovery_baseline is not None
                                and recovery_patch_applied
                                and current_test.fail_to_pass_count is not None
                                and current_test.fail_to_pass_count < active_recovery_baseline
                            ):
                                recovered_episodes += 1
                                labels.append("RECOVERED")
                                active_recovery_baseline = None
                                recovery_patch_applied = False

                            remaining_steps = self.config.harness.max_tool_steps - prior_steps - step_index - 1
                            remaining_tests = self.config.harness.max_test_calls - test_calls
                            if (
                                current_test.status is TestStatus.FAILED
                                and current_test.fail_to_pass_count is not None
                                and active_recovery_baseline is None
                                and remaining_steps >= 2
                                and remaining_tests >= 1
                            ):
                                recovery_opportunities += 1
                                active_recovery_baseline = current_test.fail_to_pass_count
                                recovery_patch_applied = False
                                labels.append("RECOVERY_OPPORTUNITY")
                else:
                    tool_started = time.perf_counter()
                    result = dispatch_action(
                        self.environment,
                        decision.action,
                        registry=self.tool_registry,
                        granted_permissions={
                            permission
                            for spec in self.tool_registry.list()
                            for permission in spec.permissions
                        },
                    )
                    tool_latency_ms = max(round((time.perf_counter() - tool_started) * 1000), 0)

                if decision.action.type is ActionType.APPLY_PATCH and result.state_changed:
                    labels.append("PATCH_APPLIED")
                    if not patch_group_open:
                        patch_groups += 1
                        patch_group_open = True
                    if active_recovery_baseline is not None:
                        recovery_patch_applied = True

                if decision.action.type is ActionType.FINISH and result.ok:
                    finish_status = decision.action.args["status"]
                    termination = (
                        TerminationReason.MODEL_FINISH_SUCCESS
                        if finish_status == "success"
                        else TerminationReason.MODEL_GIVE_UP
                    )

                new_conflict = (
                    result.error_code in {ErrorCode.PATCH_CONFLICT, ErrorCode.FILE_NOT_READ}
                    and self.environment.state_payload().get("patch_conflicts")
                    != base_context["environment_state"].get("patch_conflicts")
                )
                if result.state_changed or new_conflict:
                    action_hashes_since_change.clear()
                action_hashes_since_change.add(action_hash)

            self.artifacts.write_observation(result)
            interaction_history.append(
                {
                    "assistant": output.payload,
                    "observation": result.to_observation(),
                }
            )
            state_hash_after = self._combined_state_hash(
                prior_steps + step_index + 1, total_input_tokens + total_output_tokens, test_calls
            )
            record = StepRecord(
                schema_version=SCHEMA_VERSION,
                run_id=run_id,
                episode_id=episode_id,
                task_id=task.task_id,
                split=task.split,
                repo_id=task.repo_id,
                base_commit=task.base_commit,
                step_index=step_index,
                model_id=self.config.model.model_id,
                model_revision=self.config.model.model_revision,
                adapter_id=self.config.model.adapter_id,
                prompt_version=self.config.model.prompt_version,
                harness_version=self.config.harness.harness_version,
                experiment_config_hash=self.config.config_hash,
                state_hash_before=state_hash_before,
                intent=decision.intent if decision else "",
                action=action_dict,
                normalized_action_hash=action_hash,
                tool_result=result.to_record_summary(),
                state_hash_after=state_hash_after,
                diff_hash=self.environment.diff_hash(),
                test_state=(
                    self.environment.last_test_state.to_dict()
                    if self.environment.last_test_state
                    else None
                ),
                input_tokens=decision_input_tokens,
                output_tokens=decision_output_tokens,
                latency_ms=max(output.latency_ms, 0),
                tool_latency_ms=tool_latency_ms,
                automatic_labels=sorted(set(labels)),
                harness_policy_variant=policy_decision.variant,
                harness_allowed_tools=list(policy_decision.allowed_tools),
                harness_context_strategy=policy_decision.context_strategy,
                harness_verification_required=policy_decision.verification_required,
                harness_recovery_required=policy_decision.recovery_required,
            )
            records.append(record)
            self.artifacts.write_step(record)
            self.artifacts.write_checkpoint(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "task_id": task.task_id,
                    "step_index": step_index,
                    "next_step_index": step_index + 1,
                    "state_hash_after": record.state_hash_after,
                    "diff_hash": record.diff_hash,
                    "observation_hash": result.observation_hash,
                    "replayable": decision is not None,
                    **({"state_contract": "workspace-v1",
                        "recovery_state_ref": self.artifacts.write_recovery_state(self.environment.export_recovery_state()),
                        "budget_used": {"steps": prior_steps + step_index + 1,
                                        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
                                        "test_calls": test_calls,
                                        "elapsed_seconds": prior_elapsed + time.monotonic() - started}}
                       if isinstance(self.environment, RepositoryEnvironment) else {}),
                }
            )
            result_preview = result.to_observation()
            result_data = result_preview.get("data")
            if isinstance(result_data, dict):
                result_preview["data"] = {
                    key: (value[-4000:] if isinstance(value, str) else value)
                    for key, value in result_data.items()
                }
            emit(
                "step_completed",
                {
                    "step_index": step_index,
                    "intent": record.intent,
                    "action": record.action,
                    "tool_result": result_preview,
                    "test_state": record.test_state,
                    "diff_hash": record.diff_hash,
                    "labels": record.automatic_labels,
                    "harness_policy": policy_decision.to_dict(),
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "latency_ms": record.latency_ms,
                },
            )

            if termination is not None:
                break

        if termination is None:
            termination = TerminationReason.STEP_BUDGET_EXHAUSTED

        evaluator_result = self.environment.evaluate()
        self.artifacts.write_observation(evaluator_result)
        evaluator_artifact = self.artifacts.write_evaluator(episode_id, evaluator_result)
        evaluator_test = self._test_state_from_result(evaluator_result)
        if evaluator_test is not None and evaluator_test.fail_to_pass_count is not None:
            best_fail_to_pass_count = (
                evaluator_test.fail_to_pass_count
                if best_fail_to_pass_count is None
                else min(best_fail_to_pass_count, evaluator_test.fail_to_pass_count)
            )
        final_environment_valid = bool(
            evaluator_result.ok
            and evaluator_test is not None
            and evaluator_test.status in {TestStatus.PASSED, TestStatus.FAILED}
            and evaluator_test.fail_to_pass_count is not None
        )
        environment_valid = initial_environment_valid and final_environment_valid
        resolved = bool(
            environment_valid
            and evaluator_test is not None
            and evaluator_test.status is TestStatus.PASSED
            and evaluator_test.fail_to_pass_count == 0
            and evaluator_test.pass_to_pass_failures == 0
        )
        emit(
            "evaluator_completed",
            {
                "ok": evaluator_result.ok,
                "resolved": resolved,
                "test_state": evaluator_test.to_dict() if evaluator_test else None,
                "artifact": evaluator_artifact,
                "observation_hash": evaluator_result.observation_hash,
            },
        )

        if (
            records
            and termination is TerminationReason.MODEL_FINISH_SUCCESS
            and (patch_group_open or not resolved)
        ):
            records[-1].automatic_labels = sorted(
                set([*records[-1].automatic_labels, "PREMATURE_STOP"])
            )
            self.artifacts.rewrite_steps(records)

        episode = EpisodeRecord(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            episode_id=episode_id,
            task_id=task.task_id,
            split=task.split,
            repo_id=task.repo_id,
            model_id=self.config.model.model_id,
            adapter_id=self.config.model.adapter_id,
            experiment_config_hash=self.config.config_hash,
            seed=self.config.run.seed,
            termination_reason=termination.value,
            resolved=resolved,
            environment_valid=environment_valid,
            total_steps=len(records),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            test_calls=test_calls,
            patch_groups=patch_groups,
            verified_patch_groups=verified_patch_groups,
            duplicate_actions=duplicate_actions,
            invalid_actions=invalid_actions,
            recovery_opportunities=recovery_opportunities,
            recovered_episodes=recovered_episodes,
            final_diff_hash=self.environment.diff_hash(),
            evaluator_artifact=evaluator_artifact,
            initial_evaluator_artifact=initial_evaluator_artifact,
            initial_fail_to_pass_count=initial_fail_to_pass_count,
            best_fail_to_pass_count=best_fail_to_pass_count,
            wall_time_ms=max(round((time.monotonic() - started) * 1000), 0),
            attempt_id=attempt_id,
            parent_episode_id=parent_episode_id,
            resumed_from_step=resumed_from_step,
        )
        self.artifacts.write_episode(episode)
        emit("run_completed", episode.to_dict())
        return episode


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ValueError(f"{detail}: git {' '.join(args)}")
    return result.stdout.strip()


def build_api_experiment_config(
    *,
    run_dir: Path,
    model_id: str,
    test_command: tuple[str, ...],
    default_test_args: tuple[str, ...],
    seed: int = 42,
    harness_policy: str = "adaptive_v1",
    history_limit: int = 8,
    verification_mode: str = "required",
    recovery_mode: str = "required",
    max_tool_steps: int = 30,
    max_test_calls: int = 8,
    max_total_tokens: int = 65_536,
    max_context_tokens: int = 16_384,
    max_decision_output_tokens: int = 1_024,
    tool_timeout_seconds: int = 600,
    task_timeout_seconds: int = 1_800,
    recovery_profile: str = "full",
) -> ExperimentConfig:
    """Build a product-track config without writing credentials or temp YAML."""

    if recovery_profile not in RECOVERY_PROFILES:
        raise ValueError("Unknown recovery profile")
    raw: dict[str, Any] = {
        "recovery": dict(RECOVERY_PROFILES[recovery_profile]),
        "schema_version": "2.0.0",
        "track": "api_product",
        "run": {"seed": seed, "artifacts_dir": str(run_dir.parent)},
        "model": {
            "model_id": model_id,
            "model_revision": f"api:{model_id}",
            "adapter_id": None,
            "prompt_version": "agent_prompt_v1",
        },
        "harness": {
            "harness_version": "harness_v2_two_layer",
            "max_tool_steps": max_tool_steps,
            "max_test_calls": max_test_calls,
            "max_total_tokens": max_total_tokens,
            "max_context_tokens": max_context_tokens,
            "max_decision_output_tokens": max_decision_output_tokens,
            "tool_timeout_seconds": tool_timeout_seconds,
            "task_timeout_seconds": task_timeout_seconds,
        },
        "harness_policy": {
            "variant": harness_policy,
            "history_limit": history_limit,
            "verification_mode": verification_mode,
            "recovery_mode": recovery_mode,
        },
        "repository": {
            "test_command": list(test_command),
            "default_test_args": list(default_test_args),
        },
    }
    return ExperimentConfig(
        schema_version="2.0.0",
        run=RunConfig(seed=seed, artifacts_dir=run_dir.parent),
        model=ModelConfig(
            model_id=model_id,
            model_revision=f"api:{model_id}",
            adapter_id=None,
            prompt_version="agent_prompt_v1",
        ),
        harness=HarnessConfig(
            harness_version="harness_v2_two_layer",
            max_tool_steps=max_tool_steps,
            max_test_calls=max_test_calls,
            max_total_tokens=max_total_tokens,
            max_context_tokens=max_context_tokens,
            max_decision_output_tokens=max_decision_output_tokens,
            tool_timeout_seconds=tool_timeout_seconds,
            task_timeout_seconds=task_timeout_seconds,
        ),
        harness_policy=HarnessPolicyConfig(
            variant=harness_policy,
            history_limit=history_limit,
            verification_mode=verification_mode,
            recovery_mode=recovery_mode,
        ),
        repository=RepositoryConfig(
            test_command=test_command,
            default_test_args=default_test_args,
        ),
        config_hash=sha256_json(raw),
        raw=raw,
    )


def run_api_repository(
    *,
    repo: Path,
    issue: str,
    run_dir: Path,
    api_config: Path,
    task_id: str = "api_task",
    test_command: tuple[str, ...] = ("{python}", "-m", "pytest"),
    default_test_args: tuple[str, ...] = ("-q",),
    seed: int = 42,
    harness_policy: str = "adaptive_v1",
    history_limit: int = 8,
    verification_mode: str = "required",
    recovery_mode: str = "required",
    allow_dirty: bool = False,
    policy_override: APIModelPolicy | None = None,
    event_sink: EventSink | None = None,
    should_cancel: CancelCheck | None = None,
    resume_state: dict[str, Any] | None = None,
    attempt_id: str | None = None,
    recovery_profile: str = "full",
) -> dict[str, Any]:
    """Run one API-backed episode in an explicitly supplied repository worktree."""

    repo = repo.resolve()
    run_dir = run_dir.resolve()
    if not repo.is_dir():
        raise ValueError(f"Repository root does not exist: {repo}")
    if not issue.strip():
        raise ValueError("issue must not be empty")
    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    if run_dir.exists() and resume_state is None:
        raise ValueError(f"run_dir already exists; choose a new directory: {run_dir}")

    # RepositoryEnvironment performs the authoritative worktree check. Do the
    # status check before creating artifacts so a refusal leaves no partial run.
    _git(repo, "rev-parse", "--show-toplevel")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status and not allow_dirty and resume_state is None:
        raise ValueError(
            "Repository worktree is dirty. Use a clean isolated worktree, or "
            "explicitly pass --allow-dirty."
        )

    policy = policy_override or load_api_model_policy(api_config)
    config = build_api_experiment_config(
        run_dir=run_dir,
        model_id=policy.model_id,
        test_command=test_command,
        default_test_args=default_test_args,
        seed=seed,
        harness_policy=harness_policy,
        history_limit=history_limit,
        verification_mode=verification_mode,
        recovery_mode=recovery_mode,
        recovery_profile=recovery_profile,
    )
    if resume_state is not None:
        previous_config = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["config"]
        config.raw["recovery"] = previous_config.get("recovery", dict(RECOVERY_PROFILES["full"]))
        config = replace(config, config_hash=sha256_json(config.raw))
    profile = RepositoryProfile(
        test_command=config.repository.test_command,
        default_test_args=config.repository.default_test_args,
        timeout_seconds=config.harness.tool_timeout_seconds,
    )
    environment = RepositoryEnvironment(repo, profile)
    task = TaskSpec(
        task_id=task_id,
        split="external",
        repo_id=repo.name,
        issue=issue,
        base_commit=environment.base_commit,
    )
    store = RunArtifactStore(run_dir, allow_existing=resume_state is not None)
    episode = AgentHarness(config, environment, store).run(
        task,
        policy,
        event_sink=event_sink,
        should_cancel=should_cancel,
        resume_state=resume_state,
        attempt_id=attempt_id,
    )
    return {
        "track": "api_product",
        "run_dir": str(run_dir),
        "repository": str(repo),
        "base_commit": environment.base_commit,
        "worktree_dirty_before_run": bool(status),
        "harness_policy": config.harness_policy.variant,
        "model_id": policy.model_id,
        "episode": episode.to_dict(),
        "review": [
            f"git -C {repo} diff",
            f"inspect {run_dir / 'episodes.jsonl'}",
            f"inspect {run_dir / 'steps.jsonl'}",
        ],
    }
