from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Literal, Protocol

from jsonschema import ValidationError

from ..harness.environment import AgentEnvironment
from ..config import HarnessPolicyConfig
from ..harness.policy import HarnessPolicy, build_harness_policy
from ..harness.tools import dispatch_action
from ..hashing import canonical_json
from ..models import ActionType, AgentAction, ErrorCode, TaskSpec, TestStatus, ToolResult
from ..evaluation.rewards import RewardComponents, RewardWeights, compute_trajectory_reward
from ..schema import SchemaRegistry


class RolloutMaterializer(Protocol):
    def materialize(
        self, task_id: str, **kwargs: Any
    ) -> tuple[TaskSpec, AgentEnvironment, Callable[[], None]]: ...


@dataclass(frozen=True)
class GRPOEnvironmentConfig:
    max_tool_steps: int = 30
    max_test_calls: int = 8
    include_recovery_ablation: bool = False
    reward_hacking_audit_passed: bool = False
    harness_policy_variant: str = "static_v1"
    harness_history_limit: int = 1_000_000
    harness_verification_mode: str = "advisory"
    harness_recovery_mode: str = "advisory"
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


class CodingGRPOEnvironment:
    """Stateful repository environment compatible with TRL environment_factory."""

    def __init__(
        self,
        materializer: RolloutMaterializer,
        config: GRPOEnvironmentConfig | None = None,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        harness_policy: HarnessPolicy | None = None,
    ) -> None:
        self.materializer = materializer
        self.config = config or GRPOEnvironmentConfig()
        self.schemas = SchemaRegistry()
        self.audit_sink = audit_sink
        policy_config = HarnessPolicyConfig(
            variant=self.config.harness_policy_variant,
            history_limit=self.config.harness_history_limit,
            verification_mode=self.config.harness_verification_mode,
            recovery_mode=self.config.harness_recovery_mode,
        )
        self.harness_policy = harness_policy or build_harness_policy(
            SimpleNamespace(harness_policy=policy_config)
        )
        self._cleanup: Callable[[], None] | None = None
        self._environment: AgentEnvironment | None = None
        self._task: TaskSpec | None = None
        self._reward: RewardComponents | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.total_steps = 0
        self.test_calls = 0
        self.patch_groups = 0
        self.verified_patch_groups = 0
        self.duplicate_actions = 0
        self.invalid_actions = 0
        self.harness_interventions = 0
        self.recovery_opportunities = 0
        self.recovered_episodes = 0
        self.patch_group_open = False
        self.initial_fail_to_pass_count: int | None = None
        self.best_fail_to_pass_count: int | None = None
        self.active_recovery_baseline: int | None = None
        self.recovery_patch_applied = False
        self.finished = False
        self.finish_status: str | None = None
        self.action_hashes_since_change: set[str] = set()
        self.trajectory_records: list[dict[str, Any]] = []
        self._reward = None

    def _harness_context(self) -> dict[str, Any]:
        last = self.trajectory_records[-1] if self.trajectory_records else None
        observation = last.get("observation", {}) if isinstance(last, dict) else {}
        return {
            "task": {
                "task_id": self._task.task_id if self._task is not None else "unknown",
                "repo_id": self._task.repo_id if self._task is not None else "unknown",
                "issue": self._task.issue if self._task is not None else "",
            },
            "step_index": self.total_steps,
            "environment_state": self.environment.state_payload(),
            "history": self.trajectory_records[-8:],
            "recent_steps": self.trajectory_records[-8:],
            "policy_state": {
                "patch_pending_verification": self.patch_group_open,
                "recovery_active": self.active_recovery_baseline is not None,
                "last_action_type": (
                    last.get("action", {}).get("type")
                    if isinstance(last, dict)
                    else None
                ),
                "last_error_code": (
                    observation.get("error_code")
                    if isinstance(observation, dict)
                    else None
                ),
                "last_test_passed": bool(
                    self.environment.last_test_state
                    and self.environment.last_test_state.status is TestStatus.PASSED
                ),
                "duplicate_actions": self.duplicate_actions,
                "invalid_actions": self.invalid_actions,
            },
        }

    def _current_harness_policy(self) -> dict[str, Any]:
        """Return the per-turn Harness decision for internal adapters.

        TRL discovers public environment methods when the trainer is created,
        so this helper is private and cannot accidentally become a model tool.
        The actual runtime gate remains in ``_execute``; an adapter that
        supports dynamic tool schemas can use this snapshot to render the
        current surface before generation.
        """

        return self.harness_policy.decide(self._harness_context()).to_dict()

    @property
    def environment(self) -> AgentEnvironment:
        if self._environment is None:
            raise RuntimeError("reset() must be called before using tools")
        return self._environment

    def reset(
        self, task_id: str, prompt: Any | None = None, **kwargs: Any
    ) -> str:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unregistered GRPO dataset fields: {unknown}")
        if self._cleanup is not None:
            self._cleanup()
        self._reset_state()
        self._task, self._environment, self._cleanup = self.materializer.materialize(task_id)
        initial_result = self.environment.evaluate()
        initial_test = initial_result.data.get("test_state")
        if (
            not initial_result.ok
            or not isinstance(initial_test, dict)
            or initial_test.get("status") not in {TestStatus.PASSED.value, TestStatus.FAILED.value}
            or not isinstance(initial_test.get("fail_to_pass_count"), int)
        ):
            self._close()
            raise RuntimeError(f"Initial evaluator failed for task {task_id}")
        self.initial_fail_to_pass_count = initial_test["fail_to_pass_count"]
        self.best_fail_to_pass_count = self.initial_fail_to_pass_count
        return self._task.issue

    def _budget_failure(self, tool: str, message: str) -> str:
        result = ToolResult.failure(tool, ErrorCode.BUDGET_EXHAUSTED, message)
        return canonical_json(result.to_observation())

    def _execute(self, action: AgentAction) -> str:
        if self.finished:
            return self._budget_failure(action.type.value, "Episode has already finished")
        if self.total_steps >= self.config.max_tool_steps:
            self.finished = True
            return self._budget_failure(action.type.value, "Tool-step budget exhausted")
        if action.type is ActionType.RUN_TESTS and self.test_calls >= self.config.max_test_calls:
            self.finished = True
            return self._budget_failure(action.type.value, "Test-call budget exhausted")

        try:
            self.schemas.validate(
                "agent_action",
                {"intent": "Execute the selected tool call.", "action": action.to_dict()},
            )
        except ValidationError as exc:
            self.invalid_actions += 1
            self.total_steps += 1
            result = ToolResult.failure(
                action.type.value,
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                exc.message,
                retryable=True,
            )
            self.trajectory_records.append(
                {
                    "step_index": self.total_steps - 1,
                    "action": action.to_dict(),
                    "normalized_action_hash": action.normalized_hash,
                    "observation": result.to_observation(),
                    "automatic_labels": ["INVALID_ACTION"],
                }
            )
            return canonical_json(result.to_observation())

        policy_decision = self.harness_policy.decide(self._harness_context())
        labels: list[str] = []
        action_hash = action.normalized_hash
        if action.type is not ActionType.FINISH and action_hash in self.action_hashes_since_change:
            self.duplicate_actions += 1
            labels.append("DUPLICATE_ACTION")

        policy_block: tuple[ErrorCode, str] | None = None
        if action.type.value not in policy_decision.allowed_tools:
            policy_block = (
                ErrorCode.TOOL_NOT_AVAILABLE,
                f"Tool {action.type.value!r} is not exposed in the current Harness Policy state.",
            )
        elif (
            action.type is ActionType.FINISH
            and action.args.get("status") == "success"
            and policy_decision.verification_required
        ):
            policy_block = (
                ErrorCode.VERIFICATION_REQUIRED,
                "The Harness Policy requires a valid test after the patch before a success finish.",
            )
            labels.append("HARNESS_VERIFICATION_GUARD")
        elif (
            action.type is ActionType.FINISH
            and action.args.get("status") == "success"
            and policy_decision.recovery_required
        ):
            policy_block = (
                ErrorCode.RECOVERY_REQUIRED,
                "The Harness Policy requires resolution of the active recovery condition before a success finish.",
            )
            labels.append("HARNESS_RECOVERY_GUARD")

        if policy_block is not None:
            self.invalid_actions += 1
            self.harness_interventions += 1
            labels.append("HARNESS_TOOL_GATED")
            result = ToolResult.failure(
                action.type.value,
                policy_block[0],
                policy_block[1],
                retryable=True,
                data={
                    "allowed_tools": list(policy_decision.allowed_tools),
                    "context_strategy": policy_decision.context_strategy,
                },
            )
            self.total_steps += 1
            self.trajectory_records.append(
                {
                    "step_index": self.total_steps - 1,
                    "action": action.to_dict(),
                    "normalized_action_hash": action_hash,
                    "observation": result.to_observation(),
                    "automatic_labels": sorted(set(labels)),
                    "harness_policy": policy_decision.to_dict(),
                }
            )
            return canonical_json(result.to_observation())

        if action.type is ActionType.RUN_TESTS:
            self.test_calls += 1
        conflicts_before = self.environment.state_payload().get("patch_conflicts")
        result = dispatch_action(self.environment, action)
        self.total_steps += 1

        if action.type is ActionType.APPLY_PATCH and result.state_changed:
            labels.append("PATCH_APPLIED")
            if not self.patch_group_open:
                self.patch_groups += 1
                self.patch_group_open = True
            if self.active_recovery_baseline is not None:
                self.recovery_patch_applied = True

        if action.type is ActionType.RUN_TESTS and result.ok:
            test_state = result.data.get("test_state")
            if isinstance(test_state, dict) and isinstance(
                test_state.get("fail_to_pass_count"), int
            ):
                current_failures = test_state["fail_to_pass_count"]
                self.best_fail_to_pass_count = (
                    current_failures
                    if self.best_fail_to_pass_count is None
                    else min(self.best_fail_to_pass_count, current_failures)
                )
                if self.patch_group_open:
                    self.verified_patch_groups += 1
                    self.patch_group_open = False
                    labels.append("POST_PATCH_VERIFIED")
                if (
                    self.active_recovery_baseline is not None
                    and self.recovery_patch_applied
                    and current_failures < self.active_recovery_baseline
                ):
                    self.recovered_episodes += 1
                    labels.append("RECOVERED")
                    self.active_recovery_baseline = None
                    self.recovery_patch_applied = False
                remaining_steps = self.config.max_tool_steps - self.total_steps
                remaining_tests = self.config.max_test_calls - self.test_calls
                if (
                    test_state.get("status") == TestStatus.FAILED.value
                    and self.active_recovery_baseline is None
                    and remaining_steps >= 2
                    and remaining_tests >= 1
                ):
                    self.recovery_opportunities += 1
                    self.active_recovery_baseline = current_failures
                    self.recovery_patch_applied = False
                    labels.append("RECOVERY_OPPORTUNITY")

        if action.type is ActionType.FINISH:
            self.finished = True
            self.finish_status = str(action.args["status"])
        new_conflict = (
            result.error_code in {ErrorCode.PATCH_CONFLICT, ErrorCode.FILE_NOT_READ}
            and self.environment.state_payload().get("patch_conflicts") != conflicts_before
        )
        if result.state_changed or new_conflict:
            self.action_hashes_since_change.clear()
        self.action_hashes_since_change.add(action_hash)
        self.trajectory_records.append(
            {
                "step_index": self.total_steps - 1,
                "action": action.to_dict(),
                "normalized_action_hash": action_hash,
                "observation": result.to_observation(),
                "automatic_labels": sorted(labels),
                "harness_policy": policy_decision.to_dict(),
            }
        )
        return canonical_json(result.to_observation())

    def search_code(
        self, query: str, path: str, glob: str = "*", max_results: int = 50
    ) -> str:
        """Search repository text.

        Args:
            query: Exact text to find.
            path: Repository-relative search root.
            glob: File glob applied below the search root.
            max_results: Maximum number of matches.

        Returns:
            A JSON tool observation.
        """
        return self._execute(
            AgentAction(
                ActionType.SEARCH_CODE,
                {"query": query, "path": path, "glob": glob, "max_results": max_results},
            )
        )

    def read_file(self, path: str, start_line: int, end_line: int) -> str:
        """Read a repository file range.

        Args:
            path: Repository-relative file path.
            start_line: First one-based line.
            end_line: Last one-based line.

        Returns:
            A JSON tool observation.
        """
        return self._execute(
            AgentAction(
                ActionType.READ_FILE,
                {"path": path, "start_line": start_line, "end_line": end_line},
            )
        )

    def apply_patch(self, patch: str) -> str:
        """Apply a repository-scoped unified diff.

        Args:
            patch: Git-style unified diff.

        Returns:
            A JSON tool observation.
        """
        return self._execute(AgentAction(ActionType.APPLY_PATCH, {"patch": patch}))

    def run_tests(
        self, target: str | None = None, extra_args: list[str] | None = None
    ) -> str:
        """Run tests through the frozen repository profile.

        Args:
            target: Optional repository-relative test target.
            extra_args: Optional safe test-runner arguments.

        Returns:
            A JSON tool observation.
        """
        args: dict[str, Any] = {"extra_args": extra_args or []}
        if target is not None:
            args["target"] = target
        return self._execute(AgentAction(ActionType.RUN_TESTS, args))

    def inspect_diff(self) -> str:
        """Inspect the current repository diff.

        Returns:
            A JSON tool observation.
        """
        return self._execute(AgentAction(ActionType.INSPECT_DIFF, {}))

    def finish(self, status: Literal["success", "give_up"], summary: str) -> str:
        """Request episode termination.

        Args:
            status: Either success or give_up.
            summary: Concise outcome summary.

        Returns:
            A JSON tool observation.
        """
        return self._execute(
            AgentAction(ActionType.FINISH, {"status": status, "summary": summary})
        )

    def get_reward(self) -> float:
        if self._reward is not None:
            return self._reward.total
        evaluator = self.environment.evaluate()
        test_state = evaluator.data.get("test_state")
        resolved = bool(
            evaluator.ok
            and isinstance(test_state, dict)
            and test_state.get("status") == TestStatus.PASSED.value
            and test_state.get("fail_to_pass_count") == 0
            and test_state.get("pass_to_pass_failures") == 0
        )
        if isinstance(test_state, dict) and isinstance(
            test_state.get("fail_to_pass_count"), int
        ):
            current_failures = test_state["fail_to_pass_count"]
            self.best_fail_to_pass_count = (
                current_failures
                if self.best_fail_to_pass_count is None
                else min(self.best_fail_to_pass_count, current_failures)
            )
        episode = {
            "resolved": resolved,
            "initial_fail_to_pass_count": self.initial_fail_to_pass_count,
            "best_fail_to_pass_count": self.best_fail_to_pass_count,
            "patch_groups": self.patch_groups,
            "verified_patch_groups": self.verified_patch_groups,
            "duplicate_actions": self.duplicate_actions,
            "invalid_actions": self.invalid_actions,
            "total_steps": self.total_steps,
            "recovered_episodes": self.recovered_episodes,
        }
        self._reward = compute_trajectory_reward(
            episode,
            max_tool_steps=self.config.max_tool_steps,
            weights=self.config.reward_weights,
            include_recovery_ablation=self.config.include_recovery_ablation,
            reward_hacking_audit_passed=self.config.reward_hacking_audit_passed,
        )
        audit = {
            "schema_version": "2.0.0",
            "task_id": self._task.task_id if self._task is not None else "unknown",
            "repo_id": self._task.repo_id if self._task is not None else "unknown",
            "finish_status": self.finish_status,
            "resolved": resolved,
            "reward_components": self._reward.to_dict(),
            "reward_weights": self.config.reward_weights.to_config(),
            "total_steps": self.total_steps,
            "test_calls": self.test_calls,
            "patch_groups": self.patch_groups,
            "verified_patch_groups": self.verified_patch_groups,
            "duplicate_actions": self.duplicate_actions,
            "invalid_actions": self.invalid_actions,
            "harness_intervention_count": self.harness_interventions,
            "harness_policy": self.harness_policy.describe(),
            "recovery_opportunities": self.recovery_opportunities,
            "recovered_episodes": self.recovered_episodes,
            "evaluator_test_state": test_state if isinstance(test_state, dict) else None,
            "trajectory_records": self.trajectory_records,
        }
        self.schemas.validate("rl_trajectory", audit)
        try:
            if self.audit_sink is not None:
                self.audit_sink(audit)
        finally:
            if self._cleanup is not None:
                self._cleanup()
                self._cleanup = None
        return self._reward.total

    @property
    def reward_breakdown(self) -> dict[str, float] | None:
        return self._reward.to_dict() if self._reward is not None else None

    def _close(self) -> None:
        if self._cleanup is not None:
            self._cleanup()
            self._cleanup = None
        self._environment = None
        self._task = None
