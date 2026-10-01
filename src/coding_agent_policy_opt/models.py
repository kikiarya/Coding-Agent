from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .hashing import sha256_json


SCHEMA_VERSION = "2.0.0"


class ActionType(str, Enum):
    SEARCH_CODE = "search_code"
    READ_FILE = "read_file"
    APPLY_PATCH = "apply_patch"
    RUN_TESTS = "run_tests"
    INSPECT_DIFF = "inspect_diff"
    FINISH = "finish"


class ErrorCode(str, Enum):
    INVALID_JSON = "INVALID_JSON"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    PATH_OUTSIDE_REPOSITORY = "PATH_OUTSIDE_REPOSITORY"
    PATCH_APPLY_FAILED = "PATCH_APPLY_FAILED"
    PATCH_CONFLICT = "PATCH_CONFLICT"
    FILE_NOT_READ = "FILE_NOT_READ"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TOOL_NOT_AVAILABLE = "TOOL_NOT_AVAILABLE"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    COMMAND_NOT_ALLOWED = "COMMAND_NOT_ALLOWED"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class TerminationReason(str, Enum):
    MODEL_FINISH_SUCCESS = "MODEL_FINISH_SUCCESS"
    MODEL_GIVE_UP = "MODEL_GIVE_UP"
    STEP_BUDGET_EXHAUSTED = "STEP_BUDGET_EXHAUSTED"
    TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"
    TEST_BUDGET_EXHAUSTED = "TEST_BUDGET_EXHAUSTED"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    CANCELLED = "CANCELLED"


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class AgentAction:
    type: ActionType
    args: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentAction":
        return cls(type=ActionType(value["type"]), args=dict(value["args"]))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "args": self.args}

    @property
    def normalized_hash(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class AgentDecision:
    intent: str
    action: AgentAction

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentDecision":
        return cls(intent=value["intent"], action=AgentAction.from_dict(value["action"]))

    def to_dict(self) -> dict[str, Any]:
        return {"intent": self.intent, "action": self.action.to_dict()}


@dataclass(frozen=True)
class PolicyOutput:
    payload: Any
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass(frozen=True)
class TestState:
    status: TestStatus
    fail_to_pass_count: int | None
    pass_to_pass_failures: int | None = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "fail_to_pass_count": self.fail_to_pass_count,
            "pass_to_pass_failures": self.pass_to_pass_failures,
        }


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    data: dict[str, Any]
    state_changed: bool
    observation_hash: str
    error_code: ErrorCode | None = None
    message: str | None = None
    retryable: bool = False

    @classmethod
    def success(
        cls,
        tool: str,
        data: dict[str, Any],
        *,
        state_changed: bool = False,
    ) -> "ToolResult":
        payload = {
            "ok": True,
            "tool": tool,
            "data": data,
            "state_changed": state_changed,
        }
        return cls(
            ok=True,
            tool=tool,
            data=data,
            state_changed=state_changed,
            observation_hash=sha256_json(payload),
        )

    @classmethod
    def failure(
        cls,
        tool: str,
        error_code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        data: dict[str, Any] | None = None,
        state_changed: bool = False,
    ) -> "ToolResult":
        payload = {
            "ok": False,
            "tool": tool,
            "error_code": error_code.value,
            "message": message,
            "retryable": retryable,
            "data": data or {},
        }
        if state_changed:
            payload["state_changed"] = True
        return cls(
            ok=False,
            tool=tool,
            data=data or {},
            state_changed=state_changed,
            observation_hash=sha256_json(payload),
            error_code=error_code,
            message=message,
            retryable=retryable,
        )

    def to_observation(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "state_changed": self.state_changed,
            "observation_hash": self.observation_hash,
        }
        if not self.ok:
            value.update(
                {
                    "error_code": self.error_code.value if self.error_code else None,
                    "message": self.message,
                    "retryable": self.retryable,
                }
            )
        return value

    def to_record_summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_code": self.error_code.value if self.error_code else None,
            "observation_hash": self.observation_hash,
        }


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    repo_id: str
    issue: str
    base_commit: str


@dataclass
class StepRecord:
    schema_version: str
    run_id: str
    episode_id: str
    task_id: str
    split: str
    repo_id: str
    base_commit: str
    step_index: int
    model_id: str
    model_revision: str
    adapter_id: str | None
    prompt_version: str
    harness_version: str
    experiment_config_hash: str
    state_hash_before: str
    intent: str
    action: dict[str, Any]
    normalized_action_hash: str
    tool_result: dict[str, Any]
    state_hash_after: str
    diff_hash: str
    test_state: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    tool_latency_ms: int = 0
    retry_count: int = 0
    automatic_labels: list[str] = field(default_factory=list)
    harness_policy_variant: str = "static_v1"
    harness_allowed_tools: list[str] = field(default_factory=list)
    harness_context_strategy: str = "full_recent_history"
    harness_verification_required: bool = False
    harness_recovery_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: str
    run_id: str
    episode_id: str
    task_id: str
    split: str
    repo_id: str
    model_id: str
    adapter_id: str | None
    experiment_config_hash: str
    seed: int
    termination_reason: str
    resolved: bool
    environment_valid: bool
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    test_calls: int
    patch_groups: int
    verified_patch_groups: int
    duplicate_actions: int
    invalid_actions: int
    recovery_opportunities: int
    recovered_episodes: int
    final_diff_hash: str
    evaluator_artifact: str
    initial_evaluator_artifact: str
    initial_fail_to_pass_count: int | None
    best_fail_to_pass_count: int | None
    wall_time_ms: int
    attempt_id: str | None = None
    parent_episode_id: str | None = None
    resumed_from_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionSample:
    schema_version: str
    sample_id: str
    source_run_id: str
    source_episode_id: str
    source_step_index: int
    task_id: str
    split: str
    repo_id: str
    dataset: str
    category: str
    messages: list[dict[str, str]]
    target: dict[str, Any]
    rejected_action: dict[str, Any] | None
    error_labels: list[str]
    correction_method: str
    sandbox_validated: bool
    review_status: str
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
