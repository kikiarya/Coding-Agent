from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import ValidationError, validate

from ..models import ActionType, AgentAction, ErrorCode, ToolResult

# Canonical registered tool surface for Harness Policy and prompts.
REGISTERED_TOOLS = (
    "search_code",
    "read_file",
    "apply_patch",
    "run_tests",
    "inspect_diff",
    "finish",
)

ACTION_TYPE_BY_NAME = {item.value: item for item in ActionType}


@dataclass(frozen=True)
class ToolSpec:
    """Versioned capability metadata for one repository tool."""

    name: str
    version: str
    permissions: tuple[str, ...]
    timeout_seconds: int
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    retryable_errors: tuple[str, ...] = ()
    resource_profile: Mapping[str, Any] | None = None


class ToolRegistry:
    """Canonical registry used by prompts, Harness gates, and execution."""

    def __init__(self, specs: tuple[ToolSpec, ...] | None = None) -> None:
        self._specs = {spec.name: spec for spec in (specs or default_tool_specs())}

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown registered tool: {name}") from exc

    def list(self) -> list[ToolSpec]:
        return [self._specs[name] for name in REGISTERED_TOOLS if name in self._specs]

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.list())

    def check_permissions(self, name: str, granted: set[str] | None) -> None:
        spec = self.get(name)
        if granted is None:
            return
        missing = sorted(set(spec.permissions) - granted)
        if missing:
            raise PermissionError(f"Tool {name!r} requires permissions: {', '.join(missing)}")

    def validate_action(self, action: AgentAction) -> None:
        validate(instance=action.args, schema=self.get(action.type.value).input_schema)


def default_tool_specs() -> tuple[ToolSpec, ...]:
    search = {
        "type": "object", "additionalProperties": False,
        "required": ["query", "path"],
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 500},
            "path": {"type": "string", "minLength": 1, "maxLength": 1000},
            "glob": {"type": "string", "maxLength": 200},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
        },
    }
    read = {
        "type": "object", "additionalProperties": False,
        "required": ["path", "start_line", "end_line"],
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": 1000},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
    }
    text_patch = {
        "type": "object", "additionalProperties": False,
        "required": ["patch"],
        "properties": {"patch": {"type": "string", "minLength": 1, "maxLength": 100000}},
    }
    patch = {"oneOf": [text_patch, {
        "type": "object", "additionalProperties": False,
        "required": ["candidate_id", "workspace_version"],
        "properties": {key: {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
                       for key in ("candidate_id", "workspace_version")},
    }]}
    tests = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "target": {"type": "string", "maxLength": 1000},
            "extra_args": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 200}},
        },
    }
    empty = {"type": "object", "additionalProperties": False}
    finish = {
        "type": "object", "additionalProperties": False,
        "required": ["status", "summary"],
        "properties": {
            "status": {"enum": ["success", "give_up"]},
            "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
    }
    return (
        ToolSpec("search_code", "1.0", ("repo.read",), 30, search, empty),
        ToolSpec("read_file", "1.0", ("repo.read",), 30, read, empty),
        ToolSpec("apply_patch", "1.0", ("repo.write",), 60, patch, empty),
        ToolSpec("run_tests", "1.0", ("process.test",), 600, tests, empty),
        ToolSpec("inspect_diff", "1.0", ("repo.read",), 30, empty, empty),
        ToolSpec("finish", "1.0", ("agent.finish",), 5, finish, empty),
    )


DEFAULT_TOOL_REGISTRY = ToolRegistry()


def is_registered_tool(name: str) -> bool:
    return name in REGISTERED_TOOLS


def registered_tool_list() -> list[str]:
    return list(REGISTERED_TOOLS)


def dispatch_action(
    environment: Any,
    action: AgentAction,
    *,
    registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
    granted_permissions: set[str] | None = None,
) -> ToolResult:
    """Map a validated action onto the environment executor.

    Real repository / sandbox execution stays inside the environment implementation.
    """

    try:
        registry.validate_action(action)
        registry.check_permissions(action.type.value, granted_permissions)
    except ValidationError as exc:
        return ToolResult.failure(
            action.type.value,
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            str(exc),
            retryable=True,
        )
    except KeyError:
        return ToolResult.failure(
            action.type.value,
            ErrorCode.UNKNOWN_TOOL,
            f"Tool {action.type.value!r} is not registered",
        )
    except PermissionError as exc:
        return ToolResult.failure(
            action.type.value,
            ErrorCode.PERMISSION_DENIED,
            str(exc),
        )
    return environment.execute(action)
