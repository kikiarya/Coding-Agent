from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from jsonschema import ValidationError

from ..models import AgentAction, ErrorCode, ToolResult
from .tools import ToolRegistry


class MCPClient(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class MCPToolAdapter:
    """Optional external-tool bridge that keeps CAPO policy checks local."""

    client: MCPClient
    registry: ToolRegistry

    def call(self, action: AgentAction, granted_permissions: set[str] | None = None) -> ToolResult:
        try:
            self.registry.validate_action(action)
            self.registry.check_permissions(action.type.value, granted_permissions)
            value = self.client.call_tool(action.type.value, action.args)
            return ToolResult.success(action.type.value, value if isinstance(value, dict) else {"result": value})
        except PermissionError as exc:
            return ToolResult.failure(action.type.value, ErrorCode.PERMISSION_DENIED, str(exc))
        except ValidationError as exc:
            return ToolResult.failure(
                action.type.value, ErrorCode.SCHEMA_VALIDATION_FAILED, str(exc), retryable=True
            )
        except Exception as exc:  # noqa: BLE001 - external adapter boundary
            return ToolResult.failure(
                action.type.value,
                ErrorCode.ENVIRONMENT_ERROR,
                f"MCP adapter failed: {type(exc).__name__}: {exc}",
                retryable=True,
            )
