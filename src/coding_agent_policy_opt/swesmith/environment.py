from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

from ..models import ActionType, AgentAction, ErrorCode, TestState, TestStatus, ToolResult
from .config import SweSmithEnvironmentConfig, UpstreamSweSmithTask


WORKER_PROTOCOL_VERSION = "1.0.0"


class WorkerClient(Protocol):
    def invoke(self, request: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]: ...


class WslSweSmithWorker:
    """Invoke the pinned SWE-smith Docker worker inside one WSL2 distribution."""

    def __init__(self, distro: str, source_root: Path | None = None) -> None:
        self.distro = distro
        self.source_root = (source_root or Path(__file__).resolve().parents[1]).resolve()
        self._wsl_source_root: str | None = None

    def _resolve_wsl_source_root(self) -> str:
        if self._wsl_source_root is not None:
            return self._wsl_source_root
        result = subprocess.run(
            [
                "wsl.exe",
                "-d",
                self.distro,
                "--",
                "wslpath",
                "-a",
                "-u",
                str(self.source_root),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(result.stderr.strip() or "Unable to resolve the source path in WSL")
        self._wsl_source_root = result.stdout.strip()
        return self._wsl_source_root

    def invoke(self, request: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
        payload = {"protocol_version": WORKER_PROTOCOL_VERSION, **request}
        source_root = self._resolve_wsl_source_root()
        result = subprocess.run(
            [
                "wsl.exe",
                "-d",
                self.distro,
                "--",
                "env",
                f"PYTHONPATH={source_root}",
                "python3",
                "-m",
                "coding_agent_policy_opt.swesmith.worker",
            ],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or "SWE-smith worker failed"
                raise RuntimeError(message[-4000:]) from exc
            raise RuntimeError(
                f"SWE-smith worker returned invalid JSON: {result.stdout[-1000:]}"
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError("SWE-smith worker response must be an object")
        if response.get("protocol_version") != WORKER_PROTOCOL_VERSION:
            raise RuntimeError("SWE-smith worker protocol version mismatch")
        if result.returncode != 0 or not response.get("ok"):
            error = response.get("error", {})
            raise RuntimeError(str(error.get("message", "SWE-smith worker operation failed")))
        return response


class LocalSweSmithWorker:
    """Invoke the worker directly when the whole experiment runs inside WSL/Linux."""

    def invoke(self, request: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
        payload = {"protocol_version": WORKER_PROTOCOL_VERSION, **request}
        result = subprocess.run(
            [sys.executable, "-m", "coding_agent_policy_opt.swesmith.worker"],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"SWE-smith worker returned invalid JSON: {message[-2000:]}") from exc
        if result.returncode != 0 or not response.get("ok"):
            error = response.get("error", {}) if isinstance(response, dict) else {}
            raise RuntimeError(str(error.get("message", result.stderr.strip() or "Worker failed")))
        if response.get("protocol_version") != WORKER_PROTOCOL_VERSION:
            raise RuntimeError("SWE-smith worker protocol version mismatch")
        return response


def default_swesmith_worker(config: SweSmithEnvironmentConfig) -> WorkerClient:
    return WslSweSmithWorker(config.distro) if os.name == "nt" else LocalSweSmithWorker()


def preflight_swesmith_worker(
    config: SweSmithEnvironmentConfig,
    worker: WorkerClient | None = None,
) -> dict[str, Any]:
    client = worker or default_swesmith_worker(config)
    response = client.invoke(
        {
            "op": "dependency_preflight",
            "expected_repository_revision": config.repository_revision,
        },
        timeout_seconds=120,
    )
    return dict(response["data"])


class WslSweSmithEnvironment:
    """Stateful AgentEnvironment backed by a SWE-smith Docker container in WSL2."""

    def __init__(
        self,
        task: UpstreamSweSmithTask,
        config: SweSmithEnvironmentConfig,
        *,
        worker: WorkerClient | None = None,
        tool_timeout_seconds: int = 600,
    ) -> None:
        self.task = task
        self.config = config
        self.worker = worker or default_swesmith_worker(config)
        if tool_timeout_seconds < 1 or tool_timeout_seconds > 3600:
            raise ValueError("tool_timeout_seconds must be between 1 and 3600")
        self.tool_timeout_seconds = tool_timeout_seconds
        self.instance = task.to_upstream_dict()
        self.container_id: str | None = None
        self.created_container_id: str | None = None
        self.cleanup_error: str | None = None
        self.base_commit = ""
        self.image_id = ""
        self.image_digests: tuple[str, ...] = ()
        self.test_command = ""
        self.last_test_state: TestState | None = None
        self._diff_hash = ""
        self._closed = False

        response = self.worker.invoke(
            {"op": "materialize", "instance": self.instance},
            timeout_seconds=1800,
        )
        data = response["data"]
        self.container_id = self._required_string(data, "container_id")
        self.created_container_id = self.container_id
        try:
            self.base_commit = self._required_string(data, "base_commit")
            self.image_id = self._required_string(data, "image_id")
            self.test_command = self._required_string(data, "test_command")
            self._diff_hash = self._required_string(data, "diff_hash")
            digests = data.get("image_digests", [])
            if not isinstance(digests, list) or not all(
                isinstance(item, str) for item in digests
            ):
                raise RuntimeError("Worker image_digests must be a string list")
            self.image_digests = tuple(digests)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _required_string(value: dict[str, Any], key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise RuntimeError(f"Worker response is missing {key}")
        return item

    @staticmethod
    def _test_state(value: Any) -> TestState | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("Worker test_state must be an object or null")
        return TestState(
            status=TestStatus(value["status"]),
            fail_to_pass_count=value.get("fail_to_pass_count"),
            pass_to_pass_failures=value.get("pass_to_pass_failures"),
        )

    def _request(self, op: str, **extra: Any) -> dict[str, Any]:
        if self._closed or self.container_id is None:
            raise RuntimeError("SWE-smith environment is closed")
        return self.worker.invoke(
            {
                "op": op,
                "container_id": self.container_id,
                "instance": self.instance,
                "expected_image_id": self.image_id,
                "timeout_seconds": self.tool_timeout_seconds,
                **extra,
            },
            timeout_seconds=1200,
        )

    def _tool_result(self, response: dict[str, Any]) -> ToolResult:
        raw = response.get("tool_result")
        if not isinstance(raw, dict):
            raise RuntimeError("Worker response is missing tool_result")
        previous_test_state = self.last_test_state
        self._diff_hash = self._required_string(response, "diff_hash")
        self.last_test_state = self._test_state(response.get("test_state"))
        state_changed = bool(raw.get("state_changed")) or (
            self.last_test_state != previous_test_state
            and raw.get("tool") in {"run_tests", "evaluator"}
        )
        data = raw.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("Worker tool_result.data must be an object")
        if bool(raw.get("ok")):
            return ToolResult.success(
                self._required_string(raw, "tool"),
                data,
                state_changed=state_changed,
            )
        try:
            code = ErrorCode(str(raw.get("error_code")))
        except ValueError:
            code = ErrorCode.ENVIRONMENT_ERROR
        return ToolResult.failure(
            self._required_string(raw, "tool"),
            code,
            str(raw.get("message", "SWE-smith tool failed")),
            retryable=bool(raw.get("retryable")),
            data=data,
        )

    def state_payload(self) -> dict[str, Any]:
        return {
            "base_commit": self.base_commit,
            "image_id": self.image_id,
            "diff_hash": self._diff_hash,
            "test_state": self.last_test_state.to_dict() if self.last_test_state else None,
        }

    def diff_hash(self) -> str:
        return self._diff_hash

    def execute(self, action: AgentAction) -> ToolResult:
        if action.type is ActionType.FINISH:
            return ToolResult.success("finish", {"accepted": True})
        try:
            return self._tool_result(
                self._request("execute", action=action.to_dict())
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(
                action.type.value,
                ErrorCode.ENVIRONMENT_ERROR,
                str(exc),
                retryable=True,
            )

    def evaluate(self) -> ToolResult:
        try:
            return self._tool_result(self._request("evaluate"))
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return ToolResult.failure(
                "evaluator",
                ErrorCode.ENVIRONMENT_ERROR,
                str(exc),
                retryable=True,
            )

    def close(self) -> bool:
        if self._closed:
            return self.cleanup_error is None
        self._closed = True
        container_id = self.container_id
        if container_id is None:
            return True
        try:
            self.worker.invoke(
                {"op": "close", "container_id": container_id},
                timeout_seconds=120,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self.cleanup_error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def materialization_metadata(self) -> dict[str, Any]:
        return {
            "container_id": self.created_container_id,
            "base_commit": self.base_commit,
            "image_id": self.image_id,
            "image_digests": list(self.image_digests),
            "test_command": self.test_command,
        }

    def __enter__(self) -> "WslSweSmithEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
