from __future__ import annotations

import unittest

from pathlib import Path

from coding_agent_policy_opt.harness.tools import ToolRegistry, default_tool_specs
from coding_agent_policy_opt.harness.workspace import DockerWorkspaceProvider, ResourceProfile, WorkspaceSpec
from coding_agent_policy_opt.models import ActionType, AgentAction, ErrorCode, ToolResult


class ToolRuntimeContractTests(unittest.TestCase):
    def test_registry_has_versioned_capability_specs(self) -> None:
        registry = ToolRegistry()
        self.assertEqual(
            registry.names(),
            ("search_code", "read_file", "apply_patch", "run_tests", "inspect_diff", "finish"),
        )
        self.assertIn("repo.write", registry.get("apply_patch").permissions)
        self.assertGreater(registry.get("run_tests").timeout_seconds, 0)

    def test_registry_rejects_missing_capability(self) -> None:
        registry = ToolRegistry(default_tool_specs())
        with self.assertRaises(PermissionError):
            registry.check_permissions("apply_patch", {"repo.read"})

    def test_dispatch_returns_auditable_permission_error(self) -> None:
        from coding_agent_policy_opt.harness.tools import dispatch_action

        class Environment:
            def execute(self, action):
                raise AssertionError("permission gate should run before execution")

        result = dispatch_action(
            Environment(),
            AgentAction(ActionType.APPLY_PATCH, {"patch": "diff --git a/x b/x"}),
            granted_permissions={"repo.read"},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ErrorCode.PERMISSION_DENIED)

    def test_docker_command_has_isolation_defaults(self) -> None:
        command = DockerWorkspaceProvider().command(
            WorkspaceSpec(
                source_repo=Path("/source"),
                workspace=Path.cwd(),
                image="capo-test:locked",
                resource_profile=ResourceProfile(cpus=1.5, memory_mb=512, pids_limit=32),
            ),
            ["python", "-m", "pytest", "-q"],
        )
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("--pids-limit", command)
        self.assertIn("--memory", command)
        self.assertIn("capo-test:locked", command)


if __name__ == "__main__":
    unittest.main()
