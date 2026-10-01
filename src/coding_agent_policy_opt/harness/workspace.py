from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceProfile:
    timeout_seconds: int = 600
    cpus: float = 2.0
    memory_mb: int = 4096
    pids_limit: int = 256
    network: bool = False


@dataclass(frozen=True)
class WorkspaceSpec:
    source_repo: Path
    workspace: Path
    base_ref: str = "HEAD"
    image: str | None = None
    runtime_image_digest: str | None = None
    resource_profile: ResourceProfile = ResourceProfile()


class LocalWorkspaceProvider:
    """Windows-friendly provider for an already isolated managed worktree."""

    backend = "local"

    def validate(self, spec: WorkspaceSpec) -> None:
        if not spec.workspace.is_dir() or not (spec.workspace / ".git").exists():
            raise ValueError(f"Workspace is not a Git worktree: {spec.workspace}")

    def command(self, command: list[str]) -> list[str]:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("Local commands must be a non-empty argv list")
        return list(command)


class DockerWorkspaceProvider:
    """Build a constrained Docker invocation for the Linux runtime profile.

    The provider is deliberately command-oriented: the Agent Harness remains
    backend-independent and the server worker decides how to stream execution.
    """

    backend = "docker"

    def __init__(self, docker_binary: str = "docker") -> None:
        self.docker_binary = docker_binary

    def preflight(self) -> None:
        if shutil.which(self.docker_binary) is None:
            raise RuntimeError("Docker is not installed or not available on PATH")
        result = subprocess.run(
            [self.docker_binary, "version", "--format", "{{.Server.Version}}"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Docker daemon is unavailable")

    def command(self, spec: WorkspaceSpec, command: list[str]) -> list[str]:
        if not spec.image:
            raise ValueError("Docker workspace requires an image")
        profile = spec.resource_profile
        if not spec.workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {spec.workspace}")
        network = "default" if profile.network else "none"
        return [
            self.docker_binary,
            "run",
            "--rm",
            "--network", network,
            "--cpus", str(profile.cpus),
            "--memory", f"{profile.memory_mb}m",
            "--pids-limit", str(profile.pids_limit),
            "--mount", f"type=bind,src={spec.workspace},dst=/workspace,rw",
            "--workdir", "/workspace",
            spec.image,
            *command,
        ]
