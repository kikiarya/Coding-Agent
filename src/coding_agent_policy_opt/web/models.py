from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_path: str | None = Field(default=None, min_length=1, max_length=4096)
    source_repo: str | None = Field(default=None, min_length=1, max_length=4096)
    base_ref: str = Field(default="HEAD", min_length=1, max_length=200)
    issue: str = Field(min_length=1, max_length=20_000)
    task_id: str = Field(default="web_task", pattern=r"^[A-Za-z0-9_.:-]{1,120}$")
    model_policy: str = Field(default="api", pattern=r"^[A-Za-z0-9_.:-]{1,80}$")
    harness_policy: Literal["adaptive_v1", "static_v1"] = "adaptive_v1"
    workspace_backend: Literal["local", "docker"] = "local"
    execution_profile: str = Field(default="python", pattern=r"^[A-Za-z0-9_.:-]{1,80}$")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def require_repository(self) -> "RunRequest":
        if not self.repo_path and not self.source_repo:
            raise ValueError("repo_path or source_repo is required")
        if self.repo_path and self.source_repo and self.repo_path != self.source_repo:
            raise ValueError("repo_path and source_repo must refer to the same repository")
        if not self.repo_path:
            self.repo_path = self.source_repo
        if not self.source_repo:
            self.source_repo = self.repo_path
        return self


class RunResponse(BaseModel):
    run_id: str
    status: str
    task_id: str
    repo_path: str
    issue: str
    harness_policy: str
    model_policy: str = "api"
    source_repo: str | None = None
    base_ref: str = "HEAD"
    workspace_backend: str = "local"
    execution_profile: str = "python"
    attempt_count: int = 0
    run_dir: str | None = None
    error: str | None = None
    result: dict | None = None


class HealthResponse(BaseModel):
    status: str
    active_run: str | None = None
