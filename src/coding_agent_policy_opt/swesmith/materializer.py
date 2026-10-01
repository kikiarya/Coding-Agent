from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..data.manifest import TaskManifestEntry, read_jsonl, validate_manifest
from ..models import TaskSpec
from ..schema import SchemaRegistry
from .config import (
    SelectedSweSmithTask,
    SweSmithEnvironmentConfig,
    load_selected_swesmith_tasks,
)
from .environment import (
    WorkerClient,
    WslSweSmithEnvironment,
    default_swesmith_worker,
)


class SweSmithRolloutMaterializer:
    """Create one isolated pinned SWE-smith container for each RL rollout."""

    def __init__(
        self,
        inventory_path: Path,
        oracle_path: Path,
        manifest_path: Path,
        environment_config: SweSmithEnvironmentConfig,
        *,
        split: str = "train",
        worker: WorkerClient | None = None,
        tool_timeout_seconds: int = 600,
    ) -> None:
        report = validate_manifest(manifest_path)
        if not report.valid:
            raise ValueError(f"Task manifest is invalid: {report.errors}")
        selected = load_selected_swesmith_tasks(
            inventory_path, oracle_path, split=split
        )
        self.tasks: dict[str, SelectedSweSmithTask] = {
            item.task.instance_id: item for item in selected
        }
        registry = SchemaRegistry()
        self.manifest: dict[str, TaskManifestEntry] = {}
        for value in read_jsonl(manifest_path):
            registry.validate("task_manifest", value)
            entry = TaskManifestEntry.from_dict(value)
            if entry.split != split:
                continue
            if entry.task_id in self.manifest:
                raise ValueError(f"Duplicate manifest task_id: {entry.task_id}")
            self.manifest[entry.task_id] = entry
        if not self.manifest:
            raise ValueError(f"Manifest contains no tasks for split={split!r}")
        if not set(self.manifest).issubset(self.tasks):
            raise ValueError("Manifest contains tasks absent from inventory/oracle")
        self.environment_config = environment_config
        self.worker = worker or default_swesmith_worker(environment_config)
        self.tool_timeout_seconds = tool_timeout_seconds

    def materialize(
        self, task_id: str, **kwargs: Any
    ) -> tuple[TaskSpec, WslSweSmithEnvironment, Callable[[], None]]:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unregistered materialization options: {unknown}")
        if task_id not in self.manifest:
            raise KeyError(f"Task is not in the frozen manifest: {task_id}")
        selected = self.tasks[task_id]
        manifest = self.manifest[task_id]
        environment = WslSweSmithEnvironment(
            selected.task,
            self.environment_config,
            worker=self.worker,
            tool_timeout_seconds=self.tool_timeout_seconds,
        )
        try:
            if environment.base_commit != manifest.base_commit:
                raise RuntimeError("RL environment base commit differs from the manifest")
            image_matches = bool(
                manifest.docker_image_digest == environment.image_id
                or manifest.docker_image_digest in environment.image_digests
            )
            if not image_matches:
                raise RuntimeError("RL environment image differs from the manifest")
            task = selected.task.to_task_spec(selected.split, environment.base_commit)
        except Exception:
            environment.close()
            raise

        def cleanup() -> None:
            if not environment.close():
                raise RuntimeError(
                    environment.cleanup_error or "SWE-smith container cleanup failed"
                )

        return task, environment, cleanup
