from __future__ import annotations

import re
import shutil
import subprocess
import os
import hashlib
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..hashing import sha256_text
from ..data.manifest import read_jsonl, write_jsonl
from ..models import TaskSpec
from ..schema import SchemaRegistry


PINNED_SWESMITH_REPOSITORY_REVISION = "9b74ac08118a85c39c356802f7961893af73e07f"
PINNED_SWESMITH_DATASET_REVISION = "ea6d7173829c7ec8fa16c22055699ff2e9188091"


@dataclass(frozen=True)
class SweSmithEnvironmentConfig:
    profile: str
    platform: str
    distro: str
    minimum_python: tuple[int, int]
    require_wsl2: bool
    require_linux_engine: bool
    require_distro_integration: bool
    docker_data_root: str | None
    dependencies_root: str | None
    runtime_root_env: str | None
    require_native_linux: bool
    require_cuda: bool
    minimum_gpu_memory_gib: int
    minimum_free_gib: int
    repository: str
    repository_revision: str
    dataset_id: str
    dataset_revision: str
    hash_seed: int
    max_tasks_per_repository: int
    counts: dict[str, int]


@dataclass(frozen=True)
class UpstreamSweSmithTask:
    instance_id: str
    patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    image_name: str
    repo: str
    problem_statement: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UpstreamSweSmithTask":
        required_strings = [
            "instance_id",
            "patch",
            "image_name",
            "repo",
            "problem_statement",
        ]
        for field in required_strings:
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise ValueError(f"SWE-smith field {field} must be a non-empty string")
        fail_to_pass = value.get("FAIL_TO_PASS")
        pass_to_pass = value.get("PASS_TO_PASS")
        if not isinstance(fail_to_pass, list) or not fail_to_pass or not all(
            isinstance(item, str) and item for item in fail_to_pass
        ):
            raise ValueError("FAIL_TO_PASS must be a non-empty string list")
        if not isinstance(pass_to_pass, list) or not all(
            isinstance(item, str) and item for item in pass_to_pass
        ):
            raise ValueError("PASS_TO_PASS must be a string list")
        return cls(
            instance_id=value["instance_id"],
            patch=value["patch"],
            fail_to_pass=tuple(fail_to_pass),
            pass_to_pass=tuple(pass_to_pass),
            image_name=value["image_name"],
            repo=value["repo"],
            problem_statement=value["problem_statement"],
        )
    @property
    def source_repo_key(self) -> str:
        profile_name = self.repo.rsplit("/", 1)[-1]
        return profile_name.split(".", 1)[0]

    @property
    def gold_files(self) -> tuple[str, ...]:
        files: set[str] = set()
        for line in self.patch.splitlines():
            if not line.startswith("+++ "):
                continue
            path = line[4:].split("\t", 1)[0].strip().strip('"')
            if path == "/dev/null":
                continue
            if path.startswith("b/"):
                path = path[2:]
            if path and not path.startswith("/") and ".." not in Path(path).parts:
                files.add(path.replace("\\", "/"))
        if not files:
            raise ValueError(f"Task {self.instance_id} patch has no safe gold file")
        return tuple(sorted(files))

    def inventory_entry(self, split: str, dataset_revision: str) -> dict[str, Any]:
        return {
            "schema_version": "2.0.0",
            "task_id": self.instance_id,
            "split": split,
            "source_repo_key": self.source_repo_key,
            "repo_profile": self.repo,
            "image_name": self.image_name,
            "problem_statement": self.problem_statement,
            "issue_hash": sha256_text(self.problem_statement),
            "patch_hash": sha256_text(self.patch),
            "fail_to_pass_count": len(self.fail_to_pass),
            "pass_to_pass_count": len(self.pass_to_pass),
            "dataset_revision": dataset_revision,
        }

    def oracle_entry(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0.0",
            "task_id": self.instance_id,
            "patch": self.patch,
            "gold_files": list(self.gold_files),
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
        }

    def to_upstream_dict(self) -> dict[str, Any]:
        """Return the exact field layout expected by the pinned SWE-smith API."""
        return {
            "instance_id": self.instance_id,
            "patch": self.patch,
            "FAIL_TO_PASS": list(self.fail_to_pass),
            "PASS_TO_PASS": list(self.pass_to_pass),
            "image_name": self.image_name,
            "repo": self.repo,
            "problem_statement": self.problem_statement,
        }

    def to_task_spec(self, split: str, base_commit: str) -> TaskSpec:
        if not base_commit:
            raise ValueError("base_commit must come from a materialized container")
        return TaskSpec(
            task_id=self.instance_id,
            split=split,
            repo_id=self.source_repo_key,
            issue=self.problem_statement,
            base_commit=base_commit,
        )


@dataclass(frozen=True)
class SelectedSweSmithTask:
    task: UpstreamSweSmithTask
    split: str
    issue_hash: str


def load_swesmith_environment_config(path: Path) -> SweSmithEnvironmentConfig:
    raw = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SWE-smith environment config must be a mapping")
    if raw.get("schema_version") != "2.0.0":
        raise ValueError("environment config must use schema 2.0.0")
    profile = raw.get("profile")
    if profile not in {"swesmith_wsl_v1", "swesmith_linux_server_v1"}:
        raise ValueError("unsupported SWE-smith environment profile")
    wsl = raw.get("wsl", {})
    docker = raw.get("docker", {})
    storage = raw.get("storage", {})
    source = raw.get("source", {})
    selection = raw.get("selection", {})
    revision = source.get("repository_revision")
    dataset_revision = source.get("dataset_revision")
    if revision != PINNED_SWESMITH_REPOSITORY_REVISION:
        raise ValueError("SWE-smith repository revision is not frozen")
    if dataset_revision != PINNED_SWESMITH_DATASET_REVISION:
        raise ValueError("SWE-smith dataset revision is not frozen")
    server = raw.get("server", {})
    runtime = wsl if profile == "swesmith_wsl_v1" else server
    distro = wsl.get("distro", "")
    if profile == "swesmith_wsl_v1" and (
        not isinstance(distro, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", distro)
    ):
        raise ValueError("wsl.distro contains unsupported characters")
    minimum_python = str(runtime.get("minimum_python", "")).split(".")
    if len(minimum_python) != 2 or not all(item.isdigit() for item in minimum_python):
        raise ValueError("minimum_python must be major.minor")
    counts = selection.get("counts")
    expected_splits = {"smoke", "train", "dev", "test"}
    if not isinstance(counts, dict) or set(counts) != expected_splits:
        raise ValueError("selection.counts must define smoke/train/dev/test")
    if any(not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("selection counts must be positive integers")
    docker_data_root: str | None = None
    dependencies_root: str | None = None
    runtime_root_env: str | None = None
    if profile == "swesmith_wsl_v1":
        docker_data_root = storage.get("docker_data_root")
        if not isinstance(docker_data_root, str) or not re.match(
            r"^[Dd]:\\", docker_data_root
        ):
            raise ValueError("storage.docker_data_root must be an absolute D-drive path")
        dependencies_root = storage.get("dependencies_root")
        if not isinstance(dependencies_root, str) or not re.match(
            r"^[Dd]:\\", dependencies_root
        ):
            raise ValueError("storage.dependencies_root must be an absolute D-drive path")
    else:
        runtime_root_env = storage.get("runtime_root_env")
        if not isinstance(runtime_root_env, str) or not re.fullmatch(
            r"[A-Z][A-Z0-9_]*", runtime_root_env
        ):
            raise ValueError("storage.runtime_root_env must be an uppercase environment name")
    minimum_free_gib = storage.get("minimum_free_gib")
    if not isinstance(minimum_free_gib, int) or minimum_free_gib < 1:
        raise ValueError("storage.minimum_free_gib must be a positive integer")
    hardware = raw.get("hardware", {})
    minimum_gpu_memory_gib = int(hardware.get("minimum_gpu_memory_gib", 0))
    if profile == "swesmith_linux_server_v1" and minimum_gpu_memory_gib < 1:
        raise ValueError("hardware.minimum_gpu_memory_gib must be positive")
    return SweSmithEnvironmentConfig(
        profile=str(profile),
        platform="windows_wsl2" if profile == "swesmith_wsl_v1" else "linux_server",
        distro=distro,
        minimum_python=(int(minimum_python[0]), int(minimum_python[1])),
        require_wsl2=bool(wsl.get("require_wsl2")),
        require_linux_engine=bool(docker.get("require_linux_engine")),
        require_distro_integration=bool(docker.get("require_distro_integration")),
        docker_data_root=docker_data_root,
        dependencies_root=dependencies_root,
        runtime_root_env=runtime_root_env,
        require_native_linux=bool(server.get("require_native_linux")),
        require_cuda=bool(hardware.get("require_cuda")),
        minimum_gpu_memory_gib=minimum_gpu_memory_gib,
        minimum_free_gib=minimum_free_gib,
        repository=str(source.get("repository")),
        repository_revision=revision,
        dataset_id=str(source.get("dataset_id")),
        dataset_revision=dataset_revision,
        hash_seed=int(selection.get("hash_seed")),
        max_tasks_per_repository=int(selection.get("max_tasks_per_repository")),
        counts={key: int(value) for key, value in counts.items()},
    )


def resolve_server_runtime_root(
    config: SweSmithEnvironmentConfig,
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> Path:
    if config.platform != "linux_server" or not config.runtime_root_env:
        raise ValueError("a linux_server environment config is required")
    values = os.environ if environ is None else environ
    raw = values.get(config.runtime_root_env, "").strip()
    if not raw:
        raise ValueError(f"{config.runtime_root_env} is not set")
    if re.match(r"^[A-Za-z]:[\\/]", raw) or "\\" in raw:
        raise ValueError("server runtime root must not be a Windows path")
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError("server runtime root must be an absolute Linux path")
    resolved = root.resolve(strict=False)
    if resolved in {Path("/"), Path("/home"), Path("/root"), Path("/usr"), Path("/var")}:
        raise ValueError("server runtime root is too broad")
    return resolved


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def preflight_linux_server_environment(
    config: SweSmithEnvironmentConfig,
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"profile": config.profile}
    errors: list[str] = []
    if config.platform != "linux_server":
        raise ValueError("server preflight requires swesmith_linux_server_v1")

    checks["native_linux"] = platform.system() == "Linux"
    if not checks["native_linux"]:
        errors.append("Formal experiments must run on a native Linux GPU server")
        return {
            "schema_version": "2.0.0",
            "profile": config.profile,
            "checks": checks,
            "details": details,
            "errors": errors,
            "ready": False,
        }

    kernel_text = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    details["kernel"] = kernel_text.strip()
    checks["not_wsl"] = "microsoft" not in kernel_text.lower()
    if config.require_native_linux and not checks["not_wsl"]:
        errors.append("WSL is not accepted for the formal server profile")

    try:
        runtime_root = resolve_server_runtime_root(config, environ=environ)
        checks["runtime_root"] = True
        details["runtime_root"] = str(runtime_root)
        usage = shutil.disk_usage(_nearest_existing_parent(runtime_root))
        free_gib = usage.free / (1024**3)
        checks["storage_free_space"] = free_gib >= config.minimum_free_gib
        details["storage_free_gib"] = round(free_gib, 2)
        details["storage_minimum_free_gib"] = config.minimum_free_gib
        if not checks["storage_free_space"]:
            errors.append(
                f"Server runtime volume has {free_gib:.2f} GiB free; "
                f"{config.minimum_free_gib} GiB is required"
            )
    except (OSError, ValueError) as exc:
        checks["runtime_root"] = False
        checks["storage_free_space"] = False
        errors.append(str(exc))

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(command, 127, "", str(exc))

    python = run(["python3", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
    details["python_version"] = python.stdout.strip() or None
    try:
        version = tuple(int(item) for item in python.stdout.strip().split(".", 1))
    except ValueError:
        version = (0, 0)
    checks["python_version"] = python.returncode == 0 and version >= config.minimum_python
    if not checks["python_version"]:
        errors.append("Server Python does not meet the frozen minimum version")

    git = run(["git", "--version"])
    checks["git"] = git.returncode == 0
    details["git"] = git.stdout.strip() or None
    if not checks["git"]:
        errors.append("git is unavailable on the server")

    docker = run(["docker", "info", "--format", "{{.OSType}}|{{.Architecture}}|{{.ServerVersion}}"])
    details["docker_engine"] = docker.stdout.strip() or None
    checks["docker_engine"] = docker.returncode == 0
    checks["linux_engine"] = docker.returncode == 0 and docker.stdout.strip().startswith("linux|")
    if not checks["docker_engine"]:
        errors.append("Docker Engine is not reachable on the server")
    elif config.require_linux_engine and not checks["linux_engine"]:
        errors.append("Docker Engine is not in Linux container mode")

    gpu = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    memories: list[int] = []
    if gpu.returncode == 0:
        try:
            memories = [int(line.strip()) for line in gpu.stdout.splitlines() if line.strip()]
        except ValueError:
            memories = []
    details["gpu_memory_mib"] = memories
    checks["cuda_gpu"] = bool(memories)
    # nvidia-smi reports slightly below the marketed capacity on some 24 GB cards.
    checks["gpu_memory"] = bool(memories) and max(memories) >= config.minimum_gpu_memory_gib * 1000
    if config.require_cuda and not checks["cuda_gpu"]:
        errors.append("nvidia-smi did not report a CUDA GPU")
    elif not checks["gpu_memory"]:
        errors.append(
            f"No GPU meets the {config.minimum_gpu_memory_gib} GiB server minimum"
        )

    return {
        "schema_version": "2.0.0",
        "profile": config.profile,
        "checks": checks,
        "details": details,
        "errors": errors,
        "ready": all(checks.values()),
    }


def preflight_wsl_environment(config: SweSmithEnvironmentConfig) -> dict[str, Any]:
    # The local Windows/WSL path is retained only for historical config parsing.
    # Refuse before running WSL commands or inspecting any Windows drive.
    return {
        "schema_version": "2.0.0",
        "profile": config.profile,
        "status": "refused",
        "checks": {},
        "details": {},
        "errors": [
            "Local Windows/WSL preflight is disabled; formal experiments run on "
            "the rented Linux server"
        ],
        "ready": False,
    }

def select_repository_disjoint_tasks(
    tasks: Iterable[UpstreamSweSmithTask],
    counts: dict[str, int],
    *,
    hash_seed: int = 42,
    max_tasks_per_repository: int = 10,
) -> dict[str, list[UpstreamSweSmithTask]]:
    if max_tasks_per_repository <= 0:
        raise ValueError("max_tasks_per_repository must be positive")
    groups: dict[str, list[UpstreamSweSmithTask]] = {}
    for task in tasks:
        groups.setdefault(task.source_repo_key, []).append(task)
    ordered_repos = sorted(
        groups,
        key=lambda repo: sha256_text(f"{hash_seed}:repo:{repo}"),
    )
    for repo, values in groups.items():
        groups[repo] = sorted(
            values,
            key=lambda task: sha256_text(f"{hash_seed}:task:{task.instance_id}"),
        )

    selected: dict[str, list[UpstreamSweSmithTask]] = {}
    used_repos: set[str] = set()
    for split, required_count in counts.items():
        if required_count <= 0:
            raise ValueError(f"count for {split} must be positive")
        split_tasks: list[UpstreamSweSmithTask] = []
        for repo in ordered_repos:
            if repo in used_repos:
                continue
            remaining = required_count - len(split_tasks)
            if remaining <= 0:
                break
            take = min(len(groups[repo]), max_tasks_per_repository, remaining)
            if take:
                split_tasks.extend(groups[repo][:take])
                used_repos.add(repo)
        if len(split_tasks) != required_count:
            raise ValueError(
                f"Unable to select {required_count} repository-disjoint tasks for {split}; "
                f"selected {len(split_tasks)}"
            )
        selected[split] = split_tasks
    return selected


def build_swesmith_inventory(
    source_jsonl: Path,
    inventory_output: Path,
    oracle_output: Path,
    config: SweSmithEnvironmentConfig,
) -> dict[str, Any]:
    tasks = [UpstreamSweSmithTask.from_dict(value) for value in read_jsonl(source_jsonl)]
    selected = select_repository_disjoint_tasks(
        tasks,
        config.counts,
        hash_seed=config.hash_seed,
        max_tasks_per_repository=config.max_tasks_per_repository,
    )
    registry = SchemaRegistry()
    inventory: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for split, split_tasks in selected.items():
        for task in split_tasks:
            inventory_entry = task.inventory_entry(split, config.dataset_revision)
            oracle_entry = task.oracle_entry()
            registry.validate("swe_smith_inventory", inventory_entry)
            registry.validate("swe_smith_oracle", oracle_entry)
            inventory.append(inventory_entry)
            oracle.append(oracle_entry)
    write_jsonl(inventory_output, inventory)
    write_jsonl(oracle_output, oracle)
    return {
        "schema_version": "2.0.0",
        "inventory": str(inventory_output.resolve()),
        "oracle": str(oracle_output.resolve()),
        "counts": {split: len(values) for split, values in selected.items()},
        "repositories": {
            split: len({task.source_repo_key for task in values})
            for split, values in selected.items()
        },
    }


def load_selected_swesmith_tasks(
    inventory_path: Path,
    oracle_path: Path,
    *,
    split: str | None = None,
) -> list[SelectedSweSmithTask]:
    registry = SchemaRegistry()
    inventory_by_id: dict[str, dict[str, Any]] = {}
    oracle_by_id: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(inventory_path):
        registry.validate("swe_smith_inventory", value)
        task_id = value["task_id"]
        if task_id in inventory_by_id:
            raise ValueError(f"Duplicate inventory task_id: {task_id}")
        inventory_by_id[task_id] = value
    for value in read_jsonl(oracle_path):
        registry.validate("swe_smith_oracle", value)
        task_id = value["task_id"]
        if task_id in oracle_by_id:
            raise ValueError(f"Duplicate oracle task_id: {task_id}")
        oracle_by_id[task_id] = value
    if set(inventory_by_id) != set(oracle_by_id):
        missing_oracle = sorted(set(inventory_by_id) - set(oracle_by_id))
        missing_inventory = sorted(set(oracle_by_id) - set(inventory_by_id))
        raise ValueError(
            "Inventory/oracle task sets differ: "
            f"missing_oracle={missing_oracle}, missing_inventory={missing_inventory}"
        )

    selected: list[SelectedSweSmithTask] = []
    for task_id in sorted(inventory_by_id):
        inventory = inventory_by_id[task_id]
        if split is not None and inventory["split"] != split:
            continue
        oracle = oracle_by_id[task_id]
        if sha256_text(oracle["patch"]) != inventory["patch_hash"]:
            raise ValueError(f"Patch hash mismatch for task {task_id}")
        if len(oracle["fail_to_pass"]) != inventory["fail_to_pass_count"]:
            raise ValueError(f"FAIL_TO_PASS count mismatch for task {task_id}")
        if len(oracle["pass_to_pass"]) != inventory["pass_to_pass_count"]:
            raise ValueError(f"PASS_TO_PASS count mismatch for task {task_id}")
        task = UpstreamSweSmithTask.from_dict(
            {
                "instance_id": task_id,
                "patch": oracle["patch"],
                "FAIL_TO_PASS": oracle["fail_to_pass"],
                "PASS_TO_PASS": oracle["pass_to_pass"],
                "image_name": inventory["image_name"],
                "repo": inventory["repo_profile"],
                "problem_statement": inventory["problem_statement"],
            }
        )
        if task.source_repo_key != inventory["source_repo_key"]:
            raise ValueError(f"Repository key mismatch for task {task_id}")
        selected.append(
            SelectedSweSmithTask(
                task=task,
                split=inventory["split"],
                issue_hash=inventory["issue_hash"],
            )
        )
    if not selected:
        raise ValueError(f"No SWE-smith tasks selected for split={split!r}")
    return selected


def fetch_swesmith_source(
    output_path: Path,
    config: SweSmithEnvironmentConfig,
    *,
    dataset_loader: Any | None = None,
) -> dict[str, Any]:
    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the optional data dependencies") from exc
        dataset_loader = load_dataset
    dataset = dataset_loader(
        config.dataset_id,
        split="train",
        revision=config.dataset_revision,
        trust_remote_code=False,
    )
    count = 0

    def validated_rows() -> Iterable[dict[str, Any]]:
        nonlocal count
        for raw in dataset:
            task = UpstreamSweSmithTask.from_dict(dict(raw))
            count += 1
            yield task.to_upstream_dict()

    write_jsonl(output_path, validated_rows())
    digest = hashlib.sha256()
    with output_path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": "2.0.0",
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "split": "train",
        "rows": count,
        "output": str(output_path.resolve()),
        "file_sha256": f"sha256:{digest.hexdigest()}",
    }
