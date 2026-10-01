from __future__ import annotations

import json
import multiprocessing
import os
import queue
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..hashing import sha256_text
from ..harness.checkpoint import _combined_state_hash, read_recovery_state
from ..harness.environment import RepositoryEnvironment, RepositoryProfile
from ..harness.runner import run_api_repository
from .events import EventBus
from .models import RunRequest
from .store import RunStore

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
Runner = Callable[..., dict[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _process_worker(runner_kwargs: dict[str, Any], event_queue: Any, result_queue: Any, cancel_event: Any) -> None:
    """Run the default Agent outside the FastAPI process."""
    def sink(event_type: str, payload: dict[str, Any]) -> None:
        event_queue.put({"kind": "event", "type": event_type, "payload": payload})
    try:
        result = run_api_repository(
            **runner_kwargs, event_sink=sink, should_cancel=cancel_event.is_set,
        )
        result_queue.put({"kind": "result", "value": result})
    except BaseException as exc:  # noqa: BLE001
        result_queue.put({"kind": "error", "type": type(exc).__name__, "message": str(exc)})


@dataclass
class ManagedRun:
    run_id: str
    request: RunRequest
    source_repo: Path
    repo_path: Path
    run_dir: Path
    bus: EventBus
    cancel_event: Any
    status: str = "queued"
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    attempt_count: int = 0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now()

    def public(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "status": self.status,
            "task_id": self.request.task_id, "repo_path": str(self.repo_path),
            "source_repo": str(self.source_repo), "base_ref": self.request.base_ref,
            "issue": self.request.issue, "harness_policy": self.request.harness_policy,
            "model_policy": self.request.model_policy,
            "workspace_backend": self.request.workspace_backend,
            "execution_profile": self.request.execution_profile,
            "attempt_count": self.attempt_count,
            "run_dir": str(self.run_dir) if self.run_dir.exists() else None,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "error": self.error, "result": self.result,
        }


class RunManager:
    """Durable, single-concurrency control plane for local Agent runs."""

    def __init__(self, *, allowed_roots: tuple[Path, ...] | None = None,
                 artifact_root: Path | None = None, api_config: Path | None = None,
                 runner: Runner = run_api_repository) -> None:
        self.allowed_roots = tuple(root.resolve() for root in (allowed_roots or (Path.cwd(),)))
        self.artifact_root = (artifact_root or Path.cwd() / "artifacts" / "web-runs").resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.api_config = (api_config or Path("configs/model/api_policy.yaml")).resolve()
        self.runner = runner
        self.store = RunStore(self.artifact_root / "control.sqlite3")
        self._lock = threading.RLock()
        self._runs: dict[str, ManagedRun] = {}
        self._processes: dict[str, Any] = {}
        self._hydrate()

    @classmethod
    def from_environment(cls) -> "RunManager":
        raw_roots = os.getenv("CAPO_WEB_ALLOWED_ROOTS")
        roots = tuple(Path(item) for item in raw_roots.split(os.pathsep) if item.strip()) if raw_roots else None
        artifact_root = os.getenv("CAPO_WEB_ARTIFACT_ROOT")
        api_config = os.getenv("CAPO_WEB_API_CONFIG")
        return cls(allowed_roots=roots, artifact_root=Path(artifact_root) if artifact_root else None,
                   api_config=Path(api_config) if api_config else None)

    def _hydrate(self) -> None:
        for raw in self.store.all():
            request = RunRequest.model_validate(raw["request"])
            bus = EventBus(journal_path=Path(raw["run_dir"]) / "events.jsonl", run_id=raw["run_id"])
            status, error = raw["status"], raw.get("error")
            if status not in TERMINAL_STATES:
                status, error = "failed", "Control plane restarted while the worker was active"
                self.store.update(raw["run_id"], status=status, error=error, finished_at=_now())
            cancel_event = (
                multiprocessing.get_context("spawn").Event()
                if self.runner is run_api_repository
                else threading.Event()
            )
            self._runs[raw["run_id"]] = ManagedRun(
                run_id=raw["run_id"], request=request,
                source_repo=Path(request.source_repo or request.repo_path or raw["repo_path"]),
                repo_path=Path(raw["repo_path"]), run_dir=Path(raw["run_dir"]), bus=bus,
                cancel_event=cancel_event, status=status, created_at=raw["created_at"],
                started_at=raw.get("started_at"), finished_at=raw.get("finished_at"),
                error=error, result=raw.get("result"), attempt_count=int(raw.get("attempt_count", 0)),
            )

    def _resolve_repo(self, raw_path: str) -> Path:
        repo = Path(raw_path).expanduser().resolve()
        if not repo.is_dir() or not (repo / ".git").exists():
            raise ValueError("source_repo must be an existing Git worktree")
        if not any(repo == root or root in repo.parents for root in self.allowed_roots):
            raise ValueError("source_repo is outside CAPO_WEB_ALLOWED_ROOTS")
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                                text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
        if status.returncode != 0:
            raise ValueError(status.stderr.strip() or "Unable to inspect Git worktree")
        if status.stdout.strip():
            raise ValueError("Repository worktree is dirty; provide a clean isolated worktree")
        return repo

    def _create_workspace(self, source: Path, run_id: str, base_ref: str) -> Path:
        workspace = self.artifact_root / "workspaces" / run_id
        workspace.parent.mkdir(parents=True, exist_ok=True)
        _git(source, "rev-parse", base_ref)
        _git(source, "worktree", "add", "--detach", str(workspace), base_ref)
        return workspace

    def _active_run(self) -> ManagedRun | None:
        return next((run for run in self._runs.values() if run.status not in TERMINAL_STATES), None)

    def _persist(self, run: ManagedRun, **fields: Any) -> None:
        self.store.update(run.run_id, **fields)

    def start(self, request: RunRequest) -> ManagedRun:
        if request.workspace_backend == "docker":
            raise ValueError("Docker workspace backend is available only on the Linux runtime profile")
        with self._lock:
            if request.idempotency_key:
                existing = self.store.find_idempotency(request.idempotency_key)
                if existing:
                    return self._runs[existing["run_id"]]
            active = self._active_run()
            if active is not None:
                raise RuntimeError(f"A run is already active: {active.run_id}")
            if not self.api_config.is_file():
                raise ValueError(f"API policy config does not exist: {self.api_config}")
            source = self._resolve_repo(request.source_repo or request.repo_path or "")
            run_id = f"web_{uuid.uuid4().hex[:16]}"
            run_dir = self.artifact_root / run_id
            workspace = self._create_workspace(source, run_id, request.base_ref)
            bus = EventBus(journal_path=run_dir / "events.jsonl", run_id=run_id)
            context = multiprocessing.get_context("spawn")
            cancel_event = context.Event() if self.runner is run_api_repository else threading.Event()
            run = ManagedRun(run_id, request, source, workspace, run_dir, bus, cancel_event,
                             attempt_count=1)
            self._runs[run_id] = run
            self.store.create({
                "run_id": run_id, "status": run.status, "request": request.model_dump(mode="json"),
                "repo_path": str(workspace), "run_dir": str(run_dir), "task_id": request.task_id,
                "issue": request.issue, "harness_policy": request.harness_policy,
                "created_at": run.created_at, "idempotency_key": request.idempotency_key,
                "attempt_count": run.attempt_count,
            })
            self._write_json(run.run_dir / "workspace.json", {
                "schema_version": "1.0.0", "source_repo": str(source),
                "workspace": str(workspace), "base_ref": request.base_ref,
                "backend": request.workspace_backend,
            })
            self._write_json(run.run_dir / "runtime.json", {
                "schema_version": "1.0.0", "backend": request.workspace_backend,
                "image_digest": os.getenv("CAPO_RUNTIME_IMAGE_DIGEST"),
                "sdk_commit": os.getenv("CAPO_SDK_COMMIT"),
                "worker": "spawned_process" if self.runner is run_api_repository else "thread",
            })
            run.bus.publish("run_queued", run.public())
            self._launch(run, attempt_id="attempt_0001")
            return run

    def _runner_kwargs(self, run: ManagedRun, *, resume_state: dict[str, Any] | None = None,
                       attempt_id: str | None = None) -> dict[str, Any]:
        return {"repo": run.repo_path, "issue": run.request.issue, "run_dir": run.run_dir,
                "api_config": self.api_config, "task_id": run.request.task_id,
                "harness_policy": run.request.harness_policy, "allow_dirty": resume_state is not None,
                "resume_state": resume_state, "attempt_id": attempt_id}

    def _launch(self, run: ManagedRun, *, resume_state: dict[str, Any] | None = None,
                attempt_id: str | None = None) -> None:
        attempt_id = attempt_id or f"attempt_{max(run.attempt_count, 1):04d}"
        run.status, run.started_at = "running", _now()
        self._persist(run, status=run.status, started_at=run.started_at)
        run.bus.publish("run_started", run.public())
        attempt = {"schema_version": "1.0.0", "attempt_id": attempt_id,
                   "run_id": run.run_id, "status": "running", "started_at": run.started_at,
                   "resumed_from_step": resume_state.get("next_step_index") if resume_state else None}
        self._append_jsonl(run.run_dir / "attempts.jsonl", attempt)
        run.bus.publish("attempt_started", attempt)
        if self.runner is run_api_repository:
            context = multiprocessing.get_context("spawn")
            events, results = context.Queue(), context.Queue()
            process = context.Process(target=_process_worker,
                                      args=(self._runner_kwargs(run, resume_state=resume_state, attempt_id=attempt_id),
                                            events, results, run.cancel_event),
                                      name=f"capo-{run.run_id}", daemon=True)
            process.start()
            self._processes[run.run_id] = process
            threading.Thread(target=self._watch_process, args=(run, process, events, results),
                             name=f"capo-watch-{run.run_id}", daemon=True).start()
        else:
            threading.Thread(target=self._run_in_thread, args=(run, resume_state, attempt_id),
                             name=f"capo-{run.run_id}", daemon=True).start()

    def _run_in_thread(self, run: ManagedRun, resume_state: dict[str, Any] | None,
                       attempt_id: str | None) -> None:
        try:
            result = self.runner(**self._runner_kwargs(run, resume_state=resume_state, attempt_id=attempt_id),
                                 event_sink=run.bus.publish, should_cancel=run.cancel_event.is_set)
            self._finish(run, result)
        except Exception as exc:  # noqa: BLE001
            self._fail(run, exc)

    def _watch_process(self, run: ManagedRun, process: Any, events: Any, results: Any) -> None:
        result: dict[str, Any] | None = None
        error: Exception | None = None
        while process.is_alive() or not results.empty():
            try:
                while True:
                    message = events.get_nowait()
                    if message.get("kind") == "event":
                        run.bus.publish(message["type"], message["payload"], actor="runtime")
            except queue.Empty:
                pass
            try:
                message = results.get(timeout=0.05)
                if message.get("kind") == "result":
                    result = message["value"]
                else:
                    error = RuntimeError(f"{message.get('type')}: {message.get('message')}")
            except queue.Empty:
                continue
        process.join(timeout=1)
        if result is None and error is None:
            try:
                message = results.get(timeout=1)
                if message.get("kind") == "result":
                    result = message["value"]
                else:
                    error = RuntimeError(f"{message.get('type')}: {message.get('message')}")
            except queue.Empty:
                pass
        try:
            while True:
                message = events.get_nowait()
                if message.get("kind") == "event":
                    run.bus.publish(message["type"], message["payload"], actor="runtime")
        except queue.Empty:
            pass
        self._processes.pop(run.run_id, None)
        if result is not None:
            self._finish(run, result)
        elif error is not None:
            self._fail(run, error)
        else:
            self._fail(run, RuntimeError("Agent worker exited without a result"))

    def _finish(self, run: ManagedRun, result: dict[str, Any]) -> None:
        with self._lock:
            episode = result.get("episode", {})
            termination = episode.get("termination_reason")
            run.result = result
            run.status = "cancelled" if termination == "CANCELLED" else "completed" if episode.get("resolved") else "failed"
            run.finished_at = _now()
            self._append_jsonl(run.run_dir / "attempts.jsonl", {
                "schema_version": "1.0.0", "attempt_id": episode.get("attempt_id"),
                "run_id": run.run_id, "status": run.status, "finished_at": run.finished_at,
                "resolved": bool(episode.get("resolved")),
            })
            run.bus.publish("web_run_finished", run.public())
            self._persist(run, status=run.status, finished_at=run.finished_at, result=result, error=None)

    def _fail(self, run: ManagedRun, exc: Exception) -> None:
        with self._lock:
            run.status = "cancelled" if run.cancel_event.is_set() else "failed"
            run.error, run.finished_at = str(exc), _now()
            self._append_jsonl(run.run_dir / "attempts.jsonl", {
                "schema_version": "1.0.0", "attempt_id": f"attempt_{max(run.attempt_count, 1):04d}",
                "run_id": run.run_id, "status": run.status, "finished_at": run.finished_at,
                "error": run.error,
            })
            run.bus.publish("web_run_failed", {"error": run.error, "status": run.status})
            self._persist(run, status=run.status, finished_at=run.finished_at, error=run.error)

    def get(self, run_id: str) -> ManagedRun:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise KeyError(run_id) from exc

    def cancel(self, run_id: str) -> ManagedRun:
        run = self.get(run_id)
        if run.status not in TERMINAL_STATES:
            run.cancel_event.set()
            run.bus.publish("cancel_requested", {"run_id": run_id})
            process = self._processes.get(run_id)
            if process is not None and process.is_alive():
                threading.Thread(
                    target=self._force_terminate,
                    args=(run, process),
                    name=f"capo-cancel-{run_id}",
                    daemon=True,
                ).start()
        return run

    @staticmethod
    def _force_terminate(run: ManagedRun, process: Any) -> None:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        values: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    values.append(value)
        return values

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def _resume_state(self, run: ManagedRun) -> dict[str, Any]:
        metadata = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        checkpoints = self._jsonl(run.run_dir / "checkpoints.jsonl")
        steps = self._jsonl(run.run_dir / "steps.jsonl")
        if not checkpoints or not steps:
            raise ValueError("Run has no durable checkpoint to resume")
        latest = checkpoints[-1]
        episode_steps = [step for step in steps if step.get("episode_id") == latest["episode_id"]
                         and step.get("step_index", -1) <= latest["step_index"]]
        latest_step = max(episode_steps, key=lambda value: value["step_index"])
        current_diff = sha256_text(_git(run.repo_path, "diff", "--binary", "--no-ext-diff"))
        if current_diff != latest["diff_hash"]:
            raise ValueError("Current workspace diff does not match the latest checkpoint")
        profile = RepositoryProfile(tuple(metadata["config"]["repository"]["test_command"]),
                                    tuple(metadata["config"]["repository"]["default_test_args"]),
                                    int(metadata["config"]["harness"]["tool_timeout_seconds"]))
        environment = RepositoryEnvironment(run.repo_path, profile)
        if latest_step.get("test_state"):
            from ..models import TestState, TestStatus
            value = latest_step["test_state"]
            environment.last_test_state = TestState(TestStatus(value["status"]),
                                                     value.get("fail_to_pass_count"),
                                                     value.get("pass_to_pass_failures"))
        if _combined_state_hash(environment, metadata["config"]["harness"], episode_steps,
                                budget_used=latest.get("budget_used"),
                                legacy=latest.get("state_contract") != "workspace-v1") != latest["state_hash_after"]:
            raise ValueError("Current workspace state does not match the latest checkpoint")
        used = latest.get("budget_used", {})
        return {
            "episode_id": latest["episode_id"], "next_step_index": latest["next_step_index"],
            "input_tokens": used.get("input_tokens", sum(int(step.get("input_tokens", 0)) for step in episode_steps)),
            "output_tokens": used.get("output_tokens", sum(int(step.get("output_tokens", 0)) for step in episode_steps)),
            "test_calls": used.get("test_calls", sum(step.get("action", {}).get("type") == "run_tests" for step in episode_steps)),
            "steps_used": used.get("steps", len(episode_steps)),
            "elapsed_seconds": used.get("elapsed_seconds", 0),
            "recovery_state": read_recovery_state(run.run_dir, latest),
            "history": [{"assistant": step.get("action", {}), "observation": step.get("tool_result", {})}
                        for step in episode_steps[-8:]],
            "test_state": latest_step.get("test_state"),
        }

    def resume(self, run_id: str) -> ManagedRun:
        run = self.get(run_id)
        if run.status not in {"failed", "cancelled"}:
            raise ValueError("Only failed or cancelled runs can be resumed")
        if self._active_run() is not None:
            raise RuntimeError("A run is already active")
        state = self._resume_state(run)
        run.attempt_count += 1
        attempt_id = f"attempt_{run.attempt_count:04d}"
        self._persist(run, status="running", attempt_count=run.attempt_count, error=None)
        run.bus.publish("resume_requested", {"attempt_id": attempt_id, "resumed_from_step": state["next_step_index"]})
        self._launch(run, resume_state=state, attempt_id=attempt_id)
        return run

    def artifacts(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        root = run.run_dir.resolve()
        if not root.exists():
            return {"run_id": run_id, "available": [], "run": run.public()}
        metadata_path = root / "run.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        steps, episodes = self._jsonl(root / "steps.jsonl"), self._jsonl(root / "episodes.jsonl")
        checkpoints, events = self._jsonl(root / "checkpoints.jsonl"), self._jsonl(root / "events.jsonl")
        attempts = self._jsonl(root / "attempts.jsonl")
        evaluator: dict[str, Any] = {}
        if (root / "evaluator").is_dir():
            for path in sorted((root / "evaluator").glob("*.json")):
                evaluator[path.name] = json.loads(path.read_text(encoding="utf-8"))
        return {"run_id": run_id, "available": [name for name, exists in {
            "run.json": metadata_path.is_file(), "events.jsonl": bool(events),
            "workspace.json": (root / "workspace.json").is_file(),
            "runtime.json": (root / "runtime.json").is_file(),
            "steps.jsonl": (root / "steps.jsonl").is_file(), "episodes.jsonl": (root / "episodes.jsonl").is_file(),
            "checkpoints.jsonl": (root / "checkpoints.jsonl").is_file(), "attempts.jsonl": bool(attempts),
            "evaluator": bool(evaluator),
        }.items() if exists], "run": metadata, "steps": steps, "episodes": episodes,
                "checkpoints": checkpoints, "attempts": attempts, "events": events, "evaluator": evaluator}

    def trace(self, run_id: str) -> dict[str, Any]:
        artifacts = self.artifacts(run_id)
        checkpoints = artifacts.get("checkpoints", [])
        timeline = []
        for step in artifacts.get("steps", []):
            checkpoint = next((item for item in checkpoints if item.get("episode_id") == step.get("episode_id")
                               and item.get("step_index") == step.get("step_index")), None)
            timeline.append({"kind": "step", "step_index": step.get("step_index"),
                             "episode_id": step.get("episode_id"), "action": step.get("action"),
                             "tool_result": step.get("tool_result"), "test_state": step.get("test_state"),
                             "latency_ms": step.get("latency_ms", 0), "tool_latency_ms": step.get("tool_latency_ms", 0),
                             "input_tokens": step.get("input_tokens", 0), "output_tokens": step.get("output_tokens", 0),
                             "labels": step.get("automatic_labels", []), "checkpoint": checkpoint})
        return {"run": self.get(run_id).public(), "attempts": artifacts.get("attempts", []),
                "timeline": timeline, "events": artifacts.get("events", []),
                "summary": {"steps": len(artifacts.get("steps", [])),
                             "episodes": len(artifacts.get("episodes", [])),
                             "checkpoints": len(checkpoints)},
                "artifacts": {"available": artifacts.get("available", []), "run_dir": str(self.get(run_id).run_dir)}}
