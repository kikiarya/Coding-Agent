from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..models import EpisodeRecord, StepRecord, ToolResult
from ..schema import SchemaRegistry
from ..hashing import sha256_json


class RunArtifactStore:
    def __init__(
        self,
        run_dir: Path,
        schemas: SchemaRegistry | None = None,
        *,
        allow_existing: bool = False,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.schemas = schemas or SchemaRegistry()
        self.run_dir.mkdir(parents=True, exist_ok=allow_existing)
        (self.run_dir / "observations").mkdir(exist_ok=True)
        (self.run_dir / "evaluator").mkdir(exist_ok=True)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def write_recovery_state(self, value: dict[str, Any]) -> str:
        digest = sha256_json(value)
        directory = self.run_dir / "recovery"
        directory.mkdir(exist_ok=True)
        path = directory / (digest.removeprefix("sha256:") + ".json")
        if not path.exists():
            self._write_json(path, value)
        return digest

    @staticmethod
    def _append_jsonl(path: Path, value: Any) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_event(self, value: dict[str, Any]) -> None:
        self._append_jsonl(self.run_dir / "events.jsonl", value)

    def write_attempt(self, value: dict[str, Any]) -> None:
        self._append_jsonl(self.run_dir / "attempts.jsonl", value)

    def write_run_metadata(self, value: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "run.json", value)

    def write_observation(self, result: ToolResult) -> None:
        filename = f"{result.observation_hash.removeprefix('sha256:')}.json"
        path = self.run_dir / "observations" / filename
        if not path.exists():
            self._write_json(path, result.to_observation())

    def write_steps(self, records: list[StepRecord]) -> None:
        for record in records:
            self.write_step(record)

    def write_step(self, record: StepRecord) -> None:
        """Validate and flush one step before the next action is requested."""

        value = record.to_dict()
        self.schemas.validate("step_record", value)
        self._append_jsonl(self.run_dir / "steps.jsonl", value)

    def rewrite_steps(self, records: list[StepRecord]) -> None:
        """Rewrite the finalized step journal after evaluator-derived labels."""

        values = [record.to_dict() for record in records]
        for value in values:
            self.schemas.validate("step_record", value)
        path = self.run_dir / "steps.jsonl"
        replaced = {value["episode_id"] for value in values}
        previous = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
        values = [value for value in previous if value["episode_id"] not in replaced] + values
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    def write_checkpoint(self, value: dict[str, Any]) -> None:
        """Durably append one replay checkpoint after a completed step."""

        self.schemas.validate("checkpoint_record", value)
        self._append_jsonl(self.run_dir / "checkpoints.jsonl", value)

    def write_evaluator(self, episode_id: str, result: ToolResult) -> str:
        filename = f"{episode_id}.json"
        path = self.run_dir / "evaluator" / filename
        self._write_json(path, result.to_observation())
        return path.relative_to(self.run_dir).as_posix()

    def write_episode(self, record: EpisodeRecord) -> None:
        value = record.to_dict()
        self.schemas.validate("episode_record", value)
        self._append_jsonl(self.run_dir / "episodes.jsonl", value)
