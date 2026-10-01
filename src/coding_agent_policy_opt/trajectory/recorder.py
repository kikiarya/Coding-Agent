from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.manifest import read_jsonl
from ..harness.artifacts import RunArtifactStore

__all__ = [
    "RunArtifactStore",
    "load_episode_records",
    "load_run_metadata",
    "load_step_records",
]


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir.resolve() / "run.json").read_text(encoding="utf-8"))


def load_step_records(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir.resolve() / "steps.jsonl")


def load_episode_records(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir.resolve() / "episodes.jsonl")
