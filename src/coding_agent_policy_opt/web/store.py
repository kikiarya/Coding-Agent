from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any


class RunStore:
    """Small durable control-plane store.

    Artifacts remain append-only files; SQLite stores only lifecycle metadata and
    indexes so a server restart can reconstruct the API view of a run.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        repo_path TEXT NOT NULL,
                        run_dir TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        issue TEXT NOT NULL,
                        harness_policy TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        error TEXT,
                        result_json TEXT,
                        idempotency_key TEXT UNIQUE,
                        attempt_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        value["request"] = json.loads(value.pop("request_json"))
        raw_result = value.pop("result_json")
        value["result"] = json.loads(raw_result) if raw_result else None
        return value

    def create(self, value: dict[str, Any]) -> None:
        with self._lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, status, request_json, repo_path, run_dir, task_id,
                        issue, harness_policy, created_at, idempotency_key, attempt_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        value["run_id"], value["status"],
                        json.dumps(value["request"], ensure_ascii=False, sort_keys=True),
                        value["repo_path"], value["run_dir"], value["task_id"],
                        value["issue"], value["harness_policy"], value["created_at"],
                        value.get("idempotency_key"), value.get("attempt_count", 0),
                    ),
                )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._decode(row)

    def find_idempotency(self, key: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM runs WHERE idempotency_key = ?", (key,)).fetchone()
        return self._decode(row)

    def active(self) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM runs
                WHERE status NOT IN ('completed', 'failed', 'cancelled')
                ORDER BY created_at ASC LIMIT 1
                """
            ).fetchone()
        return self._decode(row)

    def update(self, run_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {
            "status", "started_at", "finished_at", "error", "result",
            "attempt_count", "repo_path", "run_dir",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown RunStore fields: {sorted(unknown)}")
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            assignments.append(f"{key if key != 'result' else 'result_json'} = ?")
            values.append(
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if key == "result" and value is not None
                else value
            )
        if not assignments:
            current = self.get(run_id)
            if current is None:
                raise KeyError(run_id)
            return current
        values.append(run_id)
        with self._lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?",
                    values,
                )
        current = self.get(run_id)
        if current is None:
            raise KeyError(run_id)
        return current

    def all(self) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at ASC").fetchall()
        return [value for row in rows if (value := self._decode(row)) is not None]

    def close(self) -> None:
        return None
