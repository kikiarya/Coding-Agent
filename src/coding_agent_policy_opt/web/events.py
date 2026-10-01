from __future__ import annotations

import threading
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventBus:
    """Thread-safe, append-only event buffer for one local run."""

    def __init__(
        self,
        *,
        journal_path: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self._condition = threading.Condition()
        self._journal_path = journal_path.resolve() if journal_path else None
        self._run_id = run_id
        self._events: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if self._journal_path is None or not self._journal_path.is_file():
            return []
        values: list[dict[str, Any]] = []
        for line in self._journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "seq" in value:
                values.append(value)
        return values

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        with self._condition:
            event = {
                "event_id": f"evt_{uuid.uuid4().hex}",
                "run_id": self._run_id,
                "attempt_id": attempt_id,
                "seq": len(self._events) + 1,
                "type": event_type,
                "actor": actor,
                "schema_version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
            if self._journal_path is not None:
                self._journal_path.parent.mkdir(parents=True, exist_ok=True)
                with self._journal_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            self._events.append(event)
            self._condition.notify_all()
            return event

    def latest_seq(self) -> int:
        with self._condition:
            return len(self._events)

    def wait_for(self, after_seq: int, timeout: float = 10.0) -> list[dict[str, Any]]:
        with self._condition:
            self._condition.wait_for(
                lambda: len(self._events) > after_seq,
                timeout=max(timeout, 0.0),
            )
            return [event for event in self._events if event["seq"] > after_seq]
